# retrieval/build_index.py

import os
import json
import sqlite3
import pathlib
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from kg.neo4j_client import KG

load_dotenv()

CARDS_PATH = pathlib.Path(os.getenv("POI_CARDS", "./data/poi_cards.json"))
STORE_DIR  = pathlib.Path("./retrieval/store")
INDEX_PATH = STORE_DIR / "index.faiss"
META_DB    = STORE_DIR / "meta.db"
CLEAN_REBUILD = os.getenv("CLEAN_REBUILD", "0") == "1"

def text_of(card: dict) -> str:
    parts = [str(card.get("name", "")), str(card.get("city", ""))]
    kind = card.get("kind") or "poi"
    if kind == "airbnb":
        summary = []
        room_type = card.get("room_type")
        if room_type:
            summary.append(str(room_type))
        accommodates = card.get("accommodates")
        if accommodates:
            summary.append(f"accommodates {int(accommodates)} guests")
        bedrooms = card.get("bedrooms")
        if bedrooms:
            summary.append(f"{bedrooms} bedrooms")
        bathrooms = card.get("bathrooms")
        if bathrooms:
            summary.append(f"{bathrooms} bathrooms")
        price = card.get("lodging_price")
        if price:
            summary.append(f"nightly price {price}")
        if summary:
            parts.append(" | ".join(summary))
        desc = card.get("airbnb_description")
        if desc:
            parts.append(str(desc))
    parts += (card.get("wv") or [])
    for q in (card.get("quotes") or [])[:4]:
        try:
            parts.append(f"{q.get('src','?')}: {q.get('txt','')}")
        except Exception:
            pass
    return "\n".join([p for p in parts if p])

def load_cards(path: pathlib.Path):
    cards = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            card = obj["card"] if "card" in obj else obj
            cards.append(card)
    return cards


def load_airbnb_cards(max_reviews: int = 20) -> list:
    """Fetch Airbnb listings from the KG and adapt them to the card schema."""
    try:
        kg = KG()
    except Exception:
        return []

    q = """
    MATCH (l:Listing)-[:IN_CITY]->(city:City)
    OPTIONAL MATCH (l)-[:HAS_REVIEW]->(rv:Review)
    WITH l, city, collect(rv)[0..$max_reviews] AS reviews
    RETURN l.id AS id,
           coalesce(l.name, "Unnamed stay") AS name,
           coalesce(city.slug, l.city_slug) AS city_slug,
           coalesce(city.name, l.city_slug) AS city_name,
           l.room_type AS room_type,
           l.accommodates AS accommodates,
           l.bedrooms AS bedrooms,
           l.bathrooms AS bathrooms,
           l.price AS price,
           l.description AS description,
           [r IN reviews | {src: coalesce(r.source,'airbnb'), text: r.text, rating: r.rating}] AS reviews
    """

    listings = []
    try:
        with kg._driver.session() as session:
            for rec in session.run(q, max_reviews=max_reviews):
                rec = dict(rec)
                listing_id = rec.get("id")
                name = rec.get("name")
                city_slug = rec.get("city_slug")
                city_name = rec.get("city_name")
                def _to_int(val):
                    if val is None or val == "":
                        return None
                    try:
                        return int(float(val))
                    except (TypeError, ValueError):
                        return None

                def _to_float(val):
                    if val is None or val == "":
                        return None
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None

                bedrooms = _to_float(rec.get("bedrooms"))
                bathrooms = _to_float(rec.get("bathrooms"))
                accommodates = _to_int(rec.get("accommodates"))
                price = _to_float(rec.get("price"))

                card = {
                    "id": f"listing:{listing_id}",
                    "name": name,
                    "city": (city_slug or city_name or "").lower() if city_slug or city_name else "",
                    "city_slug": city_slug or (city_name.lower() if city_name else None),
                    "kind": "airbnb",
                    "room_type": rec.get("room_type"),
                    "accommodates": accommodates,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "lodging_price": price,
                    "review_count": 0,
                    "source": "airbnb",
                    "popularity_z": 0.0,
                    "sustainability_popularity_flag": None,
                }

                description = rec.get("description")
                if description:
                    card["airbnb_description"] = description

                quotes = []
                if description:
                    quotes.append({"src": "airbnb_desc", "txt": description})
                revs = []
                for rv in rec.get("reviews", []):
                    txt = (rv or {}).get("text")
                    if txt:
                        revs.append({"src": (rv or {}).get("src", "airbnb"), "txt": txt})
                card["review_count"] = len(revs)
                if revs:
                    quotes.extend(revs[:5])
                if quotes:
                    card["quotes"] = quotes

                wv = []
                if rec.get("room_type"):
                    wv.append(str(rec.get("room_type")))
                if accommodates:
                    wv.append(f"accommodates {accommodates} guests")
                if bedrooms:
                    wv.append(f"{bedrooms} bedrooms")
                if bathrooms:
                    wv.append(f"{bathrooms} bathrooms")
                if price:
                    wv.append(f"nightly price {price}")
                if wv:
                    card["wv"] = wv

                listings.append(card)
    finally:
        try:
            kg.close()
        except Exception:
            pass

    return listings

def nuke_store_if_requested():
    if CLEAN_REBUILD and STORE_DIR.exists():
        import shutil
        shutil.rmtree(STORE_DIR)

def ensure_dir():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

def open_db():
    # Ensure any stale WAL/SHM are gone if starting fresh
    for suffix in ("", "-wal", "-shm"):
        p = pathlib.Path(str(META_DB) + suffix)
        if CLEAN_REBUILD and p.exists():
            p.unlink()
    conn = sqlite3.connect(str(META_DB))
    conn.row_factory = sqlite3.Row
    # Avoid WAL journals to reduce stray files / corruption risk
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.DatabaseError:
        pass
    return conn

def ensure_cards_schema(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS cards (
      rowid INTEGER PRIMARY KEY,
      id    TEXT,
      name  TEXT,
      city  TEXT,
      pop_z REAL,
      flag  TEXT
    )
    """)
    # migrate: add 'text' column if missing
    cols = {r[1] for r in c.execute("PRAGMA table_info(cards)")}
    if "text" not in cols:
        c.execute("ALTER TABLE cards ADD COLUMN text TEXT")
        cols.add("text")

    extra_cols = {
        "kind": "TEXT",
        "room_type": "TEXT",
        "accommodates": "INTEGER",
        "bedrooms": "REAL",
        "bathrooms": "REAL",
        "lodging_price": "REAL",
        "review_count": "INTEGER",
        "city_slug": "TEXT"
    }
    for col, coltype in extra_cols.items():
        if col not in cols:
            c.execute(f"ALTER TABLE cards ADD COLUMN {col} {coltype}")
    conn.commit()

def populate_cards(conn: sqlite3.Connection, cards, texts):
    # keep FAISS row ids aligned with SQLite row ids
    c = conn.cursor()
    c.execute("DELETE FROM cards")
    ins_sql = (
        "INSERT INTO cards(rowid,id,name,city,city_slug,pop_z,flag,kind,room_type,accommodates,"
        "bedrooms,bathrooms,lodging_price,review_count,text) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    for i, card in enumerate(cards):
        kind = card.get("kind") or "poi"
        city_slug = card.get("city_slug") or (card.get("city") or "").lower()
        accommodates = card.get("accommodates")
        bedrooms = card.get("bedrooms")
        bathrooms = card.get("bathrooms")
        price = card.get("lodging_price") or card.get("price")
        review_count = card.get("review_count")
        room_type = card.get("room_type")
        c.execute(
            ins_sql,
            (
                i,
                card.get("id"),
                card.get("name"),
                card.get("city"),
                city_slug,
                float(card.get("popularity_z") or 0.0),
                card.get("sustainability_popularity_flag"),
                kind,
                room_type,
                int(accommodates) if accommodates is not None else None,
                float(bedrooms) if bedrooms is not None else None,
                float(bathrooms) if bathrooms is not None else None,
                float(price) if price is not None else None,
                int(review_count) if review_count is not None else None,
                texts[i],
            ),
        )
    conn.commit()

def rebuild_fts_contentless(conn: sqlite3.Connection):
    """
    Build a contentless FTS table and insert rows with explicit rowids matching 'cards'.
    This avoids 'content=' linkage and the shadow-table corruption you hit.
    """
    c = conn.cursor()
    # Drop and recreate to ensure a clean slate every run
    try:
        c.execute("DROP TABLE IF EXISTS cards_fts")
        conn.commit()
    except sqlite3.DatabaseError:
        # If DROP itself errors due to corruption, recreate the DB file
        conn.close()
        if META_DB.exists(): META_DB.unlink()
        new = open_db()
        try:
            ensure_cards_schema(new)
            return rebuild_fts_contentless(new)
        finally:
            new.close()
        return

    try:
        c.execute("CREATE VIRTUAL TABLE cards_fts USING fts5(text)")
    except sqlite3.OperationalError:
        # FTS5 not available in this SQLite build; skip silently
        return False

    # Populate with explicit rowid so we can join on it later
    cur = conn.execute("SELECT rowid, text FROM cards ORDER BY rowid ASC")
    rows = cur.fetchmany(1000)
    while rows:
        c.executemany("INSERT INTO cards_fts(rowid, text) VALUES (?, ?)", rows)
        rows = cur.fetchmany(1000)
    conn.commit()
    return True

def main():
    nuke_store_if_requested()

    if not CARDS_PATH.exists():
        raise FileNotFoundError(f"POI_CARDS not found: {CARDS_PATH}")

    print(f"[build_index] Loading cards from: {CARDS_PATH}")
    cards = load_cards(CARDS_PATH)
    print(f"[build_index] Loaded {len(cards)} cards")

    airbnb_cards = load_airbnb_cards()
    if airbnb_cards:
        existing_ids = {c.get("id") for c in cards}
        fresh_cards = [c for c in airbnb_cards if c.get("id") not in existing_ids]
        cards.extend(fresh_cards)
        print(f"[build_index] Added {len(fresh_cards)} Airbnb listings")
    else:
        print("[build_index] No Airbnb listings added (none found or KG unavailable)")

    ensure_dir()

    # Build embeddings (normalized for cosine via IP)
    texts = [text_of(c) for c in cards]
    from llm.central_model import get_st_model
    model = get_st_model()

    embs  = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    # FAISS IP index (cosine with normalized vectors)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, str(INDEX_PATH))
    print(f"[build_index] Wrote FAISS index: {INDEX_PATH} (ntotal={idx.ntotal})")

    # SQLite meta
    conn = open_db()
    try:
        ensure_cards_schema(conn)
        populate_cards(conn, cards, texts)
        fts_ok = rebuild_fts_contentless(conn)
        if fts_ok:
            print("[build_index] Rebuilt contentless FTS5 index over text")
        else:
            print("[build_index] FTS5 unavailable; skipping BM25/FTS. Vector search will still work.")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print(f"[build_index] Wrote meta rows: {len(cards)} to {META_DB}")
    print("[build_index] Done.")

if __name__ == "__main__":
    main()
