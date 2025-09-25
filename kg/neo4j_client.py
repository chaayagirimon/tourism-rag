# kg/neo4j_client.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv; load_dotenv()

REVIEW_SOURCES = ["yelp", "reddit", "wikivoyage", "airbnb", "airbnb_host", "airbnb_guest"]

class KG:
    def __init__(self, uri=None, user=None, pwd=None):
        self._driver = GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI"),
            auth=(user or os.getenv("NEO4J_USER"), pwd or os.getenv("NEO4J_PASSWORD"))
        )
    def close(self): self._driver.close()
    
    def popularity(self, ids):
        if not ids: return {}
        q = """
        UNWIND $ids AS id
        MATCH (cp:CanonicalPlace {id:id})
        RETURN id AS id,
               coalesce(cp.popularity_listings_z, 0.0) AS pop_z,
               coalesce(cp.listings_nearby, 0)         AS ln,
               cp.sustainability_popularity_flag       AS flag
        """
        with self._driver.session() as s:
            return {r["id"]: dict(pop_z=r["pop_z"], ln=r["ln"], flag=r["flag"]) for r in s.run(q, ids=ids)}

    def reviews_for(self, ids, max_total_per_place=80):
        """
        Returns: { place_id: [ {src, text, rating?}, ... ] }
        Strategy:
          A) CanonicalPlace -[:HAS_REVIEW]-> Review limited to yelp/reddit/wikivoyage
          B) If none found for a pid, fallback via SourcePlace VARIANT_OF
        """
        if not ids: return {}
        def rows_to_list(rows):
            return [
                {"src": (rr.get("src") or "other"),
                 "text": (rr.get("text") or "").strip(),
                 "rating": rr.get("rating")}
                for rr in (rows or []) if (rr.get("text") or "").strip()
            ]

        out = {}
        qA = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})-[:HAS_REVIEW]->(rv:Review)
        WHERE rv.source IN $sources
        WITH pid, rv
        ORDER BY coalesce(rv.rating,0) DESC, coalesce(rv.scraped_at, datetime({epochSeconds:0})) DESC
        WITH pid, collect({src:coalesce(rv.source,'other'), text:rv.text, rating:rv.rating}) AS revs
        RETURN pid AS id, revs[0..$maxn] AS reviews
        """
        with self._driver.session() as s:
            for r in s.run(qA, ids=ids, maxn=max_total_per_place, sources=REVIEW_SOURCES):
                out[r["id"]] = rows_to_list(r["reviews"])

        missing = [pid for pid in ids if not out.get(pid)]
        if missing:
            qB = """
            UNWIND $ids AS pid
            MATCH (cp:CanonicalPlace {id:pid})<-[:VARIANT_OF]-(sp:SourcePlace)-[:HAS_REVIEW]->(rv:Review)
            WHERE rv.source IN $sources
            WITH pid, rv
            ORDER BY coalesce(rv.rating,0) DESC, coalesce(rv.scraped_at, datetime({epochSeconds:0})) DESC
            WITH pid, collect({src:coalesce(rv.source,'other'), text:rv.text, rating:rv.rating}) AS revs
            RETURN pid AS id, revs[0..$maxn] AS reviews
            """
            with self._driver.session() as s:
                for r in s.run(qB, ids=missing, maxn=max_total_per_place, sources=REVIEW_SOURCES):
                    out[r["id"]] = rows_to_list(r["reviews"])

        # Airbnb listings proxied via Listing nodes
        qC = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})<-[:NEAR]-(lst:Listing)-[:HAS_REVIEW]->(rv:Review)
        WHERE rv.source IN $sources AND coalesce(rv.text,'') <> ''
        WITH pid, rv
        ORDER BY coalesce(rv.rating,0) DESC, coalesce(rv.scraped_at, datetime({epochSeconds:0})) DESC
        WITH pid, collect({src:coalesce(rv.source,'airbnb'), text:rv.text, rating:rv.rating}) AS revs
        RETURN pid AS id, revs[0..$maxn] AS reviews
        """
        with self._driver.session() as s:
            for r in s.run(qC, ids=ids, maxn=max_total_per_place, sources=["airbnb", "airbnb_host", "airbnb_guest"]):
                cur = out.get(r["id"], [])
                cur.extend(rows_to_list(r["reviews"]))
                out[r["id"]] = cur[:max_total_per_place]

        for pid in ids:
            out.setdefault(pid, [])
        return out

    def transit_access(self, ids):
        # sample: near a Station within 500m (optional)
        q = """
        UNWIND $ids AS id
        MATCH (cp:CanonicalPlace {id:id})
        OPTIONAL MATCH (cp)-[:NEAR*1..2]->(st:Station)
        WITH id, CASE WHEN st IS NULL THEN 0.0 ELSE 1.0 END AS tb
        RETURN id, tb
        """
        with self._driver.session() as s:
            return {r["id"]: float(r["tb"]) for r in s.run(q, ids=ids)}

    def reviews_count(self, ids):
        if not ids: return {}
        q = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})-[:HAS_REVIEW]->(r:Review)
        WHERE r.source IN $sources
        RETURN pid AS id, count(r) AS n
        """
        counts = {pid: 0 for pid in ids}
        with self._driver.session() as s:
            for r in s.run(q, ids=ids, sources=REVIEW_SOURCES):
                counts[r["id"]] = counts.get(r["id"], 0) + int(r["n"])

        q_var = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})<-[:VARIANT_OF]-(sp:SourcePlace)-[:HAS_REVIEW]->(r:Review)
        WHERE r.source IN $sources
        RETURN pid AS id, count(r) AS n
        """
        with self._driver.session() as s:
            for r in s.run(q_var, ids=ids, sources=REVIEW_SOURCES):
                counts[r["id"]] = counts.get(r["id"], 0) + int(r["n"])

        q_list = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})<-[:NEAR]-(lst:Listing)-[:HAS_REVIEW]->(r:Review)
        WHERE r.source IN $sources
        RETURN pid AS id, count(r) AS n
        """
        with self._driver.session() as s:
            for r in s.run(q_list, ids=ids, sources=["airbnb", "airbnb_host", "airbnb_guest"]):
                counts[r["id"]] = counts.get(r["id"], 0) + int(r["n"])

        return counts

    def aspect_match(self, ids, query):  # (unused – kept for API compat)
        return {i:0.0 for i in ids}


    # Add to class KG
    def place_blobs(self, ids, max_reviews_per_place: int = 40) -> dict[str, str]:
        """
        Returns a short text blob per CanonicalPlace for clustering/embedding.
        Uses: name [+ categories if present] + a sample of review texts.
        """
        if not ids: return {}
        # 1) names (+ optional categories; OPTIONAL MATCH is safe if Category not in KG)
        q = """
        UNWIND $ids AS pid
        MATCH (cp:CanonicalPlace {id:pid})
        OPTIONAL MATCH (cp)-[:IN_CATEGORY]->(c:Category)
        WITH pid, cp.name AS name, collect(DISTINCT c.name) AS cats
        RETURN pid AS id, name, cats
        """
        names = {}
        cats = {}
        with self._driver.session() as s:
            for r in s.run(q, ids=ids):
                names[r["id"]] = (r.get("name") or "").strip()
                cats[r["id"]]  = [c for c in (r.get("cats") or []) if c]

        # 2) limited reviews (reuse your reviews_for)
        revs_by_id = self.reviews_for(ids, max_total_per_place=max_reviews_per_place)

        # 3) build blobs
        out = {}
        for pid in ids:
            name = names.get(pid, "")
            cat_str = ", ".join(cats.get(pid, [])) if cats.get(pid) else ""
            reviews = " ".join([r["text"] for r in revs_by_id.get(pid, [])])[:4000]
            parts = [name]
            if cat_str: parts.append(cat_str)
            if reviews: parts.append(reviews)
            out[pid] = " | ".join([p for p in parts if p])
        return out
