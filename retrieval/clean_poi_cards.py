# retrieval/clean_poi_cards.py
import json, re, sys

def plausible(card):
    name, cid = (card.get("name","").lower(), card.get("id","").lower())
    city = (card.get("city","") or "").lower()
    ok_name = any(tok in cid for tok in re.findall(r"[a-z]+", name)[:3])
    ok_city = city in cid if city else True
    return ok_name and ok_city

def clean_quotes(quotes):
    seen=set(); out=[]
    for q in quotes or []:
        txt = re.sub(r"\s+", " ", q.get("txt","")).strip()
        key=(q.get("src","?"), txt[:80])
        if not txt or key in seen: continue
        seen.add(key); out.append({"src":q.get("src","?"), "txt":txt})
    return out[:4]

def clean_wv(wv):
    # keep only short, alphanumeric-ish references
    out=[]
    for s in (wv or []):
        s = re.sub(r"\[|\]", "", s)
        if 3 <= len(s) <= 80:
            out.append(s.strip())
    return out[:3]

for line in sys.stdin:
    obj = json.loads(line)
    card = obj["card"] if "card" in obj else obj
    if not plausible(card): 
        continue
    card["quotes"] = clean_quotes(card.get("quotes"))
    card["wv"]     = clean_wv(card.get("wv"))
    print(json.dumps({"card":card}))
