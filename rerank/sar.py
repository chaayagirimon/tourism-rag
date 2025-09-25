# rerank/sar.py

# --- Fix 7: density proxy for listings_nearby ---
def listings_density(meta, density_index):
    """
    meta: dict with 'lat', 'lon', 'city'
    density_index: { city -> grid or precomputed density map }
    Returns normalized density [0,1]
    """
    city = (meta.get("city") or "").lower()
    if city not in density_index:
        return 0.0
    # simplest proxy: just return city-level avg
    return density_index[city]

def sar_score(base, *,
              pop_z=None, listings_nearby=0, flag=None,
              transit_bonus=0.0, aspect_bonus=0.0, type_bonus=0.0,
              beta=0.10, gamma=0.2, delta=0.40, eta=0.02, tau=0.10, kappa=0.50,
              type_gate_penalty=0.0):
    """
    base: retrieval score (~[0,1])
    beta : penalty for popularity z
    eta  : penalty per 10 listings (overtourism proxy)
    tau  : +/- boost for sustainability flag
    gamma: transit bonus
    delta: aspect (reviews) bonus
    kappa: type-match bonus
    type_gate_penalty: typically negative when candidate is off-type
    """
    z = 0.0 if pop_z is None else max(-2.0, min(3.0, float(pop_z)))
    s = float(base) - beta*z + gamma*transit_bonus + delta*aspect_bonus + kappa*type_bonus
    s -= eta * min(150.0, float(listings_nearby)/10.0)
    if flag == "high": s += tau
    if flag == "low":  s -= tau
    s += type_gate_penalty
    return s

def apply_sar(candidates, kg_signals, transit=None, aspects=None, type_bonus=None, type_gate=None, k=10, **params):
    transit = transit or {}
    aspects = aspects or {}
    type_bonus = type_bonus or {}
    type_gate = type_gate or {}
    # inside rerank() after you have candidates, aspects, type_gate, kg_signals...
    rescored = []
    for base, rowid, meta in candidates:
        sig = kg_signals.get(meta["id"], {})
        score = sar_score(
            base,
            pop_z=sig.get("pop_z"),
            listings_nearby=sig.get("ln", 0),
            flag=sig.get("flag"),
            transit_bonus=transit.get(meta["id"], 0.0),
            aspect_bonus=aspects.get(meta["id"], 0.0),  # ← alignment
            type_bonus=0.0,                               # ← no keyword hacks
            type_gate_penalty=type_gate.get(meta["id"], 0.0),
            **params,
        )
        rescored.append((score, rowid, meta))

    rescored.sort(reverse=True, key=lambda x: x[0])
    return [m for _,_,m in rescored[:k]]
