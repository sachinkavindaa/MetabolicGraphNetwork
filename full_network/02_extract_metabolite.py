#!/usr/bin/env python3
import re, time, random, urllib.request
import pandas as pd
from typing import List, Dict

# ---- CONFIG ----
INPUT_CSV  = "/work/samodha/sachin/MWAS/full_network/Fulldata_EX_RXN.csv"   # check filename: 'Fulldata' vs 'Full_data'
OUTPUT_CSV = "/work/samodha/sachin/MWAS/full_network/reaction_metabolites.csv"
BATCH_RN   = 10        # fetch 10 reactions per GET
BASE_DELAY = 1.0       # base delay between calls
MAX_TRIES  = 6         # exponential backoff tries
BASE_URLS  = ["https://rest.kegg.jp", "http://rest.kegg.jp"]  # try https then http

UA = {"User-Agent": "HCC-KEGG-Client/1.0"}

def http_get(url: str, tries=MAX_TRIES, base_delay=BASE_DELAY) -> str:
    """GET with polite exponential backoff & jitter; tries https then http domains."""
    last_err = None
    for base in BASE_URLS:
        full = url if url.startswith("http") else f"{base}/{url.lstrip('/')}"
        for t in range(1, tries + 1):
            try:
                req = urllib.request.Request(full, headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read().decode("utf-8", "replace")
            except Exception as e:
                last_err = e
                sleep = min(60, base_delay * (2 ** (t - 1))) + random.uniform(0, 0.5)
                print(f"[WARN] GET {full} failed (try {t}/{tries}): {e}; sleeping {sleep:.1f}s")
                time.sleep(sleep)
    print(f"[ERROR] giving up on {url}: {last_err}")
    return ""

def clean_rxn_id(r: str):
    r = (r or "").strip()
    r = r.replace("rn:", "").strip()
    if not r or r[0].upper() != "R":
        return None
    return f"R{r[1:]}"

def parse_equations_multi(txt: str) -> Dict[str, str]:
    """Parse multi-entry KEGG 'get' text → {RN: EQUATION_text}."""
    eq = {}
    current = None
    grabbing = False
    buf = []
    for line in txt.splitlines():
        if line.startswith("ENTRY"):
            # flush previous
            if current and buf:
                eq[current] = " ".join(buf).strip()
            buf = []; grabbing = False; current = None
            m = re.search(r'\b(R\d{5})\b', line)
            if m: current = m.group(1)
        elif line.startswith("EQUATION"):
            grabbing = True
            buf = [line.split("EQUATION", 1)[1].strip()]
        elif line.startswith("            "):  # continuation line (12 spaces)
            if grabbing:
                buf.append(line.strip())
        else:
            grabbing = False
    if current and buf:
        eq[current] = " ".join(buf).strip()
    return eq

def split_equation(eq: str):
    parts = re.split(r'\s*<=>\s*|\s*=>\s*', eq, maxsplit=1)
    if len(parts) != 2:
        return [], []
    left, right = parts
    def cleanse(side):
        items = [x.strip() for x in side.split('+')]
        items = [re.sub(r'^[0-9]+(?:\.[0-9]+)?\s*', '', x).strip() for x in items]
        return [x for x in items if x]
    return cleanse(left), cleanse(right)

# ---- LOAD unique reaction IDs ----
rxns_raw = pd.read_csv(INPUT_CSV)["RXN"]
rxns = (
    rxns_raw.astype(str)
            .str.split(";")
            .explode()
            .dropna()
            .map(str.strip)
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
)

rids: List[str] = []
for r in rxns:
    rclean = clean_rxn_id(r)
    if rclean:
        rids.append(rclean)
rids = sorted(set(rids))
print(f"[INFO] unique reactions: {len(rids)}")

# ---- FETCH in batches with backoff ----
rows = []
for i in range(0, len(rids), BATCH_RN):
    chunk = rids[i:i + BATCH_RN]
    query = "get/" + "+".join(f"rn:{x}" for x in chunk)
    txt = http_get(query)
    if not txt:
        continue
    eq_map = parse_equations_multi(txt)
    for rn, eq in eq_map.items():
        subs, prods = split_equation(eq)
        for s in subs:
            rows.append({"Reaction": rn, "role": "substrate", "metabolite": s})
        for p in prods:
            rows.append({"Reaction": rn, "role": "product", "metabolite": p})
    time.sleep(BASE_DELAY)  # polite gap between batches

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved {len(df)} rows for {df['Reaction'].nunique() if not df.empty else 0} reactions → {OUTPUT_CSV}")
