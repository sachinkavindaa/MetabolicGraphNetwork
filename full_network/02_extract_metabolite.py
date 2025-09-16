import re
import time
import pandas as pd
from bioservices import KEGG

# ---- CONFIG ----
INPUT_CSV = "/work/samodha/sachin/MWAS/Final_Graph/Test/Full_data_EX_RXN.csv"   # <- change if needed
OUTPUT_CSV = "/work/samodha/sachin/MWAS/Final_Graph/Test/reaction_metabolites.csv"
DELAY = 0.8     # polite delay between requests
RETRIES = 3

k = KEGG()

def clean_rxn_id(r: str):
    """Return 'rn:Rxxxxx' or None if it doesn't look like a KEGG reaction id."""
    r = (r or "").strip()
    r = r.replace("rn:", "").strip()
    if not r or r[0].upper() != "R":
        return None
    # Keep uppercase R + digits as-is
    return f"rn:{r.upper()}"

def fetch_parsed(entry_id: str, retries=RETRIES, delay=DELAY):
    """Fetch entry text and parse safely; return dict or None."""
    for attempt in range(1, retries + 1):
        res = k.get(entry_id)
        if not isinstance(res, str) or "Not Found" in res or res.strip() == "":
            time.sleep(delay)
            continue
        try:
            return k.parse(res)
        except Exception:
            time.sleep(delay)
    return None

def as_equation_text(eq_field):
    """KEGG parse may return str or list; normalize to one string."""
    if eq_field is None:
        return ""
    if isinstance(eq_field, list):
        return " ".join(str(x) for x in eq_field)
    return str(eq_field)

def split_equation(eq: str):
    """Split KEGG equation into substrates and products."""
    parts = re.split(r'\s*<=>\s*|\s*=>\s*', eq, maxsplit=1)
    if len(parts) != 2:
        return [], []
    left, right = parts
    def cleanse(side):
        items = [x.strip() for x in side.split('+')]
        # remove leading stoichiometry like '2 H2O' or '0.5 C00007'
        items = [re.sub(r'^[0-9]+(?:\.[0-9]+)?\s*', '', x).strip() for x in items]
        return [x for x in items if x]
    return cleanse(left), cleanse(right)

# ---- LOAD REACTIONS FROM YOUR CSV ----
rxns_raw = pd.read_csv(INPUT_CSV)["RXN"]

rxns = (
    rxns_raw.astype(str)
            .str.split(";")
            .explode()
            .dropna()
            .map(lambda s: s.strip())
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
)

rows = []
for idx, r in enumerate(rxns, 1):
    rid = clean_rxn_id(r)
    if rid is None:
        continue

    parsed = fetch_parsed(rid)
    if not parsed:
        # Couldn’t fetch/parse this reaction; skip
        continue

    eq = as_equation_text(parsed.get("EQUATION"))
    if not eq:
        continue

    subs, prods = split_equation(eq)
    # store without 'rn:' for readability
    r_clean = rid.replace("rn:", "")
    for s in subs:
        rows.append({"Reaction": r_clean, "role": "substrate", "metabolite": s})
    for p in prods:
        rows.append({"Reaction": r_clean, "role": "product", "metabolite": p})

    # small polite pause every few requests
    if idx % 10 == 0:
        time.sleep(DELAY)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows for {df['Reaction'].nunique() if not df.empty else 0} reactions → {OUTPUT_CSV}")
