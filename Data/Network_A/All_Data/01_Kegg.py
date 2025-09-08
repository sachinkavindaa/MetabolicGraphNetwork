# augment_using_link_endpoints.py
import re, time, urllib.request
import pandas as pd

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def kegg_link(kind, ko_list, tries=3, base_delay=0.6):
    # kind = "ec" or "reaction"
    ids = "+".join(f"ko:{k}" for k in ko_list)
    url = f"https://rest.kegg.jp/link/{kind}/{ids}"
    for t in range(1, tries+1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"[WARN] link/{kind} failed (try {t}): {e}")
            time.sleep(base_delay * t)
    return ""

# Load current table
df = pd.read_csv("All_KO_to_EC_RXN.csv")
df["EC"] = df["EC"].fillna("").astype(str)
df["RXN"] = df["RXN"].fillna("").astype(str)

kos = df["KO"].astype(str).str.strip().str.upper().tolist()
kos = [k for k in kos if re.match(r"^K\d{5}$", k)]

# Build maps KO -> set(EC), KO -> set(RXN)
ko2ec, ko2rxn = {}, {}

BATCH = 20
for batch in batched(kos, BATCH):
    txt_ec = kegg_link("ec", batch)
    txt_rx = kegg_link("reaction", batch)

    # parse "ko:K00001\tec:1.1.1.1" lines
    for line in txt_ec.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0].startswith("ko:") and parts[1].startswith("ec:"):
            ko = parts[0].split(":",1)[1]
            ec = parts[1].split(":",1)[1]
            ko2ec.setdefault(ko, set()).add(ec)

    # parse "ko:K00001\trn:Rxxxxx" lines
    for line in txt_rx.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0].startswith("ko:") and (parts[1].startswith("rn:") or parts[1].startswith("reaction:") or parts[1].startswith("R")):
            ko = parts[0].split(":",1)[1]
            rhs = parts[1]
            if ":" in rhs:
                rxn = rhs.split(":",1)[1]
            else:
                rxn = rhs
            ko2rxn.setdefault(ko, set()).add(rxn)

    time.sleep(0.3)

# Merge back: prefer existing; add what’s missing
new_ec, new_rxn = [], []
for _, row in df.iterrows():
    ko = row["KO"]
    ec_set = set(filter(None, row["EC"].split(";"))) | ko2ec.get(ko, set())
    rxn_set = set(filter(None, row["RXN"].split(";"))) | ko2rxn.get(ko, set())
    new_ec.append(";".join(sorted(ec_set)))
    new_rxn.append(";".join(sorted(rxn_set)))

df["EC"] = new_ec
df["RXN"] = new_rxn
df.to_csv("All_KO_to_EC_RXN_filled.csv", index=False)
print("[INFO] Wrote All_KO_to_EC_RXN_filled.csv")
