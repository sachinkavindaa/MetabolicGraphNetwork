import urllib.request, time, re
import pandas as pd

# Start from the "filled" file
df = pd.read_csv("All_KO_to_EC_RXN_filled.csv")
df["EC"] = df["EC"].fillna("").astype(str)
df["RXN"] = df["RXN"].fillna("").astype(str)

# Which KOs still missing?
kos_missing_ec = df.loc[df["EC"] == "", "KO"].tolist()
kos_missing_rxn = df.loc[df["RXN"] == "", "KO"].tolist()
kos_missing = sorted(set(kos_missing_ec) | set(kos_missing_rxn))

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def kegg_link(kind, ko_list, tries=3, base_delay=0.6):
    ids = "+".join(f"ko:{k}" for k in ko_list)
    url = f"https://rest.kegg.jp/link/{kind}/{ids}"
    for t in range(1, tries+1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"[WARN] {kind} fetch failed for batch: {e}")
            time.sleep(base_delay * t)
    return ""

ko2ec, ko2rxn = {}, {}
BATCH = 20

for batch in batched(kos_missing, BATCH):
    txt_ec = kegg_link("ec", batch)
    txt_rx = kegg_link("reaction", batch)
    for line in txt_ec.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0].startswith("ko:") and parts[1].startswith("ec:"):
            ko = parts[0].split(":",1)[1]
            ec = parts[1].split(":",1)[1]
            ko2ec.setdefault(ko, set()).add(ec)
    for line in txt_rx.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0].startswith("ko:"):
            ko = parts[0].split(":",1)[1]
            rhs = parts[1].split(":",1)[-1]
            ko2rxn.setdefault(ko, set()).add(rhs)
    time.sleep(0.3)

# Merge back
new_ec, new_rxn = [], []
for _, row in df.iterrows():
    ko = row["KO"]
    ec_set = set(filter(None, row["EC"].split(";"))) | ko2ec.get(ko, set())
    rxn_set = set(filter(None, row["RXN"].split(";"))) | ko2rxn.get(ko, set())
    new_ec.append(";".join(sorted(ec_set)))
    new_rxn.append(";".join(sorted(rxn_set)))

df["EC"] = new_ec
df["RXN"] = new_rxn

df.to_csv("All_KO_to_EC_RXN_final.csv", index=False)
print(f"[INFO] Wrote All_KO_to_EC_RXN_final.csv with {len(df)} rows")
