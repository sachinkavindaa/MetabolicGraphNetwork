# pip install bioservices pandas
from bioservices import KEGG
import pandas as pd

kos = [k.strip() for k in open("ko_list.txt")]
k = KEGG()

rows = []
for ko in kos:
    txt = k.get(f"ko:{ko}") or ""
    ec = []
    rxn = []
    for line in txt.splitlines():
        if line.startswith("EC number"):
            # One or multiple ECs can appear; keep whole field
            ec.append(line.split(None, 2)[-1])
        if line.startswith("REACTION"):
            rxn.append(line.split(None, 1)[-1])
    rows.append({"KO": ko, "EC": ";".join(ec), "RXN": ";".join(rxn)})

pd.DataFrame(rows).to_csv("ko_to_ec_rxn.csv", index=False)