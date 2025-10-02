import re
import pandas as pd

KO_EC_RXN = "/work/samodha/sachin/MWAS/full_network/Fulldata_EX_RXN.csv"          # change if your paths differ
RXN_METS  = "/work/samodha/sachin/MWAS/full_network/reaction_metabolites.csv"  # change if your paths differ

def rnum(s):
    m = re.search(r"(R\d{5})", str(s).upper())
    return m.group(1) if m else None

# ----- Build enzyme–metabolite -----
ko = pd.read_csv(KO_EC_RXN)                 # expects: KO, EC, RXN
rm = pd.read_csv(RXN_METS)                  # expects: Reaction, role, metabolite

rows = []
for _, r in ko.iterrows():
    for R in re.findall(r"(R\d{5})", str(r.get("RXN","")).upper()):
        rows.append({"KO": r["KO"], "EC": r.get("EC", None), "Reaction": R})
ko_long = pd.DataFrame(rows)

rm["Reaction"] = rm["Reaction"].apply(rnum)
rm = rm.dropna(subset=["Reaction"])

E = (ko_long.merge(rm, on="Reaction", how="inner")
             .drop_duplicates()
             .loc[:, ["KO","EC","Reaction","role","metabolite"]])

enz_nodes = E[["KO","EC"]].drop_duplicates().rename(columns={"KO":"id"})
enz_nodes["type"] = "enzyme"
met_nodes = E[["metabolite"]].drop_duplicates().rename(columns={"metabolite":"id"})
met_nodes["type"] = "metabolite"
N = pd.concat([enz_nodes, met_nodes], ignore_index=True)

E.to_csv("edges_enzyme_metabolite.csv", index=False)
N.to_csv("nodes_enzyme_metabolite.csv", index=False)

# ----- Build enzyme–enzyme -----
prod = E[E["role"]=="product"][["KO","metabolite","Reaction"]].rename(
    columns={"KO":"KO_producer","Reaction":"producer_rxn"})
cons = E[E["role"]=="substrate"][["KO","metabolite","Reaction"]].rename(
    columns={"KO":"KO_consumer","Reaction":"consumer_rxn"})

EE = prod.merge(cons, on="metabolite")
EE = EE[EE["KO_producer"] != EE["KO_consumer"]].drop_duplicates()
EE.to_csv("edges_enzyme_enzyme.csv", index=False)

KON = pd.DataFrame({"id": pd.unique(pd.concat([EE["KO_producer"], EE["KO_consumer"]]))})
KON.to_csv("nodes_enzyme_enzyme.csv", index=False)

print("Wrote:",
      "edges_enzyme_metabolite.csv, nodes_enzyme_metabolite.csv,",
      "edges_enzyme_enzyme.csv, nodes_enzyme_enzyme.csv")
