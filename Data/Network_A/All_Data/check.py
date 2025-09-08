import pandas as pd, re

ko = pd.read_csv("All_KO_to_EC_RXN_final.csv")
rxns_in = (ko["RXN"].astype(str).str.split(";").explode()
           .str.replace("^rn:", "", regex=True).str.strip().str.upper())
rxns_in = rxns_in[rxns_in.str.match(r"^R\d{5}$", na=False)].drop_duplicates()

out = pd.read_csv("reaction_metabolites.csv")
parsed = out["Reaction"].dropna().astype(str).str.upper().drop_duplicates()

print("Unique reactions in KO table :", rxns_in.nunique())
print("Unique reactions parsed      :", parsed.nunique())
print("Coverage (%)                 :", round(100*parsed.nunique()/max(1, rxns_in.nunique()), 1))

print("\nTop metabolites:")
print(out["metabolite"].value_counts().head(10))
