import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- 0) Charger les fichiers ----------

# Mets-les dans data/ avec ces noms :
#   scc2034_kilifi_all_contacts_within_households.csv
#   scc2034_kilifi_all_contacts_across_households.csv

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR
df_within = pd.read_csv(DATA_DIR/"scc2034_kilifi_all_contacts_within_households.csv")
df_across = pd.read_csv(DATA_DIR/"scc2034_kilifi_all_contacts_across_households.csv")

print("Within :", df_within.shape)
print("Across :", df_across.shape)
print("Colonnes :", df_within.columns.tolist())


# ---------- 1) Construire des identifiants de nœuds ----------

def build_edges(df):
    # id unique par individu : "household_member"
    id1 = df["h1"].astype(str) + "_" + df["m1"].astype(str)
    id2 = df["h2"].astype(str) + "_" + df["m2"].astype(str)

    return pd.DataFrame({
        "id1": id1,
        "id2": id2,
        "duration": df["duration"],   # durée du contact en secondes
        "day": df["day"],
        "hour": df["hour"],
    })


edges_within = build_edges(df_within)
edges_across = build_edges(df_across)

# Fusionner toutes les interactions
df_all = pd.concat([edges_within, edges_across], ignore_index=True)
print("Total lignes (contacts) :", df_all.shape)


# ---------- 2) Agréger sur les 3 jours (comme dans le papier) ----------

# Pour un graphe non orienté, on trie les paires (id1, id2)
def sorted_pair(row):
    a, b = row["id1"], row["id2"]
    return tuple(sorted((a, b)))

df_all["pair"] = df_all.apply(sorted_pair, axis=1)

# Somme des durées par paire sur les 3 jours
df_sum = df_all.groupby("pair", as_index=False).agg({
    "duration": "sum"
})

# Séparer à nouveau en id1 / id2
df_sum[["id1", "id2"]] = pd.DataFrame(df_sum["pair"].tolist(), index=df_sum.index)
df_sum = df_sum.drop(columns=["pair"])

print("Nombre de paires uniques :", df_sum.shape[0])


# ---------- 3) Construire le graphe pondéré ----------

G_village = nx.Graph()

for _, row in df_sum.iterrows():
    i = row["id1"]
    j = row["id2"]
    w = row["duration"]
    G_village.add_edge(i, j, weight=w)

print("Graphe village :",
      G_village.number_of_nodes(), "nœuds,",
      G_village.number_of_edges(), "arêtes")


# ---------- 4) Distribution des degrés (critère 1) ----------

deg_village = dict(G_village.degree())
deg_vals = np.array(list(deg_village.values()))

print("\n---- Degré (village) ----")
print("Degré moyen :", deg_vals.mean())
print("Min, max :", deg_vals.min(), deg_vals.max())
print("Coefficient de variation :", deg_vals.std() / deg_vals.mean())

# Histogramme
plt.figure()
plt.hist(deg_vals, bins=20)
plt.xlabel("Degré (nombre de voisins distincts)")
plt.ylabel("Nombre de nœuds")
plt.title("Distribution du degré – Village (3 jours)")
plt.tight_layout()
plt.savefig("degre_village.png")
plt.close()
print("Histogramme sauvegardé sous 'degre_village.png'")