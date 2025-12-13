import pandas as pd
import networkx as nx
from pathlib import Path

# -----------------------------
# 1) Charger les fichiers village
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR
df_within = pd.read_csv(DATA_DIR/"scc2034_kilifi_all_contacts_within_households.csv")
df_across = pd.read_csv(DATA_DIR/"scc2034_kilifi_all_contacts_across_households.csv")

print("Within :", df_within.shape)
print("Across :", df_across.shape)

# -----------------------------
# 2) Construire des identifiants de nœuds
#    (un individu = h_m, ex : 'E_23')
# -----------------------------
def build_edges(df):
    id1 = df["h1"].astype(str) + "_" + df["m1"].astype(str)
    id2 = df["h2"].astype(str) + "_" + df["m2"].astype(str)

    return pd.DataFrame({
        "id1": id1,
        "id2": id2,
        "duration": df["duration"],
        "day": df["day"],
        "hour": df["hour"],
        "household1": df["h1"].astype(str),
        "age1": df["age1"],
        "sex1": df["sex1"],
        "household2": df["h2"].astype(str),
        "age2": df["age2"],
        "sex2": df["sex2"],
    })



edges_within = build_edges(df_within)
edges_across = build_edges(df_across)

# Fusionner toutes les interactions (within + across)
df_all = pd.concat([edges_within, edges_across], ignore_index=True)
print("Total lignes (contacts) :", df_all.shape)

# Construire les métadonnées des nœuds (1 ligne par individu)
meta_1 = df_all[["id1", "household1", "age1", "sex1"]].rename(
    columns={"id1": "id", "household1": "household", "age1": "age", "sex1": "sex"}
)

meta_2 = df_all[["id2", "household2", "age2", "sex2"]].rename(
    columns={"id2": "id", "household2": "household", "age2": "age", "sex2": "sex"}
)

meta_all = pd.concat([meta_1, meta_2], ignore_index=True).drop_duplicates(subset=["id"])

# -----------------------------
# 3) Agréger sur les 3 jours par paire (id1, id2)
# -----------------------------
def sorted_pair(row):
    a, b = row["id1"], row["id2"]
    return tuple(sorted((a, b)))   # pour graphe non orienté

df_all["pair"] = df_all.apply(sorted_pair, axis=1)

# Somme des durées par paire
df_sum = df_all.groupby("pair", as_index=False).agg({
    "duration": "sum"
})

# Récupérer id1 / id2
df_sum[["id1", "id2"]] = pd.DataFrame(df_sum["pair"].tolist(), index=df_sum.index)
df_sum = df_sum.drop(columns=["pair"])

print("Nombre de paires uniques :", df_sum.shape[0])

# -----------------------------
# 4) Construire le graphe pondéré
# -----------------------------
G_village = nx.Graph()

for _, row in df_sum.iterrows():
    i = row["id1"]
    j = row["id2"]
    w = row["duration"]
    G_village.add_edge(i, j, weight=w)

print("Graphe village :",
      G_village.number_of_nodes(), "nœuds,",
      G_village.number_of_edges(), "arêtes")

# Ajouter les métadonnées aux nœuds
for _, r in meta_all.iterrows():
    if r["id"] in G_village:
        G_village.nodes[r["id"]]["household"] = r["household"]
        G_village.nodes[r["id"]]["age"] = int(r["age"])
        G_village.nodes[r["id"]]["sex"] = r["sex"]

# -----------------------------
# 5) Calculer les centralités (critère 2)
# -----------------------------
print("\nCalcul des centralités...")

# Degré (pour info dans le tableau final)
deg_village = dict(G_village.degree())

# Betweenness (non pondérée, comme pour l'école)
bet_village = nx.betweenness_centrality(G_village, normalized=True, weight=None)

# Closeness (non pondérée)
clo_village = nx.closeness_centrality(G_village)

# PageRank (pondéré par la durée totale du contact)
pr_village = nx.pagerank(G_village, weight="weight")  # 'weight' = duration

# -----------------------------
# 6) Tableau final + export CSV
# -----------------------------
rows = []
for node in G_village.nodes():
    node_data = G_village.nodes[node]
    rows.append({
        "id": node,
        "household": node_data.get("household"),
        "age": node_data.get("age"),
        "sex": node_data.get("sex"),
        "degree": deg_village[node],
        "betweenness": bet_village[node],
        "closeness": clo_village[node],
        "pagerank": pr_village[node],
    })

df_cent_village = pd.DataFrame(rows)


# -----------------------------
# 7) Afficher les TOP 10
# -----------------------------
print("\n=== TOP 10 Betweenness (village) ===")
print(df_cent_village.sort_values("betweenness", ascending=False).head(10))

print("\n=== TOP 10 Closeness (village) ===")
print(df_cent_village.sort_values("closeness", ascending=False).head(10))

print("\n=== TOP 10 PageRank (village) ===")
print(df_cent_village.sort_values("pagerank", ascending=False).head(10))

print("\nAnalyse des centralités (village) terminée.")
