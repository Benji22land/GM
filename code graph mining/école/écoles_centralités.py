import networkx as nx
import pandas as pd
import numpy as np

# -----------------------------
# 1) Charger le graphe
# -----------------------------
G_school = nx.read_gexf("../data/school_day1_day2_complet.gexf")
print("Graphe chargé :", G_school.number_of_nodes(), "nœuds,", G_school.number_of_edges(), "arêtes")

# -----------------------------
# 2) Charger le degré (déjà critère 1)
# -----------------------------
deg_school = dict(G_school.degree())
strength_school = dict(G_school.degree(weight="weight"))

# -----------------------------
# 3) Betweenness centrality
# -----------------------------
print("\nCalcul de la betweenness...")
bet_school = nx.betweenness_centrality(G_school, normalized=True, weight=None)

# -----------------------------
# 4) Closeness centrality
# -----------------------------
print("Calcul de la closeness...")
clo_school = nx.closeness_centrality(G_school)

# -----------------------------
# 5) PageRank (pondéré par durée)
# -----------------------------
print("Calcul du PageRank...")
pr_school = nx.pagerank(G_school, weight="weight")

# -----------------------------
# 6) Rassembler dans DataFrame
# -----------------------------
print("Construction du tableau final...")

rows = []
for node, data in G_school.nodes(data=True):
    rows.append({
        "id": node,
        "class": data.get("class"),
        "gender": data.get("sex"),
        "degree": deg_school[node],
        "strength": strength_school[node],
        "betweenness": bet_school[node],
        "closeness": clo_school[node],
        "pagerank": pr_school[node],
    })

df = pd.DataFrame(rows)



# -----------------------------
# 7) Afficher les TOP 10
# -----------------------------
pd.set_option("display.max_rows", 200)

print("\n=== TOP 10 Betweenness ===")
print(df.sort_values("betweenness", ascending=False).head(10))

print("\n=== TOP 10 Closeness ===")
print(df.sort_values("closeness", ascending=False).head(10))

print("\n=== TOP 10 PageRank ===")
print(df.sort_values("pagerank", ascending=False).head(10))

print("\nAnalyse terminée.")

#test push