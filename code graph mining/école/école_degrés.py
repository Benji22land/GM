import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Charger le graphe
G_school = nx.read_gexf("../data/school_day1_day2_complet.gexf")


# 1. Degré non pondéré
deg_school = dict(G_school.degree())
deg_vals = np.array(list(deg_school.values()))

print("---- Degré ----")
print("Degré moyen :", deg_vals.mean())
print("Min, max :", deg_vals.min(), deg_vals.max())
print("Coefficient de variation :", deg_vals.std() / deg_vals.mean())

# 2. Degré pondéré (force)
strength_school = dict(G_school.degree(weight="weight"))
strength_vals = np.array(list(strength_school.values()))

print("\n---- Force (durée cumulée) ----")
print("Force moyenne :", strength_vals.mean())
print("Min, max :", strength_vals.min(), strength_vals.max())
print("CV :", strength_vals.std() / strength_vals.mean())

# 3. Histogramme du degré
plt.figure()
plt.hist(deg_vals, bins=20)
plt.xlabel("Degré")
plt.ylabel("Nombre de nœuds")
plt.title("Distribution du degré – École (Jours 1 et 2)")
plt.tight_layout()
plt.savefig("degree_school_day1_day2_complet.png")  # l'image sera dans ton dossier du script


