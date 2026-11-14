"""Make the compact across-person confusion figure used in the paper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


labels = ["SIT", "STAND", "WALK", "UP", "DOWN"]
short = ["Si", "St", "W", "U", "D"]
models = ["LogisticRegression", "RandomForest"]
people = ["S01", "S02", "S03", "S04"]

counts = pd.read_csv("results/confusion_matrices.csv")
fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))

for row, model in enumerate(models):
    for col, person in enumerate(people):
        title = f"Across-person | {person} | {model} | argmax"
        part = counts[counts["experiment"] == title]
        table = part.pivot(index="true_label", columns="predicted_label", values="count")
        table = table.reindex(index=labels, columns=labels).fillna(0).astype(int)
        ax = axes[row, col]
        ax.imshow(table, cmap="Blues", vmin=0, vmax=125)
        ax.set_title(f"{person} - {'Logistic' if row == 0 else 'Forest'}", fontsize=13)
        ax.set_xticks(range(5), short, fontsize=11)
        ax.set_yticks(range(5), short, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        if col == 0:
            ax.set_ylabel("Actual", fontsize=11)

        for i in range(5):
            for j in range(5):
                value = table.iloc[i, j]
                color = "white" if value > 70 else "black"
                ax.text(j, i, str(value), ha="center", va="center", fontsize=10, color=color)

fig.tight_layout()
Path("results/figures").mkdir(exist_ok=True)
output = "results/figures/paper_across_person_confusions.png"
fig.savefig(output, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved {output}")
