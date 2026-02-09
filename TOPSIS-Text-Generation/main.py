import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.topsis import normalize_matrix, apply_weights, calculate_ideal_solutions, calculate_scores
from src.utils import create_folders

# Create necessary folders
create_folders()

# Load decision matrix
df = pd.read_csv("data/decision_matrix.csv")

models = df["Model"]
criteria_data = df.drop(columns=["Model"])

matrix = criteria_data.values

# Define weights and impacts
weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
impacts = ["-", "+", "+", "-", "-"]

# Step 1: Normalize
normalized = normalize_matrix(matrix)
normalized_df = pd.DataFrame(normalized, columns=criteria_data.columns, index=models)
normalized_df.to_csv("results/normalized_matrix.csv")

# Step 2: Apply weights
weighted = apply_weights(normalized, weights)
weighted_df = pd.DataFrame(weighted, columns=criteria_data.columns, index=models)
weighted_df.to_csv("results/weighted_matrix.csv")

# Step 3: Ideal best and worst
ideal_best, ideal_worst = calculate_ideal_solutions(weighted, impacts)

# Step 4: Calculate TOPSIS scores
scores = calculate_scores(weighted, ideal_best, ideal_worst)

df["TOPSIS Score"] = scores
df["Rank"] = df["TOPSIS Score"].rank(ascending=False)

df.to_csv("results/topsis_scores.csv", index=False)

# =============================
# Generate Plots
# =============================

# 1. Ranking Vertical Bar Plot with Gradient Colors
fig, ax = plt.subplots(figsize=(12, 7))
sorted_indices = np.argsort(scores)[::-1]
sorted_models = models.iloc[sorted_indices]
sorted_scores = scores[sorted_indices]
colors_gradient = plt.cm.Spectral(np.linspace(0.2, 0.8, len(sorted_models)))
bars = ax.bar(sorted_models, sorted_scores, color=colors_gradient, edgecolor='black', linewidth=2, alpha=0.85)
ax.set_ylabel("TOPSIS Score", fontsize=13, fontweight='bold')
ax.set_title("TOPSIS Ranking of Text Generation Models", fontsize=15, fontweight='bold', pad=20)
ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=11)
ax.set_ylim(0, max(sorted_scores) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, v) in enumerate(zip(bars, sorted_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.4f}\nRank #{i+1}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/ranking_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Radar Chart for Model Comparison
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='polar')
angles = np.linspace(0, 2 * np.pi, len(criteria_data.columns), endpoint=False).tolist()
angles += angles[:1]
colors_radar = plt.cm.Set3(np.linspace(0, 1, len(models)))

for idx, (model, color) in enumerate(zip(models, colors_radar)):
    values = normalized_df.iloc[idx].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color)
    ax.fill(angles, values, alpha=0.2, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria_data.columns, size=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title("Normalized Criteria Comparison (Radar Chart)", fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
ax.grid(True, linewidth=1.2)
plt.tight_layout()
plt.savefig("outputs/radar_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Grouped Bar Chart for Criteria Comparison
fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(models))
width = 0.16
criteria = criteria_data.columns.tolist()
colors_grouped = plt.cm.tab10(np.linspace(0, 1, len(criteria)))

for i, (criterion, color) in enumerate(zip(criteria, colors_grouped)):
    offset = (i - len(criteria) / 2) * width
    ax.bar(x + offset, normalized_df[criterion], width, label=criterion, color=color, edgecolor='black', linewidth=0.8)

ax.set_xlabel("Models", fontsize=13, fontweight='bold')
ax.set_ylabel("Normalized Score", fontsize=13, fontweight='bold')
ax.set_title("Normalized Criteria Scores by Model (Grouped Comparison)", fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
ax.legend(title="Criteria", loc='upper right', fontsize=10, title_fontsize=11, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("outputs/criteria_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Line Plot with Markers - Criteria Trends
fig, ax = plt.subplots(figsize=(12, 7))
criteria_list = criteria_data.columns.tolist()
colors_line = plt.cm.viridis(np.linspace(0, 1, len(models)))

for idx, (model, color) in enumerate(zip(models, colors_line)):
    ax.plot(criteria_list, normalized_df.iloc[idx], marker='o', linewidth=2.5, 
            markersize=10, label=model, color=color, markerfacecolor=color, 
            markeredgecolor='white', markeredgewidth=2)

ax.set_xlabel("Criteria", fontsize=13, fontweight='bold')
ax.set_ylabel("Normalized Score", fontsize=13, fontweight='bold')
ax.set_title("Criteria Performance Trends Across Models", fontsize=15, fontweight='bold', pad=20)
ax.set_xticklabels(criteria_list, rotation=45, ha='right', fontsize=11)
ax.legend(title="Models", fontsize=10, title_fontsize=11, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("outputs/trends_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Heatmap with improved styling
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(normalized_df, annot=True, fmt='.3f', cmap="coolwarm", cbar_kws={'label': 'Normalized Score'},
            linewidths=1, ax=ax, vmin=0, vmax=1, cbar=True)
ax.set_title("Normalized Decision Matrix Heatmap", fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel("Criteria", fontsize=13, fontweight='bold')
ax.set_ylabel("Models", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nFinal Ranking:\n")
print(df.sort_values("TOPSIS Score", ascending=False))
