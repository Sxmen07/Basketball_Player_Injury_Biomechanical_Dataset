"""EDA for basketball_injury.xlsx using pandas + matplotlib."""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "basketball_injury.xlsx"

# ---------- 1. Load & clean ----------
df = pd.read_excel(PATH, header=1)                      # real header is on 2nd row
df = df.dropna(axis=1, how="all")                       # drop the empty column
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include=["object", "string"]):
    df[c] = df[c].str.strip()
df["BMI"] = df["Weight_kg"] / (df["Height_cm"] / 100) ** 2

# ---------- 2. Text summary ----------
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nNumeric summary:\n", df[["Height_cm", "Weight_kg", "BMI"]].describe().round(2))
for c in ["Position", "Injury_Type", "Injury_Severity"]:
    print(f"\n{c}:\n", df[c].value_counts())
print("\nPosition x Severity:\n", pd.crosstab(df["Position"], df["Injury_Severity"]))
print("\nInjury_Type x Severity:\n", pd.crosstab(df["Injury_Type"], df["Injury_Severity"]))
print("\nCorrelation (Height, Weight):", round(df["Height_cm"].corr(df["Weight_kg"]), 3))

# ---------- 3. Plots ----------
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})
BLUE = "#3b6ea5"
sev_order = [s for s in ["Mild", "Moderate", "Severe"] if s in set(df["Injury_Severity"])] + \
            [s for s in df["Injury_Severity"].unique() if s not in ["Mild", "Moderate", "Severe"]]

# 3a. Distributions of numeric variables
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ["Height_cm", "Weight_kg", "BMI"]):
    a.hist(df[col], bins=10, color=BLUE, edgecolor="white")
    a.axvline(df[col].mean(), color="crimson", ls="--", label=f"mean {df[col].mean():.1f}")
    a.set(title=f"Distribution of {col}", xlabel=col, ylabel="Count"); a.legend()
fig.tight_layout(); fig.savefig("01_numeric_distributions.png"); plt.close(fig)

# 3b. Categorical counts
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["Position", "Injury_Type", "Injury_Severity"]):
    vc = df[col].value_counts()
    a.barh(vc.index[::-1], vc.values[::-1], color=BLUE)
    for i, v in enumerate(vc.values[::-1]): a.text(v + .2, i, str(v), va="center")
    a.set(title=f"{col} counts", xlabel="Count")
fig.tight_layout(); fig.savefig("02_categorical_counts.png"); plt.close(fig)

# 3c. Box plots: height / weight by severity
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, col in zip(ax, ["Height_cm", "Weight_kg"]):
    data = [df.loc[df["Injury_Severity"] == s, col] for s in sev_order]
    a.boxplot(data, tick_labels=sev_order, patch_artist=True,
              boxprops=dict(facecolor="#bcd2e8"), medianprops=dict(color="crimson"))
    a.set(title=f"{col} by Injury Severity", ylabel=col)
fig.tight_layout(); fig.savefig("03_boxplots_by_severity.png"); plt.close(fig)

# 3d. Height vs Weight scatter, coloured by severity
fig, ax = plt.subplots(figsize=(7, 5))
colors = dict(zip(sev_order, ["#4c9f70", "#e6a23c", "#c0392b", "#7f3b8f"]))
for s in sev_order:
    m = df["Injury_Severity"] == s
    ax.scatter(df.loc[m, "Height_cm"], df.loc[m, "Weight_kg"], label=s,
               color=colors[s], s=55, alpha=.8, edgecolor="white")
ax.set(title="Height vs Weight by Severity", xlabel="Height (cm)", ylabel="Weight (kg)")
ax.legend(title="Severity"); fig.tight_layout(); fig.savefig("04_height_weight_scatter.png"); plt.close(fig)

# 3e. Stacked bars: severity within position and injury type
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, col in zip(ax, ["Position", "Injury_Type"]):
    ct = pd.crosstab(df[col], df["Injury_Severity"])[sev_order]
    bottom = np.zeros(len(ct))
    for s in sev_order:
        a.bar(ct.index, ct[s], bottom=bottom, label=s, color=colors[s]); bottom += ct[s].values
    a.set(title=f"Severity by {col}", ylabel="Count"); a.legend(title="Severity")
    a.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig("05_severity_breakdown.png"); plt.close(fig)

# 3f. Heatmap: Injury type x Position
ct = pd.crosstab(df["Injury_Type"], df["Position"])
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(ct.values, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(ct.shape[1]), ct.columns); ax.set_yticks(range(ct.shape[0]), ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, ct.values[i, j], ha="center", va="center")
ax.set_title("Injury Type by Position"); fig.colorbar(im, label="Count")
fig.tight_layout(); fig.savefig("06_injury_position_heatmap.png"); plt.close(fig)

print("\nSaved 6 figures.")
