import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# LOSO 결과
loso = pd.read_csv("g:/MI_opendata/results/loso_results_20260330_195135.csv")  # 실제 파일명

# ERD summary
erd  = pd.read_csv("g:/MI_opendata/results/erd_summary_cho2017_20260331_124652.csv")
erd_mu = erd[erd["band"] == "mu"][["subject", "erd_mean"]]

# 합치기
merged = loso[loso["dataset"] == "cho2017"].merge(erd_mu, on="subject")

r, p = pearsonr(merged["erd_mean"], merged["acc"])
print(f"ERD vs Accuracy: r={r:.3f}, p={p:.3f}")

plt.scatter(merged["erd_mean"], merged["acc"] * 100)
plt.xlabel("mu ERD strength (%)")
plt.ylabel("LOSO Accuracy (%)")
plt.title(f"ERD strength vs Classification accuracy (r={r:.2f})")
plt.axhline(50, color='gray', linestyle='--')
plt.savefig("g:/MI_opendata/results/erd_vs_acc.png", dpi=150)
