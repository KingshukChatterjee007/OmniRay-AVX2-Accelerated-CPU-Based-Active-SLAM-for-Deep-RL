import os
import json
import numpy as np
import matplotlib.pyplot as plt

results_dir = r"results\ablation_20260728_125451"
manifest_path = os.path.join(results_dir, "ablation_manifest.json")

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

runs = manifest.get("runs", [])
grouped = {}

for r in runs:
    cfg = r["config"]
    if cfg not in grouped:
        grouped[cfg] = []
    
    # Load individual result json file
    res_path = r.get("result_path")
    if res_path and os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as rf:
            rdata = json.load(rf)
    else:
        rdata = {}
    
    astats = rdata.get("results", {}).get("adaptive_stats", {})
    health = astats.get("health", {}).get("score", None)
    continual = astats.get("continual", {})
    peak = continual.get("continual_peak_reward", None)
    recent = continual.get("continual_recent_mean_reward", None)
    col = astats.get("reward", {}).get("collision_rate", None)
    wall = rdata.get("results", {}).get("wall_clock_seconds", 0)
    
    grouped[cfg].append({
        "seed": r["seed"],
        "health": health,
        "peak": peak,
        "recent": recent,
        "collision": col,
        "wall_clock": wall
    })

print("\n" + "=" * 90)
print("  OmniRay 3-Seed Ablation Matrix — Statistical Summary (Mean ± Std Dev)")
print("=" * 90)
print(f"{'Config':<20} | {'Health Score':<20} | {'Peak Reward':<22} | {'Recent Mean Rew':<20}")
print("-" * 90)

configs_ordered = ["full_system", "no_health", "no_adaptive_reward", "no_meta_policy", "no_curriculum", "no_continual"]
labels_ordered = ["Full System", "-Health (L1)", "-Adapt Rew (L2)", "-Meta Policy (L3)", "-Curriculum (L4)", "-Continual (L5)"]

stats = {}

for cfg in configs_ordered:
    item_runs = grouped.get(cfg, [])
    healths = [x["health"] for x in item_runs if x["health"] is not None]
    peaks = [x["peak"] for x in item_runs if x["peak"] is not None]
    recents = [x["recent"] for x in item_runs if x["recent"] is not None]
    
    h_m, h_s = (np.mean(healths), np.std(healths)) if healths else (0.0, 0.0)
    p_m, p_s = (np.mean(peaks), np.std(peaks)) if peaks else (0.0, 0.0)
    r_m, r_s = (np.mean(recents), np.std(recents)) if recents else (0.0, 0.0)
    
    stats[cfg] = {
        "h_m": h_m, "h_s": h_s,
        "p_m": p_m, "p_s": p_s,
        "r_m": r_m, "r_s": r_s
    }
    
    h_str = f"{h_m:.3f} ± {h_s:.3f}" if healths else "N/A"
    p_str = f"{p_m:.1f} ± {p_s:.1f}" if peaks else "N/A"
    r_str = f"{r_m:.1f} ± {r_s:.1f}" if recents else "N/A"
    
    print(f"{cfg:<20} | {h_str:<20} | {p_str:<22} | {r_str:<20}")

print("=" * 90)

# Set dark theme for charts
plt.style.use("dark_background")

# 1. Error Bar Chart: Peak Reward across 3 Seeds
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d1a")
ax.set_facecolor("#0d0d1a")

x = np.arange(len(configs_ordered))
p_means = [stats[c]["p_m"] for c in configs_ordered]
p_stds = [stats[c]["p_s"] for c in configs_ordered]

bars = ax.bar(x, p_means, yerr=p_stds, capsize=5, color=["#00ff88" if c == "full_system" else "#ff5555" for c in configs_ordered], alpha=0.85, edgecolor="white", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(labels_ordered, rotation=15, color="white", fontsize=10)
ax.set_ylabel("Peak Episode Reward (3-Seed Mean ± Std)", color="white", fontsize=11)
ax.set_title("OmniRay — 3-Seed Ablation Matrix: Peak Reward with Error Bars", color="white", fontsize=13, fontweight="bold", pad=15)
ax.grid(axis="y", color="#333355", linestyle="--", alpha=0.5)
ax.tick_params(colors="white")

for bar, mean, std in zip(bars, p_means, p_stds):
    if mean > 0:
        ax.text(bar.get_x() + bar.get_width()/2.0, mean + std + 40, f"{mean:.0f}±{std:.0f}", ha="center", va="bottom", color="white", fontweight="bold", fontsize=9)

plt.tight_layout()
peak_chart_path = os.path.join(results_dir, "ablation_3seed_peak_reward_errorbars.png")
plt.savefig(peak_chart_path, dpi=150, facecolor="#0d0d1a")
plt.close()
print(f"\nSaved chart: {peak_chart_path}")

# 2. Error Bar Chart: Health Score across 3 Seeds
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d1a")
ax.set_facecolor("#0d0d1a")

h_means = [stats[c]["h_m"] for c in configs_ordered]
h_stds = [stats[c]["h_s"] for c in configs_ordered]

bars = ax.bar(x, h_means, yerr=h_stds, capsize=5, color=["#00ff88" if c == "full_system" else "#ff9f43" if c == "no_health" else "#54a0ff" for c in configs_ordered], alpha=0.85, edgecolor="white", linewidth=1.2)
ax.axhline(y=0.5, color="#ff6b6b", linestyle="--", linewidth=1.5, label="Failure Threshold (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(labels_ordered, rotation=15, color="white", fontsize=10)
ax.set_ylabel("Final Agent Health Score (3-Seed Mean ± Std)", color="white", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("OmniRay — 3-Seed Ablation Matrix: Health Score per Layer Config", color="white", fontsize=13, fontweight="bold", pad=15)
ax.grid(axis="y", color="#333355", linestyle="--", alpha=0.5)
ax.tick_params(colors="white")
ax.legend(loc="upper right", facecolor="#0d0d1a", labelcolor="white")

for bar, mean, std in zip(bars, h_means, h_stds):
    if mean > 0:
        ax.text(bar.get_x() + bar.get_width()/2.0, mean + std + 0.02, f"{mean:.3f}", ha="center", va="bottom", color="white", fontweight="bold", fontsize=9)

plt.tight_layout()
health_chart_path = os.path.join(results_dir, "ablation_3seed_health_score_errorbars.png")
plt.savefig(health_chart_path, dpi=150, facecolor="#0d0d1a")
plt.close()
print(f"Saved chart: {health_chart_path}")
