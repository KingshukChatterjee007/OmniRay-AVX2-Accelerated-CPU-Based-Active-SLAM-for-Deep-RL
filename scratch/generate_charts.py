"""
Generate the OmniRay ablation study comparison table and charts
from the saved result JSONs.
"""
import json, os, glob, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = r"results\ablation_20260728_080136"
OUT_DIR = RESULTS_DIR

# ── Load manifest ─────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "ablation_manifest.json")) as fh:
    manifest = json.load(fh)

runs = manifest["runs"]

# ── Collect metrics ───────────────────────────────────────────────────────────
LABEL_MAP = {
    "full_system":          "Full System (All 5 Layers)",
    "no_health":            "No Health Monitor (−L1)",
    "no_adaptive_reward":   "No Adaptive Reward (−L2)",
    "no_meta_policy":       "No Meta-Policy (−L3)",
    "no_curriculum":        "No Curriculum (−L4)",
    "no_continual":         "No Continual Learn (−L5)",
    "no_adaptive":          "Baseline PPO (No Adaptive)",
    "no_noise_full":        "Ideal Kinematics (No Noise)",
    "entropy_with":         "Entropy ON (0.01)",
    "entropy_without":      "Entropy OFF (0.0)",
    "frontier_high":        "High Frontier Shaping (0.5)",
    "frontier_none":        "No Frontier Shaping (0.0)",
    "noise_with":           "Physical Noise ON",
    "noise_without":        "Physical Noise OFF",
}

COLOR_MAP = {
    "full_system":          "#4CAF50",   # Green — champion
    "no_health":            "#FF5252",
    "no_adaptive_reward":   "#FF5252",
    "no_meta_policy":       "#FF5252",
    "no_curriculum":        "#FF5252",
    "no_continual":         "#FF5252",
    "no_adaptive":          "#9E9E9E",   # Grey — baseline
    "no_noise_full":        "#29B6F6",   # Blue — ideal upper bound
    "entropy_with":         "#FFA726",
    "entropy_without":      "#FFA726",
    "frontier_high":        "#AB47BC",
    "frontier_none":        "#AB47BC",
    "noise_with":           "#26C6DA",
    "noise_without":        "#26C6DA",
}

records = []
for run in runs:
    cfg = run["config"]
    result_path = run.get("result_path", "")
    if not result_path or not os.path.exists(result_path):
        continue

    with open(result_path) as fh:
        d = json.load(fh)

    res = d.get("results", {})
    wall = res.get("wall_clock_seconds", 0.0)
    astats = res.get("adaptive_stats", {})
    
    # Extract key metrics
    ep_count     = astats.get("episode_count", 0)
    health_score = astats.get("health", {}).get("score", None)
    
    continual    = astats.get("continual", {})
    peak_reward  = continual.get("continual_peak_reward", None)
    recent_rew   = continual.get("continual_recent_mean_reward", None)

    curriculum   = astats.get("curriculum", {})
    difficulty   = curriculum.get("curriculum_difficulty", None)
    obstacles    = curriculum.get("curriculum_obstacles", None)

    reward_info  = astats.get("reward", {})
    collision_r  = reward_info.get("collision_rate", None)

    records.append({
        "config":        cfg,
        "label":         LABEL_MAP.get(cfg, cfg),
        "episodes":      ep_count,
        "wall_sec":      wall,
        "health_score":  health_score,
        "peak_reward":   peak_reward,
        "recent_reward": recent_rew,
        "difficulty":    difficulty,
        "obstacles":     obstacles,
        "collision_rate":collision_r,
    })

# ── Print table ───────────────────────────────────────────────────────────────
print("=" * 95)
print(f"{'Config':<24} {'Episodes':>9} {'Wall(s)':>8} {'Health':>8} {'Peak Rew':>10} {'Recent Rew':>12} {'Difficulty':>11}")
print("-" * 95)
for r in records:
    hs = f"{r['health_score']:.3f}" if r['health_score'] is not None else "N/A"
    pr = f"{r['peak_reward']:.1f}" if r['peak_reward'] is not None else "N/A"
    rr = f"{r['recent_reward']:.1f}" if r['recent_reward'] is not None else "N/A"
    diff = f"{r['difficulty']:.3f}" if r['difficulty'] is not None else "N/A"
    print(f"{r['config']:<24} {r['episodes']:>9} {r['wall_sec']:>8.1f} {hs:>8} {pr:>10} {rr:>12} {diff:>11}")
print("=" * 95)

# ── Plot 1: Peak Reward Bar Chart ─────────────────────────────────────────────
valid = [r for r in records if r["peak_reward"] is not None]
valid_sorted = sorted(valid, key=lambda x: x["peak_reward"], reverse=True)

labels = [r["config"] for r in valid_sorted]
peaks  = [r["peak_reward"] for r in valid_sorted]
colors = [COLOR_MAP.get(r["config"], "#888888") for r in valid_sorted]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(labels)), peaks, color=colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Peak Episode Reward", fontsize=12)
ax.set_title("OmniRay Ablation Study — Peak Episode Reward per Configuration", fontsize=13, fontweight="bold")
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#0f0f1a")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("#1a1a2e")

# Value labels on bars
for bar, val in zip(bars, peaks):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8, color="white")

# Legend patches
legend_patches = [
    mpatches.Patch(color="#4CAF50", label="Full System (Champion)"),
    mpatches.Patch(color="#FF5252", label="Layer Ablated"),
    mpatches.Patch(color="#9E9E9E", label="PPO Baseline"),
    mpatches.Patch(color="#29B6F6", label="Ideal Upper Bound"),
    mpatches.Patch(color="#FFA726", label="Entropy Sweep"),
    mpatches.Patch(color="#AB47BC", label="Frontier Sweep"),
    mpatches.Patch(color="#26C6DA", label="Noise Sweep"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "ablation_peak_reward_bar.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved: {out1}")

# ── Plot 2: Health Score Bar Chart ────────────────────────────────────────────
valid_h = [r for r in records if r["health_score"] is not None]
valid_h_sorted = sorted(valid_h, key=lambda x: x["health_score"], reverse=True)

labels_h = [r["config"] for r in valid_h_sorted]
scores_h = [r["health_score"] for r in valid_h_sorted]
colors_h = [COLOR_MAP.get(r["config"], "#888888") for r in valid_h_sorted]

fig2, ax2 = plt.subplots(figsize=(14, 5))
bars2 = ax2.bar(range(len(labels_h)), scores_h, color=colors_h, edgecolor="white", linewidth=0.8)
ax2.set_xticks(range(len(labels_h)))
ax2.set_xticklabels(labels_h, rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("Final Health Score [0–1]", fontsize=12)
ax2.set_title("OmniRay Ablation Study — Final Agent Health Score per Configuration", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 1.1)
ax2.axhline(0.5, color="#FFEB3B", linewidth=1.2, linestyle="--", label="Failure Threshold (0.5)")
ax2.set_facecolor("#1a1a2e")
fig2.patch.set_facecolor("#0f0f1a")
ax2.tick_params(colors="white")
ax2.yaxis.label.set_color("white")
ax2.title.set_color("white")
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for bar, val in zip(bars2, scores_h):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "ablation_health_score_bar.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {out2}")

# ── Plot 3: Layer Drop-off (Full vs each ablated layer) ───────────────────────
layer_configs = ["full_system", "no_health", "no_adaptive_reward", "no_meta_policy", "no_curriculum", "no_continual"]
layer_labels  = ["Full System", "−Health\n(L1)", "−Adapt Rew\n(L2)", "−Meta\n(L3)", "−Curriculum\n(L4)", "−Continual\n(L5)"]
layer_records = {r["config"]: r for r in records}

layer_peaks = []
for cfg in layer_configs:
    rec = layer_records.get(cfg)
    layer_peaks.append(rec["peak_reward"] if rec and rec["peak_reward"] else 0)

full_peak = layer_peaks[0]
dropoffs = [(full_peak - v) / max(full_peak, 1) * 100 for v in layer_peaks]

fig3, ax3 = plt.subplots(figsize=(10, 5))
layer_colors = ["#4CAF50"] + ["#FF5252"] * 5
bars3 = ax3.bar(range(len(layer_labels)), layer_peaks, color=layer_colors, edgecolor="white", linewidth=0.8, width=0.6)
ax3.set_xticks(range(len(layer_labels)))
ax3.set_xticklabels(layer_labels, fontsize=10)
ax3.set_ylabel("Peak Episode Reward", fontsize=12)
ax3.set_title("OmniRay — Layer-by-Layer Contribution (Peak Reward Drop-off)", fontsize=13, fontweight="bold")
ax3.set_facecolor("#1a1a2e")
fig3.patch.set_facecolor("#0f0f1a")
ax3.tick_params(colors="white")
ax3.yaxis.label.set_color("white")
ax3.title.set_color("white")
ax3.spines["bottom"].set_color("#444")
ax3.spines["left"].set_color("#444")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
for i, (bar, val, drop) in enumerate(zip(bars3, layer_peaks, dropoffs)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{val:.0f}\n({'-' if drop > 0 else '+'}{abs(drop):.1f}%)",
             ha="center", va="bottom", fontsize=8, color="white")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "ablation_layer_dropoff.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print(f"Saved: {out3}")

print("\n✅ All charts generated successfully!")
