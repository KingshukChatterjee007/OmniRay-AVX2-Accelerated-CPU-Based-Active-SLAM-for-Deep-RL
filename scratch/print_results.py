import json, os, glob

results_dir = r'results\ablation_20260728_080136'
jsons = glob.glob(os.path.join(results_dir, '*_results.json'))

rows = []
for f in sorted(jsons):
    name = os.path.basename(f).replace('_seed42_results.json','')
    with open(f) as fh:
        d = json.load(fh)
    ep_rew = d.get('mean_episode_reward', d.get('final_ep_rew_mean', 'N/A'))
    coverage = d.get('mean_coverage', d.get('final_coverage', 'N/A'))
    rows.append((name, ep_rew, coverage))

header = f"{'Config':<22} {'Mean Reward':>14} {'Mean Coverage':>15}"
print(header)
print('-'*53)
for name, rew, cov in rows:
    print(f"{name:<22} {str(rew):>14} {str(cov):>15}")

# Also print raw keys of first json to understand structure
print("\n--- RAW KEYS (first file) ---")
with open(sorted(jsons)[0]) as fh:
    d = json.load(fh)
for k, v in d.items():
    print(f"  {k}: {v}")
