import json, os, glob

results_dir = r'results\ablation_20260728_080136'
manifest_path = os.path.join(results_dir, 'ablation_manifest.json')

with open(manifest_path) as fh:
    manifest = json.load(fh)

runs = manifest['runs']
print(f"Total runs: {len(runs)}")
print()

# Show first run structure
print("=== FIRST RUN KEYS ===")
first = runs[0]
print(json.dumps(first, indent=2))
