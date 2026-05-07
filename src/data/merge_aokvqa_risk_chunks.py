"""Merge A-OKVQA_risk_chunk*.json into A-OKVQA_risk.json."""
import json, glob, os
SRC = "./data/risk"
chunk_files = sorted(glob.glob(os.path.join(SRC, 'A-OKVQA_risk_chunk*.json')))
print(f'merging {len(chunk_files)} chunks')
merged = {}
for f in chunk_files:
    d = json.load(open(f))
    print(f'  {os.path.basename(f)}: {len(d)} samples')
    merged.update(d)
print(f'total: {len(merged)} samples')
out_path = os.path.join(SRC, 'A-OKVQA_risk.json')
json.dump(merged, open(out_path, 'w'))
print(f'saved {out_path}')
# Optional: clean up chunks
for f in chunk_files:
    os.remove(f)
print('chunks removed')
