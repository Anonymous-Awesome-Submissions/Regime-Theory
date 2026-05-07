"""ARC v4.1 Step A: extract CLIP image features for all 3 visual benchmarks.

Diagnosed v4 issue: dense text features (question / context / answer
embeddings) didn't help the learned solver beat best_fixed on Hallu,
because text-side features don't predict whether the model will
hallucinate visual content.

Fix: add CLIP ViT-L/14 pooled image features (768-dim) per sample.
These actually correlate with what's in the image and complement the text
features the controller already has.

Output: ./data/unified_v4/{bench}_image_features.npz
        with X_img (n × 768) and sample_ids (list[str])

Benchmarks with images: POPE, HallusionBench, A-OKVQA. FEVER is text-only.
"""
import json, os, sys, time
import numpy as np

CLIP_PATH = "./models/clip-vit-large-patch14"
DST = "./data/unified_v4"

POPE_DS  = "./datasets/POPE"
HALLU_DS = "./datasets/HallusionBench"
AOKVQA_SAMP = "./data/AOKVQA/samples_val.json"
ACTION_TABLE_DIR = "./data/unified"


def find_snapshot(path):
    import glob
    for cand in glob.glob(os.path.join(path, 'snapshots', '*')):
        if os.path.isdir(cand): return cand
    raise RuntimeError(f'No snapshot under {path}')


def iter_pope():
    from datasets import load_from_disk
    ds = load_from_disk(POPE_DS)
    sid_to_img = {f"pope_{ds[i]['id']}": ds[i]['image'] for i in range(len(ds))}
    rows = json.load(open(os.path.join(ACTION_TABLE_DIR, 'POPE_action_table.json')))
    for r in rows:
        sid = r['sample_id']; img = sid_to_img.get(sid)
        if img is not None:
            yield sid, img


def iter_hallu():
    from datasets import load_from_disk
    ds = load_from_disk(HALLU_DS)
    sid_to_img = {}
    for i in range(len(ds)):
        rr = ds[i]
        sid = f"hallu_{rr['set_id']}_{rr['figure_id']}_{rr['question_id']}_{rr['visual_input']}"
        sid_to_img[sid] = rr['image']
    rows = json.load(open(os.path.join(ACTION_TABLE_DIR, 'HallusionBench_action_table.json')))
    for r in rows:
        sid = r['sample_id']; img = sid_to_img.get(sid)
        if img is not None:
            yield sid, img


def iter_aokvqa():
    from PIL import Image
    samples = json.load(open(AOKVQA_SAMP))
    samp_map = {s['sample_id']: s for s in samples}
    rows = json.load(open(os.path.join(ACTION_TABLE_DIR, 'A-OKVQA_action_table.json')))
    for r in rows:
        sid = r['sample_id']; meta = samp_map.get(sid)
        if meta is None: continue
        try:
            img = Image.open(meta['image_path']).convert('RGB')
            yield sid, img
        except Exception:
            continue


def encode_bench(bench, iter_fn, processor, model, device, batch=16):
    import torch
    sample_ids = []
    images_buf = []
    feats_buf = []
    t0 = time.time()
    n = 0

    def flush():
        nonlocal images_buf
        if not images_buf:
            return
        with torch.no_grad():
            inp = processor(images=images_buf, return_tensors='pt').to(device)
            out = model.get_image_features(**inp)
            # In newer transformers versions get_image_features may return
            # BaseModelOutputWithPooling; fall back to .image_embeds or .pooler_output
            if not isinstance(out, torch.Tensor):
                out = getattr(out, 'image_embeds', None) or getattr(out, 'pooler_output', None)
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            feats_buf.append(out.cpu().numpy())
        images_buf = []

    for sid, img in iter_fn():
        sample_ids.append(sid)
        images_buf.append(img)
        n += 1
        if len(images_buf) >= batch:
            flush()
            if n % 100 == 0:
                print(f'  [{bench}] {n} images in {time.time()-t0:.1f}s')
    flush()
    X = np.concatenate(feats_buf, axis=0) if feats_buf else np.zeros((0, 768))
    print(f'  [{bench}] done: {len(sample_ids)} images in {time.time()-t0:.1f}s shape={X.shape}')
    out_path = os.path.join(DST, f'{bench}_image_features.npz')
    np.savez(out_path, X_img=X, sample_ids=np.array(sample_ids))
    print(f'  saved {out_path}')


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    os.makedirs(DST, exist_ok=True)
    snap = find_snapshot(CLIP_PATH)
    print(f'Loading CLIP from {snap}')
    processor = CLIPProcessor.from_pretrained(snap)
    model = CLIPModel.from_pretrained(snap).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'CLIP loaded on {device}')

    for bench, fn in [('POPE', iter_pope), ('HallusionBench', iter_hallu), ('A-OKVQA', iter_aokvqa)]:
        print(f'\n=== {bench} ===')
        encode_bench(bench, fn, processor, model, device)


if __name__ == '__main__':
    main()
