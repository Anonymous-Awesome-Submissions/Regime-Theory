"""ARC Step 2 benchmark loader.

One entry point per benchmark. Each yields canonical rows:

    {
      'sample_id':  str,
      'benchmark':  str,
      'category':   str or None,
      'task':       'yesno' | 'mc' | 'fever',
      'question':   str,
      'image':      PIL.Image or None,
      'gt':         str (lowercased canonical gt)    # for scoring
      'choices':    list[str] or None,               # mc only
      'label_set':  tuple[str, ...],                 # allowed answer tokens
    }

All four benchmarks aligned so Step 2 generation can iterate once per sample.
"""
import json, os

POPE_CANDS  = "./data/GRACE/POPE/candidates.json"
HALLU_CANDS = "./data/GRACE/HallusionBench/candidates.json"
AOKVQA_SAMP = "./data/AOKVQA/samples_val.json"
FEVER_CANDS = "./data/GRACE/FEVER/candidates.json"
POPE_DS   = "./datasets/POPE"
HALLU_DS  = "./datasets/HallusionBench"

FEVER_LABEL_MAP = {0: 'supports', 1: 'refutes', 2: 'nei'}
YN = ('yes', 'no')


def load_pope():
    from datasets import load_from_disk
    cands = json.load(open(POPE_CANDS))
    ds = load_from_disk(POPE_DS)
    sid_to_img = {f"pope_{ds[i]['id']}": ds[i]['image'] for i in range(len(ds))}
    for c in cands:
        img = sid_to_img.get(c['sample_id'])
        if img is None: continue
        yield {
            'sample_id': c['sample_id'], 'benchmark': 'POPE',
            'category':  c.get('category'), 'task': 'yesno',
            'question':  c['question'], 'image': img,
            'gt':        c['gt_answer'].strip().lower(),
            'choices':   None, 'label_set': YN,
        }


def load_hallu():
    """HallusionBench loader.

    The HF dataset has 951 rows but the (set_id, figure_id, question_id, visual_input)
    tuple is non-unique (only 383 unique). Adding sample_note as a disambiguator
    recovers the full 951 distinct samples. We sanitize sample_note for filename use.
    """
    from datasets import load_from_disk
    ds = load_from_disk(HALLU_DS)

    def _sid(r):
        note = (r.get('sample_note') or '').strip()
        note = ''.join(ch if ch.isalnum() else '_' for ch in note)[:30]
        return f"hallu_{r['set_id']}_{r['figure_id']}_{r['question_id']}_{r['visual_input']}_{note}"

    HALLU_GT_MAP = {'0': 'no', '1': 'yes', 'no': 'no', 'yes': 'yes'}
    for i in range(len(ds)):
        r = ds[i]
        gt_raw = (r.get('gt_answer') or '').strip().lower()
        gt = HALLU_GT_MAP.get(gt_raw, gt_raw)
        yield {
            'sample_id': _sid(r), 'benchmark': 'HallusionBench',
            'category':  r.get('category'), 'task': 'yesno',
            'question':  r['question'], 'image': r['image'],
            'gt':        gt,
            'choices':   None, 'label_set': YN,
        }


def load_aokvqa():
    from PIL import Image
    samples = json.load(open(AOKVQA_SAMP))
    for s in samples:
        try:
            img = Image.open(s['image_path']).convert('RGB')
        except Exception:
            continue
        yield {
            'sample_id': s['sample_id'], 'benchmark': 'A-OKVQA',
            'category':  s['category'], 'task': 'mc',
            'question':  s['question'], 'image': img,
            'gt':        s['gt_choice'].strip().lower(),
            'choices':   s['choices'],
            'label_set': tuple(c.lower() for c in s['choices']),
        }


def load_pope_ext():
    """Extended POPE pool (full 9000) for the appendix n-sweep."""
    from PIL import Image
    cands = json.load(open(
        "./data/GRACE/POPE_ext/candidates.json"))
    for c in cands:
        try:
            img = Image.open(c['image_path']).convert('RGB')
        except Exception:
            continue
        yield {
            'sample_id': c['sample_id'], 'benchmark': 'POPE_ext',
            'category':  c.get('category'), 'task': 'yesno',
            'question':  c['question'], 'image': img,
            'gt':        c['gt_answer'].strip().lower(),
            'choices':   None, 'label_set': YN,
        }


def load_fever():
    cands = json.load(open(FEVER_CANDS))
    for c in cands:
        yield {
            'sample_id': c['sample_id'], 'benchmark': 'FEVER',
            'category':  None, 'task': 'fever',
            'question':  c['question'], 'image': None,
            'gt':        FEVER_LABEL_MAP[int(c['gt_answer'])],
            'choices':   None,
            'label_set': ('supports', 'refutes', 'nei'),
        }


FOLIO_CANDS = "./data/GRACE/FOLIO/candidates.json"
FOLIO_LABEL_MAP = {0: 'true', 1: 'false', 2: 'uncertain'}


def load_folio():
    cands = json.load(open(FOLIO_CANDS))
    for c in cands:
        yield {
            'sample_id': c['sample_id'], 'benchmark': 'FOLIO',
            'category':  c.get('category'), 'task': 'folio',
            'question':  c['question'], 'image': None,
            'gt':        FOLIO_LABEL_MAP[int(c['gt_answer'])],
            'choices':   None,
            'label_set': ('true', 'false', 'uncertain'),
        }


LOADERS = {
    'POPE': load_pope, 'POPE_ext': load_pope_ext,
    'HallusionBench': load_hallu,
    'A-OKVQA': load_aokvqa, 'FEVER': load_fever,
    'FOLIO': load_folio,
}


def iter_bench(name):
    return LOADERS[name]()
