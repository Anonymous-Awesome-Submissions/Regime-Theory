"""ARC Step 1: preprocess A-OKVQA into canonical sample schema.

A-OKVQA replaces MMHal as the main recovery benchmark. We use the val split
(n=1145) because it has ground-truth direct answers + rationales + multiple
choice, and the image IDs are 100% resolvable via COCO 2014 files.

Canonical schema (aligned with POPE / HallusionBench / FEVER action tables):

    {
      "sample_id":    str,       # question_id
      "benchmark":    "A-OKVQA",
      "question":     str,
      "image_path":   str,       # absolute path to COCO jpg
      "gt_answer":    str,       # most-common direct answer
      "gt_choice":    str,       # canonical MC answer
      "choices":      [str],     # 4 options
      "rationales":   [str],     # used only for recovery quality check, not eval
      "category":     str,       # difficulty bucket for controller features
      "subtype":      str,       # MC vs open-ended flag
    }

The `category` field is needed so the action controller can condition on
context (matches how HallusionBench uses category). We bucket by:
  - has_choices=True always
  - `difficult_direct_answer` flag from the dataset
  - a length bucket of the question (proxy for complexity)

Output: ./data/AOKVQA/samples_val.json
"""
import json, os
from collections import Counter

AOKVQA_ROOT = './data/AOKVQA'
COCO14_TRAIN = './datasets/vqav2/images/train2014'
COCO14_VAL   = './datasets/vqav2/images/val2014'


def resolve_image(image_id: int) -> str:
    for d, pfx in [(COCO14_TRAIN, 'COCO_train2014_'), (COCO14_VAL, 'COCO_val2014_')]:
        p = os.path.join(d, f'{pfx}{image_id:012d}.jpg')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'image_id {image_id} not in COCO2014')


def majority_answer(direct_answers):
    c = Counter(a.strip().lower() for a in direct_answers)
    return c.most_common(1)[0][0]


def bucket(q: str, difficult: bool) -> str:
    n_tok = len(q.split())
    if difficult:
        return 'difficult'
    if n_tok <= 7:
        return 'short'
    if n_tok <= 14:
        return 'medium'
    return 'long'


def main():
    src = os.path.join(AOKVQA_ROOT, 'aokvqa_v1p0_val.json')
    data = json.load(open(src))
    out = []
    for r in data:
        img = resolve_image(r['image_id'])
        gt_direct = majority_answer(r['direct_answers'])
        gt_choice = r['choices'][r['correct_choice_idx']]
        out.append({
            'sample_id':  r['question_id'],
            'benchmark':  'A-OKVQA',
            'question':   r['question'],
            'image_path': img,
            'gt_answer':  gt_direct,
            'gt_choice':  gt_choice,
            'choices':    r['choices'],
            'rationales': r['rationales'],
            'category':   bucket(r['question'], r['difficult_direct_answer']),
            'subtype':    'multiple_choice',
        })
    dst = os.path.join(AOKVQA_ROOT, 'samples_val.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'wrote {len(out)} samples -> {dst}')
    print('category distribution:', Counter(r['category'] for r in out))


if __name__ == '__main__':
    main()
