"""InternVL2.5-8B judge wrapper used to replace the Qwen2.5-VL-7B judge.

Usage:
    from internvl_judge import load_judge, judge_image, judge_text
    model, tokenizer = load_judge()
    out = judge_image(model, tokenizer, image, system, user)
    out = judge_text(model, tokenizer, system, user)
"""
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

INTERNVL_PATH = os.environ.get(
    "INTERNVL_PATH",
    "./models/InternVL2_5-8B",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(sz=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(ar, ratios, w, h, sz):
    best_diff = float("inf"); best = (1, 1); area = w * h
    for r in ratios:
        ta = r[0] / r[1]; d = abs(ar - ta)
        if d < best_diff:
            best_diff = d; best = r
        elif d == best_diff and area > 0.5 * sz * sz * r[0] * r[1]:
            best = r
    return best


def _dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size; ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1) for j in range(1, n + 1)
                if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    tr = _find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    blocks = tr[0] * tr[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        x0 = (i % (tw // image_size)) * image_size
        y0 = (i // (tw // image_size)) * image_size
        out.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image_pixel_values(image_or_path, max_num=6, dtype=torch.bfloat16, device="cuda"):
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB") if image_or_path.mode != "RGB" else image_or_path
    transform = _build_transform()
    imgs = _dynamic_preprocess(img, max_num=max_num)
    pv = torch.stack([transform(i) for i in imgs]).to(dtype).to(device)
    return pv


_MODEL = None
_TOKENIZER = None


def load_judge():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    print(f"[internvl_judge] loading {INTERNVL_PATH} ...")
    _TOKENIZER = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)
    _MODEL = AutoModel.from_pretrained(
        INTERNVL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    return _MODEL, _TOKENIZER


def _gen_cfg(max_new=64):
    return dict(max_new_tokens=max_new, do_sample=False, num_beams=1)


_FORMAT_INSTR = (
    "\n\nFINAL OUTPUT FORMAT (very important):\n"
    "Choose exactly ONE option from:\n"
    "  A. grounded\n"
    "  B. contradicted\n"
    "  C. unverifiable\n"
    "Output a single line: 'Answer: A', 'Answer: B', or 'Answer: C'. "
    "Optionally, output a single 'Reason:' line BEFORE 'Answer:'. "
    "Do not write anything after 'Answer:'."
)


def _strip_then_normalize(resp: str) -> str:
    """Convert 'Answer: A' style replies to 'Verdict: grounded' so existing
    parse_judge() in each risk script keeps working."""
    s = (resp or "").strip()
    import re
    m = re.search(r"answer\s*:?[\s]*([abc])\b", s, re.I)
    label = None
    if m:
        label = {"a": "grounded", "b": "contradicted", "c": "unverifiable"}[m.group(1).lower()]
    else:
        m2 = re.search(r"\b(grounded|contradicted|unverifiable|supported|unsupported)\b", s, re.I)
        if m2:
            v = m2.group(1).lower()
            label = {"supported": "grounded", "unsupported": "contradicted"}.get(v, v)
    if label is not None:
        s = s + f"\nVerdict: {label}"
    return s


def judge_image(model, tokenizer, image, system_text, user_text, max_new=160):
    pv = load_image_pixel_values(image)
    prompt = f"{system_text}{_FORMAT_INSTR}\n\n<image>\n{user_text}"
    resp = model.chat(tokenizer, pv, prompt, _gen_cfg(max_new), history=None, return_history=False)
    return _strip_then_normalize(resp)


def judge_text(model, tokenizer, system_text, user_text, max_new=160):
    prompt = f"{system_text}{_FORMAT_INSTR}\n\n{user_text}"
    resp = model.chat(tokenizer, None, prompt, _gen_cfg(max_new), history=None, return_history=False)
    return _strip_then_normalize(resp)
