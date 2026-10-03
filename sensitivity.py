import copy
import torch
import torch.nn as nn
from data import encode, get_splits, tok
from export_model import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

base = load().m.to(DEV).eval()     # unwrap EdgeLM -> TrainTransformer
VOCAB = tok.vocab_size()

calib_s, val_s = get_splits(n_calib=200, n_val=200)
calib = torch.cat([torch.from_numpy(encode(s)) for s in calib_s]).long().to(DEV)
val = torch.cat([torch.from_numpy(encode(s)) for s in val_s]).long().to(DEV)

# ---------- helpers ----------
def fq(x, scale, bits):
    """symmetric fake-quant"""
    qmax = 2 ** (bits - 1) - 1
    return (x / scale).round().clamp(-qmax, qmax) * scale

def group_of(name, mod):
    if isinstance(mod, nn.Embedding):
        return "embedding"
    if isinstance(mod, nn.Linear) and mod.out_features == VOCAB:
        return "lm_head"
    if "attention" in name or "attn" in name:
        return "attention"
    return "ffn"

def targets(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))]

@torch.no_grad()
def perplexity(model, data, bs=50):
    nll, n = 0.0, 0
    for i in range(0, len(data), bs):
        x = data[i:i + bs]
        logits = model(x).float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = x[:, 1:]
        mask = tgt != tok.PAD_ID
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll -= tok_lp[mask].sum().item()
        n += mask.sum().item()
    return float(torch.exp(torch.tensor(nll / n)))

@torch.no_grad()
def calibrate_acts(model):
    """per-Linear input stats on calibration data: absmax + outlier ratio (max/mean abs)"""
    stats, hooks = {}, []
    for name, m in targets(model):
        if isinstance(m, nn.Linear):
            def h(mod, inp, name=name):
                a = inp[0].detach().abs()
                s = stats.setdefault(name, {"max": 0.0, "sum": 0.0, "cnt": 0})
                s["max"] = max(s["max"], a.max().item())
                s["sum"] += a.sum().item()
                s["cnt"] += a.numel()
            hooks.append(m.register_forward_pre_hook(h))
    for i in range(0, len(calib), 50):
        model(calib[i:i + 50])
    for h in hooks:
        h.remove()
    for s in stats.values():
        s["ratio"] = s["max"] / (s["sum"] / s["cnt"])
    return stats

def quantize(model, groups, w_bits=8, a_bits=None, per_channel=True, act_stats=None):
    """return a fake-quantized copy; only modules whose group is in `groups` are touched"""
    q = copy.deepcopy(model)
    for name, m in targets(q):
        if group_of(name, m) not in groups:
            continue
        w = m.weight.data
        qmax = 2 ** (w_bits - 1) - 1
        if per_channel:
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        else:
            scale = w.abs().max().clamp(min=1e-8) / qmax
        m.weight.data = fq(w, scale, w_bits)
        if a_bits and isinstance(m, nn.Linear):
            a_scale = max(act_stats[name]["max"], 1e-8) / (2 ** (a_bits - 1) - 1)
            m.register_forward_pre_hook(
                lambda mod, inp, s=a_scale, b=a_bits: (fq(inp[0], s, b),))
    return q

# ---------- run ----------
ALL = {"embedding", "attention", "ffn", "lm_head"}
print("module groups (verify these look right):")
for n, m in targets(base):
    print(f"  {group_of(n, m):10s} {n}  {tuple(m.weight.shape)}")

act_stats = calibrate_acts(base)
fp = perplexity(base, val)
rows = [("fp32", fp)]

experiments = [
    ("W8 per-tensor, all",             dict(groups=ALL, per_channel=False)),
    ("W8 per-channel, all",            dict(groups=ALL, per_channel=True)),
    ("W8A16 per-channel, all",         dict(groups=ALL, a_bits=16)),
    ("W8A8 per-tensor, all",           dict(groups=ALL, a_bits=8, per_channel=False)),
    ("W8A8 per-channel, all",          dict(groups=ALL, a_bits=8)),
]
for g in sorted(ALL):
    experiments.append((f"W8A8 only {g}", dict(groups={g}, a_bits=8)))
    experiments.append((f"W8A8 all except {g}", dict(groups=ALL - {g}, a_bits=8)))

for label, kw in experiments:
    qm = quantize(base, act_stats=act_stats, **kw)
    p = perplexity(qm, val)
    rows.append((label, p))
    del qm
    torch.cuda.empty_cache() if DEV == "cuda" else None

print("\n| Config | Perplexity | vs FP32 |")
print("| --- | --- | --- |")
for label, p in rows:
    print(f"| {label} | {p:.2f} | {p / fp - 1:+.1%} |")

print("\nTop 10 Linear inputs by outlier ratio (max|x| / mean|x|):")
print("| Module | Group | max abs | ratio |")
print("| --- | --- | --- | --- |")
mods = dict(targets(base))
for name, s in sorted(act_stats.items(), key=lambda kv: -kv[1]["ratio"])[:10]:
    print(f"| {name} | {group_of(name, mods[name])} | {s['max']:.2f} | {s['ratio']:.1f} |")

import re

def agree(model_a, model_b, data, bs=50):
    hit = tot = 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            x = data[i:i + bs]
            a = model_a(x)[:, :-1].argmax(-1)
            b = model_b(x)[:, :-1].argmax(-1)
            mask = x[:, 1:] != tok.PAD_ID
            hit += (a == b)[mask].sum().item()
            tot += mask.sum().item()
    return hit / tot

is_head  = lambda n, m: isinstance(m, nn.Linear) and m.out_features == VOCAB
is_norm  = lambda n, m: "norm" in type(m).__name__.lower()
is_block = lambda n, m: re.fullmatch(r"decoder_blocks\.\d+", n) is not None

@torch.no_grad()
def out_absmax(model, pred):
    stats, hooks = {}, []
    for n, m in model.named_modules():
        if pred(n, m):
            def h(mod, inp, out, n=n):
                stats[n] = max(stats.get(n, 0.0), out.detach().abs().max().item())
            hooks.append(m.register_forward_hook(h))
    for i in range(0, len(calib), 50):
        model(calib[i:i + 50])
    for h in hooks:
        h.remove()
    return stats

def add_out_quant(model, preds, stats, bits):
    for n, m in model.named_modules():
        if any(p(n, m) for p in preds):
            s = max(stats[n], 1e-8) / (2 ** (bits - 1) - 1)
            m.register_forward_hook(lambda mod, inp, out, s=s, b=bits: fq(out, s, b))
    return model

print("\nnorm modules found:", [n for n, m in base.named_modules() if is_norm(n, m)][:4], "...")
print("block modules found:", [n for n, m in base.named_modules() if is_block(n, m)])

out_stats = out_absmax(base, lambda n, m: is_head(n, m) or is_norm(n, m) or is_block(n, m))

# dead-ReLU check: fraction of exact zeros entering each FFN linear2
zeros, hooks = {}, []
for n, m in base.named_modules():
    if n.endswith("feed_forward.linear2"):
        hooks.append(m.register_forward_pre_hook(
            lambda mod, inp, n=n: zeros.setdefault(n, []).append((inp[0] == 0).float().mean().item())))
with torch.no_grad():
    base(calib[:50])
for h in hooks:
    h.remove()
print("\n| FFN linear2 input | % exactly zero |")
print("| --- | --- |")
for n, v in zeros.items():
    print(f"| {n} | {100 * sum(v) / len(v):.1f}% |")

full_exps = [
    ("W8A8 linears + INT8 logits",       8, [is_head]),
    ("W8A8 linears + INT8 norm outputs", 8, [is_norm]),
    ("W8A8 linears + INT8 residual",     8, [is_block]),
    ("W8A8 linears + all three",         8, [is_head, is_norm, is_block]),
    ("W8A16 linears + all three (A16)", 16, [is_head, is_norm, is_block]),
]
print("\n| Config | Perplexity | vs FP32 | Top-1 agree |")
print("| --- | --- | --- | --- |")
for label, bits, preds in full_exps:
    qm = quantize(base, groups=ALL, a_bits=bits, act_stats=act_stats)
    qm = add_out_quant(qm, preds, out_stats, bits)
    p = perplexity(qm, val)
    a = agree(qm, base, val)
    print(f"| {label} | {p:.2f} | {p / fp - 1:+.1%} | {a:.1%} |")
    del qm