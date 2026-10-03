# please note that this accuracy file was written by AI ~ Vraj

import numpy as np
import torch
import qai_hub as hub
from data import encode, get_splits, tok, SEQ
from export_model import load

DEVICE_NAME = "Samsung Galaxy S26 (Family)"
N_VAL = 32   # each sample downloads 64 x 28879 float32 logits (~7.4 MB); 32 samples ~ 240 MB per variant

VARIANTS = {
    "fp16_npu": "mq26yp1ln",  
    "w8a8":     "mn0g52lxm",
    "w8a16":    "mn7583e4m",
}

client = hub.Client()
device = hub.Device(DEVICE_NAME)

_, val_sents = get_splits(n_val=N_VAL)
batch = [encode(s) for s in val_sents]            # list of int32 [1, SEQ]

def metrics(logits_list, ref_list):
    """Perplexity on real (non-pad) next tokens + top-1 agreement with the FP32 reference."""
    nll, n, agree = 0.0, 0, 0
    for x, lg, ref in zip(batch, logits_list, ref_list):
        t = x[0]
        lg = np.asarray(lg, dtype=np.float32).reshape(SEQ, -1)
        ref = np.asarray(ref, dtype=np.float32).reshape(SEQ, -1)
        targets = t[1:]
        mask = targets != tok.PAD_ID
        if mask.sum() == 0:
            continue
        lp = torch.log_softmax(torch.from_numpy(lg[:-1]), dim=-1).numpy()
        nll -= lp[np.arange(SEQ - 1), targets][mask].sum()
        n += int(mask.sum())
        agree += int((lg[:-1].argmax(-1) == ref[:-1].argmax(-1))[mask].sum())
    return float(np.exp(nll / n)), agree / n

# --- FP32 reference on CPU ---
model = load()
with torch.no_grad():
    ref = [model(torch.from_numpy(x)).numpy() for x in batch]
ppl, agr = metrics(ref, ref)
results = {"torch_fp32_cpu": (ppl, agr)}
print(f"torch_fp32_cpu   ppl={ppl:8.3f}   top1_agree={agr:.2%}")

# --- on-device variants ---
for name, model_id in VARIANTS.items():
    job = client.submit_inference_job(
        model=hub.get_model(model_id),
        device=device,
        inputs=dict(tokens=batch),
        name=f"edge-lm-{name}-accuracy",
    )
    print(f"{name}: inference job {job.url} (waiting...)")
    out = job.download_output_data()
    key = next(iter(out))
    logits = out[key]
    print(f"{name}: output '{key}', {len(logits)} samples, shape {np.asarray(logits[0]).shape}")
    ppl, agr = metrics(logits, ref)
    results[name] = (ppl, agr)
    print(f"{name:16s} ppl={ppl:8.3f}   top1_agree={agr:.2%}")

print("\n| Variant | Perplexity | Top-1 agree vs FP32 |")
print("| --- | --- | --- |")
for name, (ppl, agr) in results.items():
    print(f"| {name} | {ppl:.3f} | {agr:.1%} |")