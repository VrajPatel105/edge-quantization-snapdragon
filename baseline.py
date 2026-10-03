import os, time
import numpy as np
from data import encode, get_splits, tok, SEQ, ADD_SOS, ADD_EOS

ckpt = "decoder_only.pt"
print(f"checkpoint modified: {time.ctime(os.path.getmtime(ckpt))}")
print(f"vocab size: {tok.vocab_size()}   ADD_SOS={ADD_SOS} ADD_EOS={ADD_EOS}")

probe = ["lily", "timmy", "bunny", "puppy", "playground", "tom", "the", "happy"]
print("probe words in vocab:", {w: (w in tok.word2idx) for w in probe})

_, val = get_splits(n_calib=1, n_val=200)
real = unk = 0
lengths = []
for s in val:
    ids = encode(s)[0]
    content = [i for i in ids if i not in (tok.PAD_ID, tok.SOS_ID, tok.EOS_ID)]
    real += len(content)
    unk += sum(1 for i in content if i == tok.UNK_ID)
    lengths.append(len(content))

print(f"UNK rate on val: {unk / real:.1%}   avg content tokens/sample: {np.mean(lengths):.1f}")
print("\nsample decoded (what the model actually sees):")
print(tok.decode_sentence([i for i in encode(val[0])[0] if i != tok.PAD_ID]))