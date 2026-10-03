import csv, random, pickle
import numpy as np
from transformer.tokenizer import Tokenizer  

SEQ = 64
ADD_SOS = False
ADD_EOS = False
DATASET = "tinystories"                  
TSV_PATH = "transformer/English-German.tsv"
N_TRAIN_ROWS = 200_000                    

tok = pickle.load(open("tokenizer.pkl", "rb"))

def encode(s):
    ids = tok.encode_sentence(s, add_sos=ADD_SOS, add_eos=ADD_EOS)[:SEQ]
    ids = ids + [tok.PAD_ID] * (SEQ - len(ids))
    return np.array([ids], dtype=np.int32)

def _tatoeba():
    with open(TSV_PATH, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    col = next(i for i, v in enumerate(rows[0]) if not v.strip().isdigit())
    sents = [r[col] for r in rows if len(r) > col and r[col].strip()]
    train, held = sents[:N_TRAIN_ROWS], sents[N_TRAIN_ROWS:]
    if len(held) < 200:
        print(f"WARNING: only {len(held)} rows beyond the training slice; val overlaps training")
        held = train[-5000:]
    return train, held

def _tinystories():
    from datasets import load_dataset
    train = load_dataset("roneneldan/TinyStories", split="train[:200000]")["text"]
    held = load_dataset("roneneldan/TinyStories", split="validation")["text"]
    return train, held

def get_splits(n_calib=500, n_val=100, seed=0):
    train, held = _tatoeba() if DATASET == "tatoeba" else _tinystories()
    rng = random.Random(seed)
    return rng.sample(train, n_calib), rng.sample(held, n_val)