import random
import pickle
import numpy as np
from datasets import load_dataset
from transformer.tokenizer import Tokenizer

SEQ = 64
ADD_SOS = False
ADD_EOS = False

tok = pickle.load(open("tokenizer.pkl", "rb"))

def encode(s):
    ids = tok.encode_sentence(s, add_sos=ADD_SOS, add_eos=ADD_EOS)[:SEQ]
    ids = ids + [tok.PAD_ID] * (SEQ - len(ids))
    return np.array([ids], dtype=np.int32)          # shape [1, SEQ]

def get_splits(n_calib=500, n_val=100, seed=0):
    rng = random.Random(seed)

    train = load_dataset("roneneldan/TinyStories", split="train[:200000]")["text"]
    calib = rng.sample(train, n_calib)

    val_all = load_dataset("roneneldan/TinyStories", split="validation")["text"]
    val = rng.sample(val_all, n_val)

    return calib, val