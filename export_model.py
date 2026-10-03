import torch
from transformer.train import build_train_transformer
from transformer.config import transformer_configurations as cfg

SEQ = 64

class EdgeLM(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, tokens):             # tokens: int32 [1, SEQ]
        return self.m(tokens.long())        # logits: [1, SEQ, vocab]

def load():
    m = build_train_transformer(cfg, cfg["tgt_vocab_size"])
    m.load_state_dict(torch.load("decoder_only.pt", map_location="cpu"))
    return EdgeLM(m.eval().float())

if __name__ == "__main__":
    model = load()
    x = torch.randint(0, 1000, (1, SEQ), dtype=torch.int32)
    with torch.no_grad():
        ref = model(x)
        ep = torch.export.export(model, (x,))
        assert torch.allclose(ep.module()(x), ref, atol=1e-4)
    print("export OK", ref.shape)