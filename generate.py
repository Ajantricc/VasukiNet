from __future__ import annotations

import argparse
from pathlib import Path

import torch

from loopgnn_slm.model import LoopGNNConfig, LoopGNNSmallLM
from loopgnn_slm.tokenizer import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--loops", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    tok = CharTokenizer.load(ckpt_path.parent / "tokenizer.json")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = LoopGNNConfig(**ckpt["config"])
    device = pick_device(args.device)
    model = LoopGNNSmallLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    prompt = args.prompt or " "
    x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, loops=args.loops)
    print(tok.decode(y[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
