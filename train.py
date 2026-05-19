from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from loopgnn_slm.data import CharBlockDataset
from loopgnn_slm.model import LoopGNNConfig, LoopGNNSmallLM
from loopgnn_slm.tokenizer import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to UTF-8 training text")
    p.add_argument("--out", default="runs/loopgnn")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--max-loops", type=int, default=8)
    p.add_argument("--gnn-radius", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--shortcut-weight", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(out / "tokenizer.json")
    ids = tokenizer.encode(text)

    dataset = CharBlockDataset(ids, args.block_size)
    val_len = max(1, int(0.05 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = LoopGNNConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        max_loops=args.max_loops,
        gnn_edge_radius=args.gnn_radius,
    )
    device = pick_device(args.device)
    model = LoopGNNSmallLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb, loops=args.max_loops)
                losses.append(float(loss.detach().cpu()))
                if len(losses) >= 20:
                    break
        model.train()
        return sum(losses) / max(1, len(losses))

    iterator = iter(train_loader)
    best_val = float("inf")
    pbar = tqdm(range(1, args.steps + 1), desc="training")
    for step in pbar:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)

        short_loops = random.randint(1, max(1, args.max_loops // 2))
        long_loops = random.randint(short_loops, args.max_loops)

        logits_s, lm_loss_s, h_s = model(xb, yb, loops=short_loops, return_hidden=True)
        with torch.no_grad():
            _, _, h_l = model(xb, yb, loops=long_loops, return_hidden=True)
        consistency = F.mse_loss(F.normalize(h_s, dim=-1), F.normalize(h_l, dim=-1))
        loss = lm_loss_s + args.shortcut_weight * consistency

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", loops=f"{short_loops}->{long_loops}")
        if step % 250 == 0 or step == args.steps:
            val = evaluate()
            ckpt = {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "val_loss": val,
                "step": step,
            }
            torch.save(ckpt, out / "ckpt.pt")
            if val < best_val:
                best_val = val
                torch.save(ckpt, out / "best.pt")
            (out / "metrics.json").write_text(json.dumps({"step": step, "val_loss": val}, indent=2))
            print(f"\nstep={step} val_loss={val:.4f} best={best_val:.4f}")

    print(f"Saved checkpoint to {out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
