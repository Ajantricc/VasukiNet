# LoopGNN-SLM

A compact character-level small language model combining:

- LoopFormer-style elastic recurrent transformer refinement
- Time and step-size loop conditioning
- Shortcut consistency between short and long loop trajectories
- Graph Neural Network message passing over token nodes

This is an educational, end-to-end PyTorch implementation. It is intentionally small enough to train on CPU/Mac/GPU using a text file.

## Setup

```bash
cd loopgnn_slm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare data

Place any UTF-8 text into `data/input.txt`.

```bash
mkdir -p data
cat > data/input.txt <<'TXT'
LoopGNN is a small language model. It refines token states through loops and graph messages.
TXT
```

## Train

```bash
python train.py --data data/input.txt --out runs/loopgnn --steps 1000 --device cpu
```

For Apple Silicon, try:

```bash
python train.py --data data/input.txt --out runs/loopgnn --steps 1000 --device mps
```

## Generate

```bash
python generate.py --ckpt runs/loopgnn/ckpt.pt --prompt "LoopGNN" --tokens 300 --loops 6
```
