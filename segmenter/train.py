#!/usr/bin/env python3
"""Train the transparency head (segmenter/model.py) on TransCG's ground-truth masks.

    python3 segmenter/train.py --name vits_rgb [--encoder vits] [--use-valid] \
        [--epochs 6 --bs 12 --lr 3e-4 --seed 1234]

Writes segmenter/runs/<name>/{best.pt,last.pt,provenance.txt,train.log}. Selection
is on the held-out val IoU (600 frames, 8 training scenes excluded from the
training set), never on the test subset. The pre-registered thresholds are in
segmenter/README.md; the trainer does not read them.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segmenter.data import TransCGSeg  # noqa: E402
from segmenter.model import TransSegHead, save_checkpoint, seg_loss  # noqa: E402


def iou_batch(logits, target, thresh=0.5):
    pred = logits > 0 if thresh == 0.5 else torch.sigmoid(logits) > thresh
    tgt = target > 0.5
    inter = (pred & tgt).sum(dim=(1, 2)).float()
    union = (pred | tgt).sum(dim=(1, 2)).float()
    return (inter / union.clamp(min=1)).tolist()


def evaluate(model, loader, device, amp):
    model.eval()
    ious = []
    with torch.no_grad():
        for b in loader:
            x, m, v = b["rgb"].to(device), b["mask"].to(device), b["valid"].to(device)
            with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                y = model(x, v if model.use_valid else None)
            ious.extend(iou_batch(y.float(), m))
    return float(np.mean(ious)), len(ious)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--use-valid", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train/val sizes")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    amp = not args.no_amp

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "segmenter" / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    train_ds = TransCGSeg("train", limit=args.limit)
    val_ds = TransCGSeg("val", limit=args.limit)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = TransSegHead(args.encoder, use_valid=args.use_valid).to(device)
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    with open(out / "provenance.txt", "w") as f:
        f.write(f"name: {args.name}\nencoder: {args.encoder}\nuse_valid: {args.use_valid}\n"
                f"train frames: {len(train_ds)}  val frames: {len(val_ds)}\n"
                f"epochs: {args.epochs}  bs: {args.bs}  lr: {args.lr}  wd: {args.wd}  seed: {args.seed}  amp: {amp}\n"
                f"trainable params: {n_train}\ncommit: {commit}\ngpu: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}\n"
                f"started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    log(f"{args.name}: {len(train_ds)} train / {len(val_ds)} val frames, {n_train} trainable params, "
        f"{len(train_dl)} steps/epoch, device {device}")

    best, best_epoch = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i, b in enumerate(train_dl):
            x, m, v = b["rgb"].to(device, non_blocking=True), b["mask"].to(device), b["valid"].to(device)
            with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                y = model(x, v if model.use_valid else None)
            loss, parts = seg_loss(y.float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
            if i % 200 == 0:
                log(f"  epoch {epoch} step {i}/{len(train_dl)} loss {loss:.4f} (bce {parts['bce']:.4f} dice {parts['dice']:.4f}) "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        val_iou, n = evaluate(model, val_dl, device, amp)
        log(f"epoch {epoch}: train loss {np.mean(losses):.4f}, held-out val IoU {val_iou:.4f} over {n} frames, "
            f"{(time.time() - t0) / 60:.1f} min")
        save_checkpoint(out / "last.pt", model, epoch=epoch, val_iou=val_iou, args=vars(args))
        if val_iou > best:
            best, best_epoch = val_iou, epoch
            save_checkpoint(out / "best.pt", model, epoch=epoch, val_iou=val_iou, args=vars(args))
            log(f"  saved best.pt (epoch {epoch}, val IoU {val_iou:.4f})")
    with open(out / "provenance.txt", "a") as f:
        f.write(f"best epoch: {best_epoch}  val IoU: {best:.4f}\nfinished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    log(f"Training finished. Best held-out val IoU {best:.4f} at epoch {best_epoch}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
