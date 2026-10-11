"""Experiment 2: fine-tune DINOv3 on the area crops, inside each CV round.

DINOv3 ViT-B/16 at 224 px, last 4 blocks + final norm trainable, with two heads:
- stock: empty / half / full (all training crops);
- product: which of the planogram products is shown, trained only on crops labelled compliant and not
  empty (shelfmon.prototypes.album_mask), so it never learns a swap as the planned product.
P(compliant) = the product head's probability of the expected product (1 for crops it calls empty,
by the team convention). Out-of-fold predictions are cached per round in data/experiments/finetune/,
then scored alone and blended 50/50 with the current model (scripts/experiments/common.py).

    .venv/bin/python scripts/experiments/finetune_cv.py [--seed 1]   # ~2 min per round on an RTX 2060

Each --seed trains its own models (cache in data/experiments/finetune/seed<k>/, seed 0 at the top level);
the blend uses the average of every seed trained so far.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, cross_validate, load, report  # noqa: E402

from shelfmon.prototypes import album_mask  # noqa: E402

HF_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TRAINABLE = tuple(f"model.layer.{i}." for i in range(8, 12)) + ("norm.",)
SIZE, EPOCHS, BATCH = 224, 8, 32
LR_BACKBONE, LR_HEADS = 2e-5, 1e-3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OUT = ROOT / "data/experiments/finetune"
DEVICE = "cuda"


class Net(nn.Module):
    def __init__(self, n_products: int):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(HF_ID)
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith(TRAINABLE)
        dim = self.backbone.config.hidden_size
        self.stock, self.product = nn.Linear(dim, 3), nn.Linear(dim, n_products)

    def forward(self, x):
        h = self.backbone(pixel_values=x).pooler_output
        return self.stock(h), self.product(h)


def load_crops(rows) -> torch.Tensor:
    """(n, 3, SIZE, SIZE) uint8 RGB, in manifest order."""
    out = np.zeros((len(rows), SIZE, SIZE, 3), np.uint8)
    for i, r in enumerate(rows):
        out[i] = cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT / "data/areas" / r["crop_path"])), (SIZE, SIZE),
                                         interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def prepare(batch: torch.Tensor, augment: bool) -> torch.Tensor:
    x = batch.to(DEVICE).float() / 255
    if augment:  # horizontal flip + brightness/contrast jitter, per sample
        flip = torch.rand(len(x), device=DEVICE) < 0.5
        x[flip] = x[flip].flip(-1)
        a = torch.empty(len(x), 1, 1, 1, device=DEVICE).uniform_(0.85, 1.15)
        b = torch.empty(len(x), 1, 1, 1, device=DEVICE).uniform_(-0.08, 0.08)
        x = (x * a + b).clamp(0, 1)
    return (x - MEAN.to(DEVICE)) / STD.to(DEVICE)


def train_round(crops, stock, product, product_ok, train_idx, n_products, seed) -> Net:
    torch.manual_seed(seed)
    net = Net(n_products).to(DEVICE)
    backbone = [p for p in net.backbone.parameters() if p.requires_grad]
    heads = list(net.stock.parameters()) + list(net.product.parameters())
    opt = torch.optim.AdamW([{"params": backbone, "lr": LR_BACKBONE}, {"params": heads, "lr": LR_HEADS}],
                            weight_decay=0.05)
    steps = EPOCHS * int(np.ceil(len(train_idx) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BACKBONE, LR_HEADS], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(EPOCHS):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            x = prepare(crops[idx], augment=True)
            with torch.autocast(DEVICE, dtype=torch.float16):
                s_logits, p_logits = net(x)
                loss = ce(s_logits.float(), stock[idx].to(DEVICE))
                ok = product_ok[idx]
                if ok.any():
                    loss = loss + ce(p_logits.float()[ok.to(DEVICE)], product[idx][ok].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return net


@torch.inference_mode()
def predict(net: Net, crops, idx) -> tuple[np.ndarray, np.ndarray]:
    """Softmax stock and product probabilities, averaged with a horizontally flipped view."""
    net.eval()
    ps, pp = [], []
    for start in range(0, len(idx), 64):
        x = prepare(crops[idx[start:start + 64]], augment=False)
        with torch.autocast(DEVICE, dtype=torch.float16):
            s1, p1 = net(x)
            s2, p2 = net(x.flip(-1))
        ps.append(((s1.float().softmax(-1) + s2.float().softmax(-1)) / 2).cpu().numpy())
        pp.append(((p1.float().softmax(-1) + p2.float().softmax(-1)) / 2).cpu().numpy())
    return np.concatenate(ps), np.concatenate(pp)


def run_dir(seed: int) -> Path:
    return OUT if seed == 0 else OUT / f"seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = run_dir(args.seed)
    table = load()
    rows = table.rows
    products = sorted({r["product_id"] for r in rows})
    product = torch.tensor([products.index(r["product_id"]) for r in rows])
    stock = torch.from_numpy(table.stock).long()
    product_ok = torch.from_numpy(album_mask(table.compliant, table.stock))
    crops = load_crops(rows)
    out.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    p_stock, p_prod = np.zeros((n, 3)), np.zeros((n, len(products)))
    for fold in np.unique(table.fold):
        cache = out / f"round{fold}.npz"
        test_idx = np.where(table.fold == fold)[0]
        if cache.exists():
            data = np.load(cache)
            p_stock[test_idx], p_prod[test_idx] = data["p_stock"], data["p_product"]
            continue
        start = time.time()
        net = train_round(crops, stock, product, product_ok, np.where(table.fold != fold)[0], len(products),
                          seed=fold + 1000 * args.seed)
        p_stock[test_idx], p_prod[test_idx] = predict(net, crops, test_idx)
        np.savez(cache, p_stock=p_stock[test_idx], p_product=p_prod[test_idx])
        del net
        torch.cuda.empty_cache()
        print(f"round {fold}: {len(test_idx)} crops, {time.time() - start:.0f}s")

    p_stock, p_comp = load_oof(table, [out])
    current = cross_validate(table)
    avg = lambda a, b: (a + b) / 2  # noqa: E731
    report(f"2a. fine-tuned DINOv3 alone (seed {args.seed})", table,
           {"p_stock": p_stock, "p_compliant": p_comp, "flagged": np.zeros(n, bool)})
    report(f"2b. blend 50/50 (seed {args.seed})", table, blended(table, current, [out]))
    seeds = trained_seeds(table)
    if len(seeds) > 1:
        report(f"2b. blend 50/50 (average of {len(seeds)} seeds)", table, blended(table, current, seeds))
    p_stock, p_comp = load_oof(table, [out])
    report("2c. blend stock only", table, current | {"p_stock": avg(current["p_stock"], p_stock)})
    report("2d. blend compliance only", table, current | {"p_compliant": avg(current["p_compliant"], p_comp)})


def trained_seeds(table) -> list[Path]:
    """Run folders with every round cached."""
    dirs = [OUT] + sorted(OUT.glob("seed*"))
    return [d for d in dirs if all((d / f"round{k}.npz").exists() for k in np.unique(table.fold))]


def load_oof(table, dirs: list[Path] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fine-tuned out-of-fold P(empty, half, full) and P(compliant) from cached rounds, averaged over runs."""
    dirs = dirs or [OUT]
    products = sorted({r["product_id"] for r in table.rows})
    product = np.array([products.index(r["product_id"]) for r in table.rows])
    n = len(table.rows)
    p_stock, p_prod = np.zeros((n, 3)), np.zeros((n, len(products)))
    for d in dirs:
        for fold in np.unique(table.fold):
            data = np.load(d / f"round{fold}.npz")
            test_idx = np.where(table.fold == fold)[0]
            p_stock[test_idx] += data["p_stock"] / len(dirs)
            p_prod[test_idx] += data["p_product"] / len(dirs)
    expected = p_prod[np.arange(n), product]
    return p_stock, np.where(p_stock.argmax(1) == 0, 1.0, expected)  # empty => compliant (team convention)


def blended(table, current: dict, dirs: list[Path] | None = None) -> dict:
    """The current model's out-of-fold probabilities averaged 50/50 with the fine-tuned model's."""
    p_stock, p_comp = load_oof(table, dirs)
    return {"p_stock": (current["p_stock"] + p_stock) / 2, "p_compliant": (current["p_compliant"] + p_comp) / 2,
            "flagged": current["flagged"]}


if __name__ == "__main__":
    main()
