"""Identify the ORIGINAL ad-hoc 300-step protocol by scanning the baseline
genome across (clip, lr) variants until one reproduces the recorded 0.7956
per-task vector on set 13.

Recorded (2026-11-20 11:11, ad-hoc harness, pre-clip):
  mean=0.7956 per-task=[0.9556, 0.75, 0.9441, 0.78, 0.8056, 0.92, 0.6736,
                        0.5556, 0.92, 0.8, 0.8889, 0.5625, 0.75, 0.95,
                        0.5432, 0.93]
Known: unclipped lr0.05 -> 0.7492; clipped lr0.05 -> 0.6015.
"""
import numpy as np

import train_cppn_arc as T

_SGD_ADAPT_ORIG = T.sgd_adapt  # clipped (current code)

import eval_set13_300steps as E

TARGET = [0.9556, 0.75, 0.9441, 0.78, 0.8056, 0.92, 0.6736, 0.5556,
          0.92, 0.8, 0.8889, 0.5625, 0.75, 0.95, 0.5432, 0.93]


def sgd_adapt_noclip(phen, ep, steps, lr, device):
    import torch, torch.nn.functional as F
    phen.train()
    opt = torch.optim.SGD(phen.parameters(), lr=lr)
    epd = T._ep_on_device(ep, device)
    target = (epd.target.squeeze(0) * 9.0).round().long().unsqueeze(0)
    for _ in range(steps):
        opt.zero_grad()
        pred = phen(epd)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        opt.step()
    phen.eval()


def run_variant(label, lr, noclip):
    E.RUNS = [(label, "sweep_ctrl_delta48_e6.pkl", lr)]
    T.sgd_adapt = sgd_adapt_noclip if noclip else _SGD_ADAPT_ORIG
    out = {}
    # call main but capture prints: simplest is to run and read the
    # printed per-task line
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.main()
    txt = buf.getvalue()
    for line in txt.splitlines():
        if "mean=" in line and "per-task=" in line:
            per = [float(x) for x in line.split("per-task=")[1]
                   .strip().strip("[]").split(", ")]
            out["per"] = per
            out["mean"] = float(line.split("mean=")[1].split()[0])
            break
    d = np.mean(np.abs(np.array(out["per"]) - np.array(TARGET)))
    print(f"{label}: mean={out['mean']:.4f}  mean_abs_diff_vs_recorded={d:.4f}")
    print(f"  per-task={out['per']}")
    return out


if __name__ == "__main__":
    import sys
    variants = [("unclipped lr0.1", 0.10, True),
                ("clipped   lr0.1", 0.10, False)]
    if len(sys.argv) > 1 and sys.argv[1] == "lr001":
        variants = [("unclipped lr0.001", 0.001, True),
                    ("clipped   lr0.001", 0.001, False)]
    for label, lr, nc in variants:
        run_variant(label, lr, nc)
