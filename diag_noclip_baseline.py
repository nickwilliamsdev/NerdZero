"""Diagnostic: reproduce the PRE-clip (2026-11-20 11:11) 300-step baseline
score on set-13 by monkeypatching sgd_adapt with an unclipped variant.

Committed (pre-clip) baseline: mean=0.7956
Post-clip re-score (eval_set13_300steps.py): mean=0.6015

If unclipped reproduces ~0.7956, the clip fix is the cause of the absolute
shift, and all pre-clip numbers in the repo are under a different protocol.
"""
import pickle

import numpy as np

import train_cppn_arc as T
from train_cppn_arc import sgd_adapt as sgd_adapt_clipped


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


T.sgd_adapt = sgd_adapt_noclip

import eval_set13_300steps as E

# only score the baseline genome
E.RUNS = [("ctrl e6 s0 (UNCLIPPED)", "sweep_ctrl_delta48_e6.pkl", 0.05)]
E.main()
