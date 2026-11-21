"""Score the new gated96/pop4/e300 winner with UNCLIPPED 300-step in-context
SGD on unseen set 13 — the same pre-clip protocol under which the baseline's
0.7956 was recorded (2026-11-20 11:11, before commit 8012613 added the
grad-norm clip to sgd_adapt). Pairs with diag_noclip_baseline.py so both
genomes are judged under both protocols:

  protocol            baseline ctrl-e6-s0   gated96-pop4-e300
  clipped (current)   0.6015                0.7994
  unclipped (pre-clip) <this file>          <this file>
"""
import pickle

import numpy as np

import train_cppn_arc as T


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

E.RUNS = [("gated96 pop4 e300 s0 (unclipped lr0.1)",
          "cppn_arc_gated96_pop4_e300.pkl", 0.10)]
E.main()
