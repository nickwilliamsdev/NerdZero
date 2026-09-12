from __future__ import annotations

import math
import random
import os
import pickle
import tempfile
import textwrap
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sefer.algebra.rollout import apply_program_actions, apply_selected_actions, differentiable_rollout, operator_separation_loss
from sefer.controllers.reasoner import TinyReasoner
from sefer.evaluation.planning import evaluate_muzero_program_search, evaluate_root_search_diagnostics
from sefer.evaluation.primitives import evaluate_held_out, evaluate_operator_rollout
from sefer.evaluation.programs import evaluate_exact_variable_program_search, evaluate_greedy_program_accuracy
from sefer.evolution.evolve_transport import antithetic_es_step_, evolve_transport_cppn
from sefer.planning.bellman import exact_horizon_policy_target, exact_horizon_value_target, soft_policy_cross_entropy
from sefer.planning.puct import SearchConfig, goal_puct_search, root_action_diagnostics
from sefer.tasks.synthetic_algebra import SyntheticTaskBatch
from sefer.training.losses import transport_supervision_loss
from sefer.training.phases import freeze_for_program_phase, set_query_codec_trainable
from sefer.utils import seed_all

def train_smoke_test(
    steps: int = 4200,
    batch_size: int = 32,
    inner_rollout_steps: int = 1,
    warmup_steps: int = 250,
    algebra_steps: int = 1200,
    es_every: int = 0,
    diagnostic_every: int = 25,
    mcts_train_every: int = 5,
    mcts_train_samples: int = 4,
    mcts_simulations: int = 96,
    mcts_eval_simulations: int = 384,
    mcts_eval_batch: int = 16,
    neat_generations: int = 100,
    neat_population: int = 256,
    neat_workers: int = 12,
    neat_inner_steps: int = 32,
    neat_inner_lr: float = 1e-2,
    neat_seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_all(0)
    torch_device = torch.device(device)
    model = TinyReasoner().to(torch_device)
    tasks = SyntheticTaskBatch(dim=32)

    # v21: topology/weights of the shared edge CPPN are evolved before gradient
    # training. NEAT evaluation is kept on CPU for compatibility; the installed
    # PyTorch-NEAT CPPN evaluates torch tensors on whatever device they are given.
    print(f"evolving shared transport CPPN: generations={neat_generations} population={neat_population}")
    evolve_transport_cppn(model, generations=neat_generations, pop_size=neat_population, seed=neat_seed, workers=neat_workers, inner_steps=neat_inner_steps, inner_lr=neat_inner_lr)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)

    print(f"device={device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"substrate_nodes={model.core.coords.shape[0]}")
    print(f"operators={model.operator_count} + STOP")
    print(f"mode=v30 margin-Bellman + greedy-recovery DeltaNet policy + phase-aware primitive control + semantic-stopping/tiered memetic PyTorch-NEAT + functional-equivalence + transposition MuZero warmup({warmup_steps}) -> supervised transport algebra -> frozen-algebra variable-length (1..4) goal-conditioned program inference")
    print(f"NEAT generations={neat_generations} population={neat_population} workers={neat_workers} innerSteps={neat_inner_steps} innerLR={neat_inner_lr:g} seed={neat_seed}")
    print(f"es_every={es_every} mcts_every={mcts_train_every} mcts_samples={mcts_train_samples} train_sims={mcts_simulations} eval_sims={mcts_eval_simulations}")

    for step in range(1, steps + 1):
        model.train()
        in_program_phase = step > algebra_steps
        if step == algebra_steps + 1:
            # The algebra is already established. From here on only learn how
            # to infer/operator-sequence programs from demonstrations.
            freeze_for_program_phase(model)

        if in_program_phase:
            demo_x, demo_y, query_x, query_y, program_targets, program_lengths = tasks.sample_program_batch(batch_size, device)
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            demo_x, demo_y, query_x, query_y, task_ids = tasks.sample(batch_size, device)
            program_targets = None
            program_lengths = None
        rule = model.encode_rule(demo_x, demo_y)
        H0 = model.encode_query(query_x)

        # Stable task-independent query workspace.
        identity_pred = model.decode_query(H0)
        identity_loss = F.mse_loss(identity_pred, query_x)
        direct_pred = model.decode_direct(H0, rule)
        direct_loss = F.mse_loss(direct_pred, query_y)

        task_logits = model.task_logits_rule(rule)
        if in_program_phase:
            task_loss = torch.zeros((), device=torch_device)
            task_acc = torch.ones((), device=torch_device)
        else:
            task_loss = F.cross_entropy(task_logits, task_ids)
            task_acc = (task_logits.argmax(dim=-1) == task_ids).float().mean()

        in_warmup = step <= warmup_steps
        # Freeze the query coordinate system exactly when operator training starts.
        # The optimizer may still contain these parameters; requires_grad=False
        # prevents subsequent operator losses from moving them.
        if step == warmup_steps + 1:
            set_query_codec_trainable(model, False)
        routing_stats = []

        # Synthetic-stage semantic anchors: four primitive tasks map to four
        # distinct operator identities. This isolates algebra/composition from
        # unsupervised symbol discovery; the other 18 operators remain free.
        primitive_operator_targets = task_ids  # task 0..3 -> operator 0..3
        policy_logits_h0, _ = model.policy_value(H0, rule, primitive_phase=0)
        if in_program_phase:
            support_H0 = model.encode_query(demo_x)
            support_target = model.encode_query(demo_y)

            # Teacher forcing across a padded 4-step program. Targets after the
            # true program length are STOP(4), so the controller learns both
            # action identity and when to terminate.
            support_teacher = support_H0
            ce_terms, acc_terms, bellman_entropies, teacher_states = [], [], [], [support_H0]
            for t in range(4):
                remaining = 4 - t
                logits_t = model.program_policy(support_teacher, support_target, remaining_steps=remaining)[:, :5]
                target_t = program_targets[:, t]
                literal_ce = F.cross_entropy(logits_t, target_t)
                functional_target, _ = exact_horizon_policy_target(
                    model, support_teacher.detach(), support_target.detach(), remaining,
                    beta=2.0, margin=0.01,
                )
                functional_ce = soft_policy_cross_entropy(logits_t, functional_target)
                # v30: functional correctness dominates; retain a small literal anchor
                # only to keep program strings interpretable when several are valid.
                ce_terms.append(0.10 * literal_ce + 0.90 * functional_ce)
                acc_terms.append((logits_t.argmax(dim=-1) == target_t).float().mean())
                bellman_entropies.append(
                    -(functional_target * torch.log(functional_target.clamp_min(1e-8))).sum(dim=-1).mean() / math.log(5)
                )
                support_teacher = apply_program_actions(model, support_teacher, target_t)
                teacher_states.append(support_teacher)
            route_supervision_loss = torch.stack(ce_terms).mean()
            primitive_stop_loss = torch.zeros((), device=torch_device)
            program_ce_1, program_ce_2 = ce_terms[0], torch.stack(ce_terms[1:]).mean()
            program_acc_1, program_acc_2 = acc_terms[0], torch.stack(acc_terms[1:]).mean()
            bellman_target_entropy = torch.stack(bellman_entropies).mean()

            # DAgger-style recovery supervision. Roll the current greedy controller
            # off the teacher trajectory for 1..3 steps (cycled by SGD step), then
            # ask the exact Bellman solver what actions remain near-optimal there.
            recovery_depth = 1 + ((step - algebra_steps - 1) % 3)
            recovery_state = support_H0.detach()
            for d in range(recovery_depth):
                rem_before = max(4 - d, 1)
                with torch.no_grad():
                    recovery_logits = model.program_policy(
                        recovery_state, support_target.detach(), remaining_steps=rem_before
                    )[:, :5]
                    recovery_action = recovery_logits.argmax(dim=-1)
                    recovery_state = apply_program_actions(
                        model, recovery_state, recovery_action
                    ).detach()
            recovery_remaining = max(4 - recovery_depth, 0)
            recovery_live = model.program_policy(
                recovery_state, support_target, remaining_steps=recovery_remaining
            )[:, :5]
            recovery_target, _ = exact_horizon_policy_target(
                model, recovery_state.detach(), support_target.detach(), recovery_remaining,
                beta=2.0, margin=0.01,
            )
            recovery_policy_loss = soft_policy_cross_entropy(recovery_live, recovery_target)

            # Exact finite-horizon Bellman supervision at root and teacher states.
            goal_value_losses = []
            for t, vs in enumerate(teacher_states[:-1]):
                horizon = 4 - t
                _, vv = model.program_policy_value(vs, support_target, horizon)
                vt = exact_horizon_value_target(model, vs, support_target, horizon, action_limit=5, beta=2.0)
                goal_value_losses.append(F.mse_loss(vv, vt))
            # Add random one-step off-policy states at horizons 0..3 without
            # enumerating the entire 5^4 tree during every SGD batch.
            for horizon in range(4):
                aa = torch.randint(0, 4, (batch_size,), device=device)
                vs = apply_selected_actions(model, support_H0, aa)
                _, vv = model.program_policy_value(vs, support_target, horizon)
                vt = exact_horizon_value_target(model, vs, support_target, horizon, action_limit=5, beta=2.0)
                goal_value_losses.append(F.mse_loss(vv, vt))
            goal_value_loss = torch.stack(goal_value_losses).mean()

            # MuZero policy improvement at root plus one randomly selected
            # teacher-reached intermediate horizon.
            mcts_policy_loss = torch.zeros((), device=torch_device)
            mcts_root_entropy = torch.zeros((), device=torch_device)
            if mcts_train_every > 0 and step % mcts_train_every == 0:
                n_mcts = min(mcts_train_samples, batch_size)
                mcts_losses, mcts_entropies = [], []
                for b in range(n_mcts):
                    _, visit0, _ = goal_puct_search(
                        model, support_H0[b:b+1].detach(), support_target[b:b+1].detach(),
                        SearchConfig(simulations=mcts_simulations, max_depth=4, action_limit=5)
                    )
                    live0 = model.program_policy(support_H0[b:b+1], support_target[b:b+1], remaining_steps=4)[:, :5]
                    mcts_losses.append(soft_policy_cross_entropy(live0, visit0[None, :]))
                    mcts_entropies.append(-(visit0 * torch.log(visit0.clamp_min(1e-8))).sum() / math.log(5))
                    t = 1 + (b % 3)
                    rem = 4 - t
                    state_t = teacher_states[t][b:b+1]
                    _, visit_t, _ = goal_puct_search(
                        model, state_t.detach(), support_target[b:b+1].detach(),
                        SearchConfig(simulations=mcts_simulations, max_depth=rem, action_limit=5)
                    )
                    live_t = model.program_policy(state_t, support_target[b:b+1], remaining_steps=rem)[:, :5]
                    mcts_losses.append(soft_policy_cross_entropy(live_t, visit_t[None, :]))
                if mcts_losses:
                    mcts_policy_loss = torch.stack(mcts_losses).mean()
                    mcts_root_entropy = torch.stack(mcts_entropies).mean()
        else:
            # Primitive semantics are one operator followed by HALT. First action is
            # anchored to operator 0..3; post-transform HALT is supervised below.
            route_supervision_loss = F.cross_entropy(
                policy_logits_h0, primitive_operator_targets
            )
            primitive_stop_loss = torch.zeros((), device=torch_device)
            program_ce_1 = program_ce_2 = torch.zeros((), device=torch_device)
            program_acc_1 = program_acc_2 = torch.zeros((), device=torch_device)
            bellman_target_entropy = torch.zeros((), device=torch_device)
            recovery_policy_loss = torch.zeros((), device=torch_device)
            goal_value_loss = torch.zeros((), device=torch_device)
            mcts_policy_loss = torch.zeros((), device=torch_device)
            mcts_root_entropy = torch.zeros((), device=torch_device)
        if in_warmup:
            H = H0
            pred = direct_pred
            recon = direct_loss
        else:
            if in_program_phase:
                # Keep reconstruction tied to the known closed algebra while the
                # controller learns program identification; do not backprop into
                # operator execution in this phase.
                H = H0
                for t in range(4):
                    H = apply_program_actions(model, H, program_targets[:, t])
                pred = model.decode_query(H)
                recon = F.mse_loss(pred, query_y)
                routing_stats = []
            else:
                progress = (step - warmup_steps) / max(algebra_steps - warmup_steps, 1)
                temperature = max(0.30, 1.0 - 0.70 * progress)
                # Primitive training is deliberately atomic: exactly one operator.
                # Multi-step sequencing belongs to the separate program controller.
                H, routing_stats = differentiable_rollout(
                    model, rule, H0, steps=1,
                    temperature=temperature, hard=True,
                )
                pred = model.decode_query(H)
                recon = F.mse_loss(pred, query_y)

        per_sample_error = F.mse_loss(pred, query_y, reduction="none").mean(dim=-1)
        value_target = torch.exp(-per_sample_error.detach())
        value_phase = 1 if ((not in_warmup) and (not in_program_phase)) else 0
        _, value = model.policy_value(H, rule, primitive_phase=value_phase)
        value_loss = F.mse_loss(value, value_target)

        codes = F.normalize(model.core.operator_codes, dim=-1)
        gram = codes @ codes.t()
        eye = torch.eye(model.operator_count, device=torch_device)
        diversity = ((gram - eye) ** 2).mean()
        A = model.core.adjacency()
        graph_reg = A.pow(2).mean()

        op_sep_loss = torch.zeros((), device=torch_device)
        op_pair_train = torch.zeros((), device=torch_device)
        route_entropy = torch.zeros((), device=torch_device)
        task_route_entropy = torch.zeros((), device=torch_device)
        task_route_overlap = torch.zeros((), device=torch_device)
        oracle_atomic_loss = torch.zeros((), device=torch_device)
        latent_transition_loss = torch.zeros((), device=torch_device)
        latent_cosine = torch.zeros((), device=torch_device)
        transport_loss = torch.zeros((), device=torch_device)
        transport_stats = {}

        if (not in_warmup) and (not in_program_phase):
            op_sep_loss, op_pair_train = operator_separation_loss(model, H0)
            route_p = torch.stack(routing_stats, dim=0).mean(dim=0)
            route_entropy = -(route_p * torch.log(route_p + 1e-8)).sum(dim=-1).mean() / math.log(model.operator_count)

            # Same primitive should prefer a consistent routing prototype.
            prototypes = []
            proto_entropies = []
            for tid in range(len(tasks.TRAIN_NAMES)):
                mask = task_ids == tid
                if mask.any():
                    proto = route_p[mask].mean(dim=0)
                    proto = proto / proto.sum().clamp_min(1e-8)
                    prototypes.append(proto)
                    proto_entropies.append(-(proto * torch.log(proto + 1e-8)).sum() / math.log(model.operator_count))
            if proto_entropies:
                task_route_entropy = torch.stack(proto_entropies).mean()
            if len(prototypes) > 1:
                P = torch.stack(prototypes, dim=0)
                P = F.normalize(P, dim=-1)
                sim = P @ P.t()
                mask = ~torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
                task_route_overlap = sim[mask].mean()

            # Train the designated primitive operator directly on every sample so
            # early routing mistakes cannot starve it of useful gradients.
            A_oracle = model.core.adjacency()
            oracle_states = []
            for b_idx in range(H0.shape[0]):
                oracle_states.append(
                    model.core.apply_operator(
                        H0[b_idx:b_idx + 1],
                        int(primitive_operator_targets[b_idx].item()),
                        A_oracle,
                    )
                )
            H_oracle = torch.cat(oracle_states, dim=0)
            oracle_atomic_loss = F.mse_loss(model.decode_query(H_oracle), query_y)

            # Explicit primitive termination: after the anchored operator has been
            # applied, the primitive policy should HALT rather than repeatedly apply
            # the same transform. This keeps atomic reasoning distinct from programs.
            halt_logits, _ = model.policy_value(H_oracle.detach(), rule, primitive_phase=1)
            halt_targets = torch.full(
                (H_oracle.shape[0],), model.HALT_ACTION, device=H_oracle.device, dtype=torch.long
            )
            primitive_stop_loss = F.cross_entropy(halt_logits, halt_targets)

            # Explicit algebraic closure target: an operator applied to the
            # encoding of x should land at the encoding of T_k(x).  The target
            # is detached because the query coordinate system is fixed after
            # warmup.  This is the key compositional objective in v10.
            with torch.no_grad():
                H_target = model.encode_query(query_y)
            # Do not average the closure error away over feature width. v10's
            # ordinary MSE diluted a one-channel positional error by node_dim.
            # Sum over feature channels, then average over batch and vertices.
            latent_transition_loss = (H_oracle - H_target).pow(2).sum(dim=-1).mean()
            h_oracle_flat = F.normalize(H_oracle.flatten(1), dim=-1)
            h_target_flat = F.normalize(H_target.flatten(1), dim=-1)
            latent_cosine = (h_oracle_flat * h_target_flat).sum(dim=-1).mean()

            # v12: directly teach the code-conditioned transport law the exact
            # permutation associated with each anchored primitive. This is
            # synthetic-stage supervision used to test whether the generated
            # operator family can represent a closed algebra at all.
            transport_loss, transport_stats = transport_supervision_loss(model)

        if in_program_phase:
            # Program phase: freeze the learned algebra and optimize only rule
            # inference/controller sequencing. A small primitive rehearsal batch
            # below keeps single-step routing from drifting.
            p_demo_x, p_demo_y, _, _, p_task_ids = tasks.sample(batch_size, device)
            p_support = model.encode_query(p_demo_x)
            p_target = model.encode_query(p_demo_y)
            p_logits = model.program_policy(p_support, p_target, remaining_steps=1)
            primitive_rehearsal = F.cross_entropy(p_logits, p_task_ids)
            p_after = apply_program_actions(model, p_support, p_task_ids)
            p_stop_logits = model.program_policy(p_after, p_target, remaining_steps=0)
            p_stop_targets = torch.full_like(p_task_ids, 4)
            primitive_stop_rehearsal = F.cross_entropy(p_stop_logits, p_stop_targets)
            # Only program_controller has gradients in this phase. Do not add
            # value/diversity losses from frozen modules to the optimization.
            loss = (
                1.00 * route_supervision_loss
                + 0.50 * recovery_policy_loss
                + 0.15 * primitive_rehearsal
                + 0.15 * primitive_stop_rehearsal
                + 0.50 * goal_value_loss
                + 0.50 * mcts_policy_loss
            )
        elif in_warmup:
            loss = (
                direct_loss
                + 0.50 * identity_loss
                + 0.10 * task_loss
                + 0.10 * route_supervision_loss
                + 0.25 * value_loss
                + 0.01 * diversity
                + 1e-4 * graph_reg
            )
        else:
            loss = (
                recon
                + 0.00 * identity_loss
                + 0.10 * direct_loss
                + 0.10 * task_loss
                + 0.25 * value_loss
                + 0.01 * diversity
                + 1e-4 * graph_reg
                + 0.08 * op_sep_loss
                + 1.50 * route_supervision_loss
                + 0.25 * primitive_stop_loss
                + 0.50 * oracle_atomic_loss
                + 1.00 * latent_transition_loss
                + 0.00 * transport_loss  # evolved CPPN is frozen; this is diagnostic only
                + 0.010 * task_route_entropy
                + 0.010 * task_route_overlap
            )

        mean_op_delta = max_op_delta = normalized_entropy = operator_pair_distance = None
        if diagnostic_every > 0 and (step == 1 or step % diagnostic_every == 0):
            with torch.no_grad():
                A_diag = model.core.adjacency()
                op_states = torch.stack(
                    [model.core.apply_operator(H0, k, A_diag) for k in range(model.operator_count)],
                    dim=1,
                )
                deltas = (op_states - H0[:, None]).pow(2).mean(dim=(-1, -2)).sqrt()
                mean_op_delta = deltas.mean()
                max_op_delta = deltas.max()
                pair_diff = op_states[:, :, None] - op_states[:, None, :]
                pair_dist = (pair_diff.pow(2).mean(dim=(-1, -2)) + 1e-8).sqrt()
                upper = torch.triu(torch.ones(model.operator_count, model.operator_count, device=H0.device, dtype=torch.bool), diagonal=1)
                operator_pair_distance = pair_dist[:, upper].mean()
                logits_diag, _ = model.policy_value(H0, rule, primitive_phase=0)
                p_diag = torch.softmax(logits_diag[:, :model.operator_count], dim=-1)
                entropy = -(p_diag * torch.log(p_diag + 1e-8)).sum(dim=-1).mean()
                normalized_entropy = entropy / math.log(model.operator_count)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        es_info = None
        if es_every > 0 and step % es_every == 0:
            model.eval()
            es_info = antithetic_es_step_(model, tasks, device, pairs=4, sigma=0.005, lr=0.01, eval_batch_size=24, rollout_steps=inner_rollout_steps)

        if step == 1 or step % 25 == 0:
            msg = (
                f"step={step:04d} loss={loss.item():.5f} recon={recon.item():.5f} "
                f"direct={direct_loss.item():.5f} identity={identity_loss.item():.5f} "
                f"task={task_loss.item():.4f} taskAcc={task_acc.item():.3f} "
                f"value={value_loss.item():.5f} div={diversity.item():.5f} "
                f"phase={'program' if in_program_phase else ('warmup' if in_warmup else 'ops')}"
                f" routeCE={route_supervision_loss.item():.4f}"
            )
            if in_program_phase:
                msg += (
                    f" progCE1={program_ce_1.item():.4f} progCE2={program_ce_2.item():.4f}"
                    f" progAcc1={program_acc_1.item():.3f} progAcc2={program_acc_2.item():.3f}"
                    f" recoverCE={recovery_policy_loss.item():.4f}"
                    f" bellmanH={bellman_target_entropy.item():.3f}"
                    f" deltaGate={torch.sigmoid(model.delta_state.output_gate).item():.3f}"
                    f" goalV={goal_value_loss.item():.4f} mctsCE={mcts_policy_loss.item():.4f}"
                    f" mctsH={mcts_root_entropy.item():.3f}"
                )
            elif not in_warmup:
                msg += (
                    f" sep={op_sep_loss.item():.4f} trainPair={op_pair_train.item():.5f}"
                    f" routeH={route_entropy.item():.3f}"
                    f" taskRouteH={task_route_entropy.item():.3f}"
                    f" taskOverlap={task_route_overlap.item():.3f}"
                    f" routeCE={route_supervision_loss.item():.4f}"
                    f" stopCE={primitive_stop_loss.item():.4f}"
                    f" oracle={oracle_atomic_loss.item():.5f}"
                    f" latent={latent_transition_loss.item():.5f}"
                    f" latentCos={latent_cosine.item():.3f}"
                    f" transportCE={transport_loss.item():.4f}"
                    + (
                        " " + " ".join(
                            f"T{pid}Acc={transport_stats[pid][0].item():.2f}/H={transport_stats[pid][1].item():.2f}"
                            for pid in sorted(transport_stats)
                        ) if transport_stats else ""
                    )
                )
            if mean_op_delta is not None:
                msg += (
                    f" opΔ={mean_op_delta.item():.5f} opΔmax={max_op_delta.item():.5f}"
                    f" opPair={operator_pair_distance.item():.5f} policyH={normalized_entropy.item():.3f}"
                )
            print(msg)

    heldout_program = evaluate_greedy_program_accuracy(model, tasks, device, batch_size=256)
    print("\nHeld-out goal-conditioned program inference (exact programs never used in program-training phase)")
    for name, vals in heldout_program.items():
        if len(vals) == 3:
            exact, mse, functional = vals
            print(f"  {name:20s} stringExact={exact:.3f} functional={functional:.3f} greedyMSE={mse:.5f}")
        else:
            exact, mse = vals
            print(f"  {name:20s} stringExact={exact:.3f} greedyMSE={mse:.5f}")

    mcts_eval = evaluate_muzero_program_search(
        model, tasks, device, batch_size=mcts_eval_batch, simulations=mcts_eval_simulations
    )
    print("\nHeld-out variable-length MuZero-style PUCT (functional equivalence + transpositions)")
    for name, vals in mcts_eval.items():
        exact, mse, functional = vals
        print(f"  {name:20s} stringExact={exact:.3f} functional={functional:.3f} mctsMSE={mse:.5f}")

    checkpoint_path = "yetirah_v30_posttrain.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "steps": steps}, checkpoint_path)
    print(f"\nsaved checkpoint: {checkpoint_path}")

    held_out = evaluate_held_out(model, tasks, device, batch_size=1024)
    print("\nHeld-out factorized representation diagnostics (1024 fresh tasks)")
    print(
        f"direct MSE: {held_out['direct_mse']:.5f} | query-identity MSE: {held_out['identity_mse']:.5f} | "
        f"rule task-ID: {held_out['task_acc_rule']:.3f}"
    )
    for name, (direct_mse, identity_mse, task_acc, count) in held_out["per_task"].items():
        print(f"  {name:8s} n={count:4d} direct={direct_mse:.5f} identity={identity_mse:.5f} taskRule={task_acc:.3f}")

    op_eval = evaluate_operator_rollout(model, tasks, device, batch_size=1024, rollout_steps=2)
    print("\nHeld-out atomic operator+HALT diagnostics (1024 fresh tasks)")
    print(
        f"direct MSE: {op_eval['direct_mse']:.5f} | operator-rollout MSE: {op_eval['rollout_mse']:.5f} | "
        f"opPair: {op_eval['op_pair']:.5f} | sepLoss: {op_eval['sep_loss']:.4f}"
    )
    print("top routed operators:", list(zip(op_eval['top_ops'], [round(x, 3) for x in op_eval['top_usage']])))
    print(f"first-action accuracy: {op_eval['first_action_acc']:.3f} | post-action HALT accuracy: {op_eval['post_halt_acc']:.3f}")
    print(f"rollout normalized MSE: {op_eval['rollout_nmse']:.5f}")
    print("primitive operator patterns:")
    for name, stats in op_eval["per_task"].items():
        print(f"  {name:8s} direct={stats['direct_mse']:.5f} rollout={stats['rollout_mse']:.5f} nmse={stats['rollout_nmse']:.4f}")
        for pattern, count, frac in stats["sequences"][:3]:
            print(f"    {pattern}  n={count:3d} frac={frac:.3f}")

    exact_var = evaluate_exact_variable_program_search(
        model, tasks, device, batch_size=mcts_eval_batch, max_depth=4
    )
    print("\nExact variable-length primitive+STOP oracle (support-selected, max depth 4)")
    print("  same action space/horizon as PUCT; shorter equivalent programs are allowed")
    for name, stats in exact_var.items():
        print(
            f"  {name:20s} exactSearch={stats['search_mse']:.5f} "
            f"knownProgram={stats['oracle_mse']:.5f} stringExact={stats['exact_string']:.3f} "
            f"support={stats['mean_support_mse']:.5f}"
        )

    root_diag = evaluate_root_search_diagnostics(
        model, tasks, device, simulations=mcts_eval_simulations, samples_per_task=8, max_depth=4
    )
    print("\nHeld-out root planning diagnostics")
    print("  priorBest = learned prior already picks an exact Bellman-best first action")
    print("  puctBest  = PUCT picks an exact Bellman-best first action")
    print("  valueRatio = exact continuation value of PUCT choice / optimum")
    for name, stats in root_diag.items():
        print(
            f"  {name:20s} priorBest={stats['prior_best_rate']:.3f} "
            f"puctBest={stats['puct_best_rate']:.3f} valueRatio={stats['puct_value_ratio']:.3f}"
        )

    # Goal-conditioned PUCT diagnostic on a held-out composition.
    model.eval()
    demo_x, demo_y, _, _ = tasks.sample_composition("roll2", 1, device)
    support = model.encode_query(demo_x)
    target_support = model.encode_query(demo_y)
    action, probs, root_err = goal_puct_search(
        model, support, target_support, SearchConfig(simulations=mcts_eval_simulations, max_depth=4, action_limit=5)
    )
    print("\nVariable-length MuZero/PUCT smoke test (roll2, max depth 4, transpositions on)")
    print("root support-goal error:", root_err)
    print("chosen action:", action)
    top = torch.topk(probs, k=min(4, probs.numel()))
    print("top policy visits:", top.indices.tolist())
    print("top visit probs:", top.values.tolist())
    print("root action prior / exact best continuation value:")
    for a, prior, exact_v in root_action_diagnostics(model, support, target_support, remaining_steps=4, action_limit=5):
        print(f"  a={a} prior={prior:.4f} exactV={exact_v:.4f}")

    return model
