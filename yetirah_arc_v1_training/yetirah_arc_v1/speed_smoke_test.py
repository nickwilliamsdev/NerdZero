import torch
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.controllers.reasoner import TinyReasoner
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.planning.bellman import exact_horizon_policy_target, exact_horizon_value_target
from yetirah_arc_v1_training.yetirah_arc_v1.sefer.training.phases import freeze_for_program_phase


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyReasoner().to(device)
    # This smoke test uses the bootstrap operator generator; no NEAT dependency needed.
    freeze_for_program_phase(model)
    model.core.materialize_operator_bank()
    B = 8
    x = torch.randn(B, 32, device=device)
    y = torch.randn(B, 32, device=device)
    H = model.encode_query(x)
    G = model.encode_query(y)
    actions = torch.randint(0, 4, (B,), device=device)
    out = model.core.apply_operator_batch(H, actions)
    assert out.shape == H.shape
    assert model.core._transport_bank.shape == (22, 32, 32)
    v = exact_horizon_value_target(model, H, G, 4, action_limit=4)
    p, q = exact_horizon_policy_target(model, H, G, 4)
    assert v.shape == (B,) and p.shape == q.shape == (B, 5)
    assert torch.allclose(p.sum(-1), torch.ones(B, device=device), atol=1e-5)
    print("speed smoke test passed")
    print("device:", device)
    print("operator bank:", tuple(model.core._transport_bank.shape))
    print("bellman target:", tuple(v.shape), tuple(p.shape))

if __name__ == "__main__":
    main()
