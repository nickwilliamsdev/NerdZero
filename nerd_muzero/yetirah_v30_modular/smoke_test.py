import torch
from sefer.controllers.reasoner import TinyReasoner
from sefer.tasks.synthetic_algebra import SyntheticTaskBatch
from sefer.planning.bellman import exact_horizon_policy_target

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyReasoner().to(device)
    tasks = SyntheticTaskBatch(dim=32)
    demo_x, demo_y, query_x, query_y, task_ids = tasks.sample(4, device)
    rule = model.encode_rule(demo_x, demo_y)
    H = model.encode_query(query_x)
    goal = model.encode_query(query_y)
    logits, value = model.program_policy_value(H, goal, 2)
    target, q = exact_horizon_policy_target(model, H, goal, 2)
    assert rule.shape[0] == 4
    assert logits.shape == (4, 5)
    assert value.shape == (4,)
    assert target.shape == (4, 5) and q.shape == (4, 5)
    assert torch.allclose(target.sum(-1), torch.ones(4, device=device), atol=1e-5)
    print("smoke test passed")
    print("device:", device)
    print("params:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("program logits:", tuple(logits.shape))
    print("bellman target:", tuple(target.shape))

if __name__ == "__main__":
    main()
