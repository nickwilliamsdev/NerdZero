"""Characterise the stuck set-13 tasks (low per-task scores) so the next
fix is targeted: grid sizes, colour counts, demo counts, and whether the
input->output is a shape transform (different dims)."""
import arckit

train_ds, _test_ds = arckit.load_data()
tasks = {}
for t in train_ds.tasks:
    tasks[str(getattr(t, "id", ""))] = t

# per-task 300-step scores for the best genome (gated96) on set 13, from the
# unclipped lr0.1 run (diag_noclip_gated96_lr01.txt) — the protocol the
# baseline was originally judged under. Used only to flag which task
# families are hard, not as a comparison number.
score = {
    "99306f82": 0.9333, "d13f3404": 0.6111, "9bebae7a": 0.972, "14b8e18c": 0.84,
    "28e73c20": 0.6944, "f45f5ca7": 0.92, "9f27f097": 0.6597, "c8cbb738": 0.4444,
    "8e301a54": 0.92, "e633a9e5": 0.84, "e45ef808": 0.8611, "cf5fd0ad": 0.5556,
    "b942fd60": 0.8056, "fcc82909": 0.93, "27f8ce4f": 0.4815, "4e469f39": 0.97,
}

rows = []
for tid, s in score.items():
    t = tasks[tid]
    pairs = list(getattr(t, "train", []))
    in_dims = set((p[0].shape[0], p[0].shape[1]) for p in pairs)
    out_dims = set((p[1].shape[0], p[1].shape[1]) for p in pairs)
    cols_in = sorted({int(c) for p in pairs for c in p[0].flatten()})
    cols_out = sorted({int(c) for p in pairs for c in p[1].flatten()})
    shape_transform = in_dims != out_dims
    rows.append((tid, s, len(pairs), sorted(in_dims), sorted(out_dims),
                 cols_in, cols_out, shape_transform))

rows.sort(key=lambda r: r[1])
print(f"{'id':>9s} {'score':>5s} demos  in_dims     out_dims    col_in  col_out  shapeX")
for (tid, s, nd, idm, odm, ci, co, shp) in rows:
    print(f"{tid:>9s} {s:5.2f}  {nd:3d}  {str(idm):>14s}  {str(odm):>14s}  "
          f"{str(ci):>9s}  {str(co):>9s}  {'YES' if shp else 'no'}")
