Yetirah ARC: independent scratch-init integration patch
This patch replaces the v30 initialization and seed-loading paths in the retrieved ARC source files. It does not implement a new architecture: it retains the current ARCReasoner, YetirahCore, task-conditioned operator bank, and CPPN feature interface, but creates all neural parameters and the initial CPPN within this ARC project, without importing model weights from v30.
Before applying
The copies of run_arc_v1.py and arc_trainer.py available to construct this patch are older than the versions visible in your current terminal logs: the log has ~303 launcher lines and ~822 trainer lines while these retrieved files are shorter. Review/merge with diff -u, rather than blindly replacing newer local source. Back up the project or commit before replacing anything.
Files included:
- sefer/config.py: removes V30Config/V30DGXConfig and v30 file/config flags; NEAT config path is local by default.
- sefer/training/arc_trainer.py: removes the v30 checkpoint locator and initializer; creates a fresh CPPN before training and reconstructs it from the ARC-owned seed on loading.
- sefer/evolution/arc_neat_outer.py: no v30 winner file lookup, no parent-project recursive config search; creates/saves initial ARC CPPN and evolves from that seed.
- run_arc_v1.py: removes v30 flags; --arc-neat-only requires a fresh ARC checkpoint (it cannot create one).
- arc_neat_config.ini: standalone neat-python configuration with input count 40, hidden count 22, threshold 0.75; new genomes use initial_connection=full.
First training run
Run from the ARC project root without --arc-neat-only:
python run_arc_v1.py --arc-neat-config ./arc_neat_config.ini --no-arc-neat
This trains the codec, direct path, operator discovery and controller from fresh ARC neural parameters. It generates yetirah_arc_initial_cppn.pkl and an ARC-only yetirah_arc_v1_pre_neat.pt. Leaving --no-arc-neat off instead will run NEAT after normal ARC training; for debugging, start with it enabled as above.
After independent ARC training, start NEAT:
python run_arc_v1.py --arc-neat-only --arc-neat-config ./arc_neat_config.ini
The same independently generated yetirah_arc_initial_cppn.pkl must remain beside its corresponding fresh ARC checkpoint. An inherited checkpoint lacking the new marker is explicitly rejected. Do not use any previous yetirah_arc_v1_pre_neat.pt created with v30, even if you rename it. Back up or move existing ARC artifacts before first scratch run so that old/evolved artifacts cannot be confused with new runs.
Removing v30 directories
After merging, audit the ENTIRE local ARC tree, not just these files:
rg -n -i 'v30|yetirah_v30|V30Config|initialize_from_v30' . --glob '*.py'
rg -n '^from |^import ' sefer/algebra sefer/evolution
python -m compileall -q run_arc_v1.py sefer
Your current architecture still imports the ARC-local sefer.algebra.core.YetirahCore and sefer.evolution.evolve_transport.require_pytorch_neat. Those implementations and their transitive dependencies were not supplied with the retrieved source set and must remain in your ARC project or be ported/rewritten before deleting any directory they import. PyTorch-NEAT and neat-python are third-party runtime dependencies, not v30 model weights. Check your local import resolution (python -c "import inspect; from sefer.algebra.core import YetirahCore; from sefer.evolution.evolve_transport import require_pytorch_neat; print(inspect.getfile(YetirahCore)); print(inspect.getfile(require_pytorch_neat))"). This patch has not been end-to-end tested against your current live tree.
Scientific limitation
The initial random CPPN feeds the operator bank before training; subsequent Baldwinian NEAT evolves structure while trained ARC weights stay frozen. That is scratch initialization, but not a new reasoning architecture. To test whether the previous plateau is architectural, independently evaluate a task-conditioned recurrence or program-learning objective after establishing this baseline.