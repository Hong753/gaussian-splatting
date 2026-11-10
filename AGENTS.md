# Reconstruction experiments

Always produce actual-image versus rendered-image comparisons after training.
Use `scripts/render.py --split both` for the saved train and held-out eval views;
the training launcher enables this by default. Include per-view images, PSNR,
error maps, and a browsable report. When comparing runs, use shared image names
and a shared saved iteration, and state whether their camera poses differ.

Do not report training or rendering as completed without observing the outputs.
If the dataset, checkpoints, or CUDA host are unavailable, finish and validate
the runnable workflow and clearly identify what still requires server execution.

For Proxy-GS, run preparation, training, and automatic rendering without Python
supervisor subprocesses. Use process replacement for environment/runtime changes
so training errors remain the direct traceback. Retain proxy/depth caches when
restarting a failed attempt.
