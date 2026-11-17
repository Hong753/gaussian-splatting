# Reconstruction experiments

Place new fine-tuning experiment outputs under
`/data/3dgs_outputs/<scene>/3dgs_finetuning/`, with a separate model directory
for each experiment. Dataset directories are inputs, not output roots; they may
be read-only or symlink to read-only storage. For orange4f, use
`/data/3dgs_outputs/orange4f/3dgs_finetuning/`. Keep baseline
lookup at the original scene root. Fresh-run commands must not require a saved
plan from a previous run; stop an older run only when it is known to exist.

Keep live GT/render/error debugging enabled for every native 3DGS run: step 1,
every 500 steps, and the final step, with fixed random train and held-out frames.
For train-only mesh culling, also produce matching culling-ON training previews.
Only disable these previews if the user explicitly requests it.

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
