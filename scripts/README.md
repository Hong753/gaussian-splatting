# Training

Edit the defaults at the top of `scripts/train.py` (especially `DATA_ROOT` and
`SCENES`), then run `python scripts/train.py`. Or provide settings as arguments:

```bash
python scripts/train.py --data-root /path/to/scenes --scenes scan65 scan69 \
  --output-root output --name baseline --gpu 0 --iterations 30000
```

The launcher trains scenes sequentially, with outputs in
`OUTPUT_ROOT/SCENE/NAME`. It uses the active Python environment and the native
`train.py` implementation. The scene must have a COLMAP `sparse/0` model and
images, or a supported NeRF synthetic dataset. Use `--dry-run` to check the
commands, and `--help` for options. Pass additional native options after `--`:

```bash
python scripts/train.py --scenes scan65 -- --antialiasing
```

The launcher refuses to use a nonempty output directory unless
`--allow-existing` is specified; that option does not resume training by
itself. Native `--start_checkpoint` is required to resume a checkpoint.
