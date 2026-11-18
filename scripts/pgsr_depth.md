# Orange4F PGSR with LiDAR depth

Run `python -u scripts/run_orange4f_pgsr.py --depth-supervision` in the 3dgs environment. Add `--gpu N` to select a physical GPU or `--plan-only` to inspect the input mode. The control is the existing guided 50k PGSR experiment. Both use the same RGB images, train/test split, original poses, LiDAR seeds and densification schedule.

The launcher looks for `lidar_scans.json` in the original scene directory. You can specify a different file with `--scan-manifest FILE`. The manifest describes registered first returns in metres. Each NPY contains finite XYZ samples in the local scanner frame. Group returns with one common optical origin per entry and provide its rigid transform into the same world frame as the training cloud. Moving sweeps may require short packets with separate poses. The input format is:

```json
{"format":"scanner_xyz_v1","scans":[{"xyz":"scans/0001.npy","scanner_to_world":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}
```

Scanner mode renders a 90-degree face at the recorded origin every fourth RGB step. Each occupied pixel contributes its nearest measured optical-axis depth. With a fused cloud alone, projected mode uses the fixed original points and the current training camera. It keeps the nearest sample per 4-pixel-wide cell and rejects isolated hits and background samples adjacent to a closer surface. This is approximate visibility.

Both runs start with the usual 7k-step warmup. The relative Huber depth weight ramps to 0.05 at 9k and tapers from 0.05 at 30k to 0.01 at 50k. PGSR plane depth receives supervision at supported pixels with accumulated opacity of at least 0.5. The launcher prints the mode, live preview path, and accepted depth count. `lidar_depth_metrics.csv` records counts and training residuals every 500 active steps.

The run directory contains `debug_previews/index.html` and the 30k/50k `psnr_all_frames/iteration_<step>/summary.json` reports. When the control evaluation exists, `depth_comparison.json` in each evaluation directory gives paired held-out and train PSNR on the identical frames and, when available in both previews, a foreground floor texture diagnostic for 09882.jpg. Repeating the command after the control finishes generates the comparison from completed outputs.
