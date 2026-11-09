#!/usr/bin/env python3
"""Run conservative image/LiDAR BA; export a separate, training-ready dataset."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_pipeline.refinement import main

if __name__ == "__main__":
    main()
