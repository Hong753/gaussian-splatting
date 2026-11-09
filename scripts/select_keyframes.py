#!/usr/bin/env python3
"""Select a reusable keyframes.json/image_list.txt without changing the dataset."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_pipeline.selection import main

if __name__ == "__main__":
    main()
