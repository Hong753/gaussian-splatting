#!/usr/bin/env python3
"""Read actual population history without loading CUDA or a Gaussian checkpoint."""
import argparse
import json
from pathlib import Path


def audit(model):
    path=Path(model)/'densification.jsonl'
    rows=[]
    if path.is_file():
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    empty=next((r for r in rows if r.get('points_after')==0),None)
    fields=('iteration','image','points_before','points_after','pruned','cloned_parents','split_parents',
            'prune_opacity_candidates','prune_screen_candidates','prune_world_candidates',
            'prune_opacity_threshold','prune_screen_radius','prune_world_sigma','radius_history_preserved')
    failure=Path(model)/'pruning_failure.json'
    return dict(model=str(Path(model).resolve()),population_events=len(rows),
        first_zero_population=empty,
        final_population=rows[-1].get('points_after') if rows else None,
        interpretation=('The entire model reached zero Gaussians; this is not an empty camera view.' if empty else
                        'No zero-population pruning event found in the available log.'),
        reason_counts_recorded=any('prune_opacity_candidates' in r for r in rows),
        recent=[{k:r[k] for k in fields if k in r} for r in rows[-8:]],
        rejected_pruning=json.loads(failure.read_text()) if failure.is_file() else None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(audit(args.model),indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
