#!/usr/bin/env python3
"""Audit saved Gaussian counts and growth/pruning history without CUDA or PLY payloads."""
import argparse
import ast
import json
from pathlib import Path


PRUNE_REASONS = ('replaced_split_parents', 'pruned_low_opacity',
                 'pruned_world_size', 'pruned_screen_size')


def ply_vertex_count(path):
    """Read only the header, including for multi-gigabyte binary checkpoints."""
    with Path(path).open('rb') as stream:
        if stream.readline(64).strip() != b'ply':
            raise ValueError(f'Not a PLY file: {path}')
        count = None
        for _ in range(10000):
            line = stream.readline(65537)
            if not line or len(line) > 65536:
                break
            fields = line.split()
            if fields[:2] == [b'element', b'vertex']:
                count = int(fields[2])
                if count < 0:
                    raise ValueError(f'Negative vertex count: {path}')
            if fields == [b'end_header']:
                if count is None:
                    raise ValueError(f'PLY has no vertex element: {path}')
                return count
    raise ValueError(f'Missing or oversized PLY header: {path}')


def saved_settings(path):
    """Read literal Namespace keywords without executing the saved config."""
    if not path.is_file():
        return {}
    node = ast.parse(path.read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace':
        raise ValueError(f'Expected a literal Namespace config: {path}')
    keys = ('surface_mode', 'position_warmup_steps', 'densification_phase_offset',
            'occlusion_mode', 'occlusion_backend', 'optimization_settings')
    values = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg in keys}
    return values


def audit_model(model):
    model = Path(model).resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    source = model/'input.ply'
    initial = ply_vertex_count(source) if source.is_file() else None
    checkpoints = []
    for path in (model/'point_cloud').glob('iteration_*/point_cloud.ply'):
        iteration = int(path.parent.name.removeprefix('iteration_'))
        count = ply_vertex_count(path)
        checkpoints.append(dict(iteration=iteration, points=count, path=str(path),
            fraction_of_input=count/initial if initial else None))
    checkpoints.sort(key=lambda row: row['iteration'])
    events, warnings = [], []
    log = model/'view_tangent_densification.jsonl'
    if log.is_file():
        with log.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    for key in ('iteration', 'points_before', 'added_children', 'pruned', 'points_after'):
                        if not isinstance(event[key], int) or event[key] < 0:
                            raise ValueError(f'Invalid {key}')
                except (ValueError, KeyError, TypeError) as error:
                    warnings.append(f'Log line {number} unreadable: {error}')
                    continue
                if event['points_before']+event['added_children']-event['pruned'] != event['points_after']:
                    warnings.append(f"Step {event['iteration']}: inconsistent population accounting")
                if events and events[-1]['points_after'] != event['points_before']:
                    warnings.append(f"Step {event['iteration']}: count differs from preceding log event")
                if all(key in event for key in PRUNE_REASONS):
                    if sum(event[key] for key in PRUNE_REASONS) != event['pruned']:
                        warnings.append(f"Step {event['iteration']}: pruning causes do not sum to pruned")
                events.append(event)
    else:
        warnings.append('No tangent densification log; checkpoint counts alone do not identify the deletion cause')
    reasons_available = bool(events) and all(all(key in e for key in PRUNE_REASONS) for e in events)
    if events and not reasons_available:
        warnings.append('Legacy pruning totals combine opacity, size and split replacement; exact deletion causes were not recorded')
    drops = []
    for event in events:
        net = event['points_before']-event['points_after']
        if net > 0:
            drops.append(dict(event, net_loss=net, net_loss_fraction=net/max(event['points_before'], 1)))
    drops.sort(key=lambda row: row['net_loss'], reverse=True)
    history = dict(events=len(events), added_children=sum(e['added_children'] for e in events),
        pruned_including_split_replacement=sum(e['pruned'] for e in events),
        first_event=events[0] if events else None, last_event=events[-1] if events else None,
        reasons_available_for_all_events=reasons_available, largest_net_drops=drops[:10])
    if reasons_available:
        history['pruning_causes'] = {key: sum(e[key] for e in events) for key in PRUNE_REASONS}
    if initial is not None and checkpoints and checkpoints[-1]['points'] < .01*initial:
        warnings.append('Latest checkpoint has less than 1% of the input population; inspect training pruning before another full run')
    return dict(model=str(model), input_points=initial, settings=saved_settings(model/'cfg_args'),
        checkpoints=checkpoints, densification=history, warnings=warnings,
        interpretation='Counts are full saved PLY rows, not per-view visible points. No geometry or opacity validation is implied.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: MODEL/population_audit.json')
    args = parser.parse_args(argv)
    report = audit_model(args.model)
    print(f"Model: {report['model']}")
    print(f"Initialization PLY: {report['input_points']:,} points" if report['input_points'] is not None else 'Initialization PLY missing')
    for row in report['checkpoints']:
        relative = f" ({row['fraction_of_input']:.4%} of input count)" if row['fraction_of_input'] is not None else ''
        print(f"Step {row['iteration']:>6}: {row['points']:>12,} Gaussians{relative}")
    print('Largest net population drops:')
    for row in report['densification']['largest_net_drops']:
        print(f"  Step {row['iteration']:>6}: {row['points_before']:,} -> {row['points_after']:,}; "
              f"added {row['added_children']:,}, pruned {row['pruned']:,}, net loss {row['net_loss_fraction']:.1%}")
    for warning in report['warnings']:
        print(f'NOTE: {warning}')
    output = args.output or args.model/'population_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
