#!/usr/bin/env python3
"""Select a LiDAR region, inspect overlapping RGB views, export original poses."""
import argparse
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.region_dataset import RegionData
from scripts.region_review import CameraReview
from scripts.keyframe_pipeline.common import file_stamp


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, help='Supplies original input.ply and keyframes.json')
    p.add_argument('--point-cloud', type=Path)
    p.add_argument('--keyframes', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--anchor', default='06663.jpg')
    p.add_argument('--center', type=float, nargs=3)
    p.add_argument('--size', type=float, nargs=3, default=[4.,4.,4.])
    p.add_argument('--halo', type=float, default=1.)
    p.add_argument('--min-overlap', type=float, default=.02)
    p.add_argument('--max-camera-distance', type=float, default=8.)
    p.add_argument('--frame-start', default='', help='Optional inclusive lexical image-name bound')
    p.add_argument('--frame-end', default='')
    p.add_argument('--exclude', nargs='*', default=[])
    p.add_argument('--sample-points', type=int, default=1000000)
    p.add_argument('--cache-dir',type=Path,help='Packed projection masks; defaults to OUTPUT.projection_cache')
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--review-file', type=Path, help='Autosaved review; defaults to OUTPUT.review.json')
    p.add_argument('--export', action='store_true', help='Headless export of kept review images, or all candidates with --automatic')
    p.add_argument('--automatic', action='store_true', help='With --export, use automatic candidates without manual review')
    p.add_argument('--reuse', action='store_true', help='Reuse a matching automatic export; requires --automatic --export')
    return p


def automatic_export(data, args, config):
    """Reuse only exports with the same inputs, box, filtering and sampling."""
    if args.output.exists():
        meta_path=args.output/'selection.json'
        if not args.reuse or not meta_path.is_file():
            raise FileExistsError(f'Use a fresh region directory: {args.output}')
        meta=json.loads(meta_path.read_text())
        expected=dict(schema=1,region=str(args.output.resolve()),selection_mode='automatic',
                      source=str(data.source),original_point_cloud=str(data.point_cloud),
                      original_keyframes=str(data.keyframes),center=config['center'],size=config['size'],
                      settings={k:v for k,v in config.items() if k not in ('center','size')},
                      sample_points=data.sample_count,sample_points_requested=data.max_points,
                      inputs=[*data.pose_stamps,file_stamp(data.point_cloud),file_stamp(data.keyframes)])
        changed=[key for key,value in expected.items() if meta.get(key)!=value]
        if changed:
            raise ValueError(f'Existing region differs in {changed}; use a fresh --output')
        print(f'Reusing automatic region: {args.output} ({len(meta["selection"])} views)',flush=True)
        return
    print('Finding automatic regional candidates (approximate LiDAR visibility)...',flush=True)
    rows=data.find(**config)
    data.export(args.output,rows=rows,**config)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.automatic and (not args.export or args.review_file):
        raise ValueError('--automatic requires --export and cannot use --review-file')
    if args.reuse and not (args.automatic and args.export):
        raise ValueError('--reuse requires --automatic --export')
    if args.baseline:
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
    if args.point_cloud is None or args.keyframes is None:
        raise ValueError('Supply --baseline or both --point-cloud and --keyframes')
    if args.output.exists() and not args.reuse:
        raise FileExistsError(f'Use a new region directory: {args.output}')
    output=args.output.expanduser().resolve()
    cache=(args.cache_dir or output.with_name(output.name+'.projection_cache')).expanduser().resolve()
    if cache==output or output in cache.parents:
        raise ValueError('Keep projection cache outside the final export directory')
    cache.mkdir(parents=True,exist_ok=True)
    print(f'Region destination: {output}\nProjection cache: {cache}\nLoading cameras and LiDAR sample...',flush=True)
    data = RegionData(args.source, args.point_cloud, args.keyframes, args.sample_points,cache_dir=cache)
    print(f'Loaded {len(data.images)} cameras; {data.sample_count:,} projection points '
          f'from {len(data.vertices):,} original LiDAR points.',flush=True)
    settings = dict(halo=args.halo,min_overlap=args.min_overlap,max_camera_distance=args.max_camera_distance,
                    frame_start=args.frame_start,frame_end=args.frame_end,exclude=args.exclude)
    if args.automatic:
        if args.center is None and args.anchor not in {im.name for im in data.images}:
            raise ValueError(f'Anchor {args.anchor!r} not registered; specify --anchor IMAGE_NAME')
        center=args.center if args.center is not None else data.anchor_center(args.anchor)
        automatic_export(data,args,dict(center=list(map(float,center)),size=args.size,**settings))
        return 0
    review_file=(args.review_file or args.output.with_name(args.output.name+'.review.json')).expanduser().resolve()
    if review_file == args.output.resolve() or args.output.resolve() in review_file.parents:
        raise ValueError('Keep --review-file outside the fresh export directory (default: OUTPUT.review.json)')
    identity=dict(source=str(data.source),point_cloud=file_stamp(data.point_cloud),
                  keyframes=file_stamp(data.keyframes),poses=data.pose_stamps,sample_points=data.max_points)
    session=CameraReview.load(review_file,identity) if review_file.is_file() else None
    if session is not None:
        initial=session.config
        args.anchor=session.current
    else:
        if args.anchor not in {im.name for im in data.images}:
            raise ValueError(f'Anchor {args.anchor!r} not registered; specify --anchor IMAGE_NAME')
        center = args.center if args.center is not None else data.anchor_center(args.anchor)
        initial=dict(center=list(map(float,center)),size=args.size,**settings)
    if args.export:
        if session is None:
            raise ValueError('No saved manual review. Open the viewer and keep/reject candidates before --export.')
        data.export(args.output,rows=session.rows,review=session.decisions,**session.config)
        return 0
    import viser
    from scripts.region_review_app import create_app
    server=viser.ViserServer(host='0.0.0.0',port=args.port)
    create_app(server,data,args.output,review_file,identity,initial,session,args.anchor)
    print(f'Camera review: http://0.0.0.0:{args.port} (CPU only); autosave: {review_file}',flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    raise SystemExit(main())
