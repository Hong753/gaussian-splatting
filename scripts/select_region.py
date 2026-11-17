#!/usr/bin/env python3
"""Select a LiDAR region, inspect overlapping RGB views, export original poses."""
import argparse
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
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--review-file', type=Path, help='Autosaved review; defaults to OUTPUT.review.json')
    p.add_argument('--export', action='store_true', help='Headless export of explicitly kept images from the saved review')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.baseline:
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
    if args.point_cloud is None or args.keyframes is None:
        raise ValueError('Supply --baseline or both --point-cloud and --keyframes')
    if args.output.exists():
        raise FileExistsError(f'Use a new region directory: {args.output}')
    data = RegionData(args.source, args.point_cloud, args.keyframes, args.sample_points)
    settings = dict(halo=args.halo,min_overlap=args.min_overlap,max_camera_distance=args.max_camera_distance,
                    frame_start=args.frame_start,frame_end=args.frame_end,exclude=args.exclude)
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
