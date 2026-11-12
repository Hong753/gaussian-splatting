"""Join identical GT/pose/checkpoint previews into culling-off/on columns."""
import json
import os
from pathlib import Path

from scripts.view_comparisons import write_gallery


def combine(off, on, output, model, iteration):
    output=Path(output)
    links=[]
    for split in ('train','eval'):
        roots=[Path(off)/split/'model_0',Path(on)/split/'model_0']
        if not (roots[0]/'frames.json').is_file():
            continue
        records=[json.loads((root/'frames.json').read_text()) for root in roots]
        if set(records[0]) != set(records[1]):
            raise ValueError('Culling comparison views differ')
        destination=output/split
        rows=[]
        for name in sorted(records[0]):
            if records[0][name]['target_sha256'] != records[1][name]['target_sha256']:
                raise ValueError('Culling comparison ground truth differs')
            entries=[]
            for root,record in zip(roots,records):
                entry={key:os.path.relpath(root/record[name][key],destination) for key in ('image','error','comparison')}
                entry['psnr_db']=record[name]['psnr_db'];entries.append(entry)
            rows.append(dict(name=name,gt=os.path.relpath(roots[0]/records[0][name]['gt'],destination),renders=entries))
        models=[dict(label='Culling '+mode,path=str(model),iteration=iteration,
                     mean_psnr_db=sum(float(row['renders'][i]['psnr_db']) for row in rows)/len(rows))
                for i,mode in enumerate(('OFF','ON'))]
        write_gallery(destination,f'{split.title()}: same checkpoint and poses, culling off/on',rows,models,
                      dict(split=split,rendered_views=len(rows),camera_poses='refined',iteration=iteration))
        links.append(f'<li><a href="{split}/index.html">{split.title()} views: GT | culling OFF | culling ON</a></li>')
    output.mkdir(parents=True,exist_ok=True)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Culling comparison</h1><ul>'+''.join(links)+'</ul>')
    return output/'index.html'
