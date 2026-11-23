"""Score identical held-out images with the published LPIPS AlexNet model."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiment import save_json


def score_folder(folder):
    import torch
    import lpips

    start=perf_counter()
    torch.set_num_threads(1)
    metric=lpips.LPIPS(net="alex",version="0.1",verbose=False).eval()
    folder=Path(folder)
    values={}
    for arm_dir in sorted(x for x in folder.iterdir() if x.is_dir()):
        arms=[]
        for file in sorted(arm_dir.glob("final_*.npz")):
            with np.load(file) as raw:
                pair=np.stack((raw["rgb"],raw["reference_rgb"]))
            tensor=torch.from_numpy(pair.astype(np.float32)).permute(0,3,1,2)*2-1
            with torch.no_grad():
                arms.append(float(metric(tensor[:1],tensor[1:]).item()))
        if not arms:
            continue
        name=arm_dir.name
        result_file=folder/f"{name}.json"
        result=json.loads(result_file.read_text())
        if len(arms)!=len(result["views"]):
            raise ValueError(f"Missing paired renders for {name}.")
        for view,score in zip(result["views"],arms):
            view["lpips"]=score
        save_json(result_file,result)
        values[name]=float(np.mean(arms))
    table=folder/"paired_results.csv"
    with table.open(newline="") as file:
        reader=csv.DictReader(file)
        rows=list(reader)
        columns=reader.fieldnames
    if "lpips_alex" not in columns:
        columns=columns+["lpips_alex"]
    for row in rows:
        row["lpips_alex"]=values[row["arm"]]
    with table.open("w",newline="") as file:
        writer=csv.DictWriter(file,columns)
        writer.writeheader()
        writer.writerows(rows)
    save_json(folder/"lpips_setup.json",{"architecture":"AlexNet","LPIPS_version":"0.1",
             "full_unmasked_RGB_resolution":True,"torch_version":torch.__version__,
             "wall_s":perf_counter()-start})
