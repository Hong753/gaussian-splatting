import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import orange4f_keyframe_trial as trial
from utils import read_write_model as colmap
from utils.checkpoint_snapshots import digest


class KeyframeTrialTests(unittest.TestCase):
    def test_full_split(self):
        names = [f"{i}.jpg" for i in range(0, 5009, 3)]
        result = trial.thin(names[::-1], 240)
        self.assertEqual((len(set(result)), result[0], result[-1]), (240, names[0], names[-1]))
        self.assertTrue(set(np.diff([names.index(n) for n in result])) <= {6, 7})

    def test_paired_staging_and_localization(self):
        # Independently synthesize projections and patches from depth-varying landmarks.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, baseline, fit = (root/n for n in ("capture", "base", "fit"))
            photos, full = capture/"images", capture/"sparse/0"
            for folder in (photos, full, fit):
                folder.mkdir(parents=True)
            names = [f"frame{i}.png" for i in range(7)]
            train = names[::3]
            rng = np.random.default_rng(23)
            grid = np.array([(x,y) for x in range(30,310,28) for y in range(30,228,25)])
            depth = rng.uniform(3,7,len(grid))
            xyz = np.c_[(grid-[159.5,119.5])*depth[:,None]/255, depth]
            patterns = rng.integers(30,255,(len(grid),13,13),dtype=np.uint8)
            cam = colmap.Camera(1,"PINHOLE",320,240,np.array([255.,255.,160.,120.]))
            fitted, original = {}, {}
            point_ids = np.arange(1,len(grid)+1)
            true_R = None
            for i, name in enumerate(names):
                im = np.full((240,320),12,dtype=np.uint8)
                r = np.array([0.,0.001*i,0.])
                t = np.array([-0.012*i,0.,0.])
                R = cv2.Rodrigues(r)[0]
                if i == 2:
                    true_R = R
                camera_xyz = xyz@R.T + t
                uv = [159.5,119.5]+255*camera_xyz[:,:2]/camera_xyz[:,2,None]
                for (x,y),pattern in zip(np.rint(uv).astype(int),patterns):
                    im[y-6:y+7,x-6:x+7] = pattern
                if i == 5:
                    im[:] = 0
                cv2.imwrite(str(photos/name),im)
                fitted[i+1] = colmap.Image(i+1, colmap.rotmat2qvec(R),t,1,name,uv+.5,point_ids)
                # Deliberately invalid source poses; localizer must ignore them.
                original[i+1] = fitted[i+1]._replace(tvec=np.array([15.*i,20.,35.]))
            ids = [1, 7]
            pts = {int(pid): colmap.Point3D(pid,x,np.array([50,80,10]),0.,np.array(ids),np.array([j,j]))
                   for j,(pid,x) in enumerate(zip(point_ids,xyz))}
            colmap.write_model({1:cam},original,pts,full,".bin")
            colmap.write_model({1:cam},{i:fitted[i] for i in ids},pts,fit,".bin")
            (baseline/"scene").mkdir(parents=True)
            (baseline/"scene/images").symlink_to(photos)
            (baseline/"scene/sparse").symlink_to(full.parent)
            (capture/"spirula-input.txt").write_text(str(photos))
            hashes = {f:digest(full/f) for f in trial.BINARIES}
            trial.atomic_json(baseline/"protocol.json",dict(train=train,test=[n for n in names if n not in train],
                source=str(capture),pose_provenance="wrong test fixture poses",seed_provenance=str(full),sfm_sha256=hashes))
            # Only the external native mapper is a fixture. Both staging paths and query PnP run normally.
            binary = root/"spirula"
            binary.write_text("""#!/bin/bash
case "$1 $2" in
'sfm --version') echo 'spirula sfm 2026.11.12 (667e353)' ;;
'sam devices') echo 'test' ;;
'sfm auto') mkdir -p "$5/sparse/0"; cp "$FIT_MAP"/*.bin "$5/sparse/0/"; exit 0 ;;
*) exit 1 ;;
esac
""")
            binary.chmod(0o755)
            argv = ["test","--baseline",str(baseline),"--views","2","--prepare-only"]
            with patch("sys.argv",argv), patch.dict(os.environ,SPIRULA_BIN=str(binary),FIT_MAP=str(fit)):
                trial.main()
                trial.main()
            group = root/"base_2kf_loma-b"
            fixed, remap = group/"fixed", group/"remap"
            ref = trial.read(fixed/"protocol.json")
            new = trial.read(remap/"protocol.json")
            self.assertEqual(ref["train"],[names[0],names[-1]])
            self.assertEqual(ref["train"],new["train"])
            self.assertEqual(new["unregistered"],[names[5]])
            self.assertEqual(ref["sfm_sha256"],hashes)
            self.assertEqual(new["sfm_sha256"]["points3D.bin"],digest(fit/"points3D.bin"))
            cams = {v.name:v for v in colmap.read_images_binary(remap/"scene/sparse/0/images.bin").values()}
            np.testing.assert_allclose(cams[names[0]].tvec,fitted[1].tvec)
            np.testing.assert_allclose(cams[names[2]].tvec,[-0.024,0.,0.],atol=.02)
            np.testing.assert_allclose(cams[names[2]].qvec2rotmat(),true_R,atol=.01)
            self.assertEqual(len(cams[names[2]].point3D_ids),0)
            for index, arm in enumerate((fixed,remap,baseline)):
                folder = arm/"evaluation/iteration_7000"
                rows = {name:{"psnr_db":10*index+i} for i,name in enumerate(names)
                        if name in trial.read(arm/"protocol.json")["test"]}
                trial.atomic_json(folder/"eval/model_0/frames.json",rows)
                trial.atomic_json(folder/"psnr.json",{})
            trial.score_coverage(remap,trial.read(baseline/"protocol.json"),7000)
            self.assertEqual(trial.read(remap/"evaluation/iteration_7000/psnr.json")["unregistered"], [names[5]])
            trial.compare(group,baseline,trial.read(baseline/"protocol.json"),7000)
            score = trial.read(group/"paired_psnr_7000.json")["groups"]["original_stride3_heldout"]
            self.assertEqual((score["scored"],score["expected"]),(3,4))
            self.assertAlmostEqual(score["mean_psnr_db"]["fixed"],7/3)
            self.assertAlmostEqual(score["mean_psnr_db"]["stride3_baseline"],20+7/3)


if __name__ == "__main__":
    unittest.main()
