import yaml
from glob import glob
import os
import os.path as osp

for trained_dir in glob("/ai/share/workspace/sxia/scripts/pignet_md_dev/results_archive/sphysnet_md_pretraining/*"):
    prev_cfg = osp.join(trained_dir, "configs.yml")
    with open(prev_cfg) as f:
        prev_cfg = yaml.safe_load(f)
    prev_cfg["restart_file"] = osp.join(trained_dir, "save_best.pt")
    prev_cfg["save_dir"] = prev_cfg["save_dir"] + "_ft"
    prev_cfg["use_md_data"] = False

    with open(f"/ai/share/workspace/sxia/scripts/pignet_md_dev/configs/ft_{osp.basename(trained_dir)}.yml", "w") as f:
        yaml.safe_dump(prev_cfg, f)
