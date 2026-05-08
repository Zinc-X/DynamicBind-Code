import subprocess
from glob import glob
from tqdm import tqdm
import os
import os.path as osp

cmd = "/ai/share/workspace/xushang/conda_env/gmxMMPBSA/bin.AVX2_256/gmx editconf -f nowat.tpr -o nowat.gro"

for tpr_file in tqdm(glob("/ai/share/workspace/sxia/smart_logic_data/DynamicBinding_*/????/*/nowat.tpr")):
    workdir = osp.dirname(tpr_file)
    subprocess.run(cmd, shell=True, cwd=workdir)
    # break
