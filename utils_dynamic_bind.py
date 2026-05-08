from dataclasses import dataclass
import pickle
import subprocess
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from rdkit import Chem
from omegaconf import MISSING


def dic_to_device(dic: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
        elif isinstance(dic_value, np.ndarray):
            dic_value = torch.tensor(np.expand_dims(dic_value, axis=0)*1, dtype=torch.float32).to(device)
            dic[dic_key] = dic_value
        elif dic_key=='rotor':
            dic_value = torch.tensor(np.expand_dims(dic_value, axis=0)*1, dtype=torch.float32).to(device)
            dic[dic_key] = dic_value

    return dic


def set_cuda_visible_device(num_gpus: int, max_num_gpus: int = 16) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding="utf-8")
            out = proc.communicate()

            if "No devices were found" in out[0]:
                break

            if "No running" in out[0]:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


def initialize_model(
    model: nn.Module, device: torch.device, load_save_file: bool = False, strict: bool = False
) -> nn.Module:
    if load_save_file:
        if device.type == "cpu":
            model.load_state_dict(torch.load(load_save_file, map_location="cpu"), strict=strict)
        else:
            model.load_state_dict(torch.load(load_save_file, map_location=device), strict=strict)
            # model.load_state_dict(torch.load(load_save_file))
    else:
        for name, param in model.named_parameters():
            if name.startswith(("sphysnet_interaction_net", "rbf")):
                continue
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)
    
    model.to(device)
    return model


def read_data(
    affinity_file: str, key_dir: str, train: bool = True, use_generalset=False
) -> Tuple[Union[List[str], Dict[str, float]]]:
    with open(affinity_file) as f:
        lines = f.readlines()
        lines = [l.split(',') for l in lines]
        id_to_y = {l[0]: float(l[1].split('\n')[0]) for l in lines}

    with open(f"{key_dir}/keys_coreset.txt") as f:
        lines = f.readlines()
        test_keys = [l.split('\n')[0] for l in lines]
    
    if use_generalset:
        train_key_path = f"{key_dir}/keys_trainset.txt"
    else:
        train_key_path = f"{key_dir}/keys_trainset_refineset.txt" 

    if train:
        with open(train_key_path) as f:
            lines = f.readlines()
            train_keys = [l.split('\n')[0] for l in lines]

        return train_keys, test_keys, id_to_y
    else:
        return test_keys, id_to_y

def read_data_lp_pdbbind(affinity_file: str, key_dir: str, train: bool):
    with open(affinity_file) as f:
        lines = f.readlines()
        lines = [l.split(',') for l in lines]
        id_to_y = {l[0]: float(l[1].split('\n')[0]) for l in lines}
    
    if train:
        test_name = "val_set_valid.txt"
    else:
        test_name = "test_set_valid.txt"

    with open(f"{key_dir}/{test_name}") as f:
        lines = f.readlines()
        test_keys = [l.split('\n')[0] for l in lines]

    if train:
        with open(f"{key_dir}/train_set_valid.txt") as f:
            lines = f.readlines()
            train_keys = [l.split('\n')[0] for l in lines]
        return train_keys, test_keys, id_to_y
    else:
        return test_keys, id_to_y

def write_result(
    filename: str, pred: Dict[str, List[float]], true: Dict[str, List[float]]
) -> None:
    with open(filename, "w") as w:
        for k in pred.keys():
            w.write(f"{k}\t{true[k]:.3f}\t")
            w.write(f"{pred[k].sum():.3f}\t")
            w.write(f"{0.0}\t")
            for j in range(pred[k].shape[0]):
                w.write(f"{pred[k][j]:.3f}\t")
            w.write("\n")
    return

def save_ligand_sdf(mol, path):
    with Chem.SDWriter(path) as w:
        w.write(mol)

def save_pdb_file(protein, path):
    pdbblock = Chem.MolToPDBBlock(protein)
    with open(path,'w') as newfile:
        newfile.write(pdbblock)

def get_model_state_dict(model: nn.Module):
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        return model.module.state_dict()
    return model.state_dict()

@dataclass
class DynamicBindConfig:
    # Model Parameters
    dim_gnn: int = 128
    n_gnn: int = 3
    interaction_net: bool = False
    sphysnet_interaction_net: bool = False
    sphysnet_module: bool = False
    use_sphysnet_out: bool = False
    sphysnet_cutoff: float = 10.
    no_rotor_penalty: bool = False
    dropout_rate: float = 0.1
    vdw_N: float = 6.0
    max_vdw_interaction: float = 0.0356
    min_vdw_interaction: float = 0.0178
    lr: float = 0.0001
    lr_decay: float = 1.0
    lr_decay_period: float = 200
    weight_decay: float = 0.0
    scaling: float =1.0
    lattice_dim: float = 24
    grid_rotation: bool = False
    swa_weight: Optional[float] = None

    # Training Paramters
    ngpu: int = 0
    restart_file: Optional[str] = None
    save_dir: str = MISSING
    data_path: str = "./data"
    md_data_path: str = "./data"
    md_first_n: Optional[int] = None
    # training data
    use_lp_pdbbind: bool = True
    use_generalset: bool = True

    batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 300
    best_saving_epochs: int = 50

    loss_der1_ratio: float = 5.0 # 10
    loss_der2_ratio: float = 5.0 # 10
    min_loss_der2: float = 0

    dm_correction: bool = True
    dev_vdw_radius: float = 0.2 # 0.2

    hbond_enhancement: bool = True
    hbond_enhancement_max_ratio: float = 4.0
    hbond_enhancement_min_ratio: float = 0.5

    # MD data related
    use_md_data: bool = False
    use_md_rank_loss: bool = False
    loss_md_ratio: float = 0.25
    static_sample_weight: float = 0.
    whole_traj_learning: bool = False
    rbf_sigmoid: bool = False
    freeze_non_rbf_layers: bool = False
    rbf_residual_weight: Optional[float] = None
    rbf_pos_encoding_max_len: int = 500
    rbf_num_heads: int = 8
    n_total_frames: int = 10
    md_use_first_n_frames: Optional[int] = None

    # inference
    model_weight_dir: str = MISSING
    output_dir: str = "AUTO"
    casf2016_val: bool = False
    lp_pdbbind_test: bool = True
    casf2016_coreset_path: str = "/ai/share/workspace/jhzhou/data_LT/CASF-2016"
    target_file: Optional[str] = None
    reference_file: Optional[str] = None
