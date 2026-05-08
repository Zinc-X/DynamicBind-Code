import os, glob
import os.path as osp
import csv
import numpy as np
import pandas as pd
import torch
import argparse

from omegaconf import OmegaConf
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Mol

from models import DynamicBind
from utils_dynamic_bind import initialize_model, dic_to_device, DynamicBindConfig, read_data_lp_pdbbind
from dataset import get_dataset_dataloader, get_dataset_dataloader_train, load 
from utils.scores.casf_scores import plot_scatter_info
import utils_dynamic_bind

class DataLoaderFactory:
    def __init__(self, cfg_valid: DynamicBindConfig, model_cfg: DynamicBindConfig) -> None:
        self.cfg_valid = cfg_valid
        self.model_cfg = model_cfg

        self.cfg_valid.casf2016_val = not model_cfg.use_lp_pdbbind
        self.cfg_valid.lp_pdbbind_test = model_cfg.use_lp_pdbbind

    def get_screening_list(self):
        # you can only specify one test set
        assert not (self.cfg_valid.lp_pdbbind_test and self.cfg_valid.casf2016_val), "you can only specify one test set"

        if self.cfg_valid.lp_pdbbind_test:
            return self.get_lp_pdbbind_test_screening_list()
        elif self.cfg_valid.casf2016_val:
            return self.get_casf_screening_list()
        else:
            return self.get_single_reference()
        
    def get_lp_pdbbind_test_screening_list(self):
        key_dir = "./data/keys_lppdbbind"
        data_dir = f'./data/pdbbind2020_clean'
        affinity_file = f'./data/pdb_to_affinity_scoring.csv'
        md_data_dir = "./data/dynamic_bind"
        test_keys, id_to_y = read_data_lp_pdbbind(affinity_file, key_dir, False)
        __, test_dataloader = get_dataset_dataloader_train(self.cfg_valid,
                        test_keys, data_dir, md_data_dir, id_to_y, 
                        self.model_cfg.batch_size, 0, 
                        use_md_data=self.model_cfg.use_md_data, train=False,
        )
        return test_dataloader, None, "lp_pdbbind"
        
    def get_single_reference(self):
        cfg_valid = self.cfg_valid
        assert cfg_valid.reference_file is not None and  cfg_valid.target_file is not None
        ref_mol, tar_mol, ref_name, tar_name = load(cfg_valid.reference_file, cfg_valid.target_file)
        if not isinstance(ref_mol, Mol) or not isinstance(tar_mol, Mol):
            print('rdkit error: reference or target file is failed to sanitize!')
            exit()
        screening_list = [(ref_mol, tar_mol, ref_name, tar_name)]
        __, test_dataloader = get_dataset_dataloader(
                    screening_list, {}, cfg_valid.batch_size, self.model_cfg.num_workers, False,
                    )
        return test_dataloader, None, os.path.basename(cfg_valid.reference_file) + os.path.basename(cfg_valid.target_file)

    def get_casf_screening_list(self):
        data_dir = f'./data/pdbbind2020_clean'
        md_data_dir = "./data/dynamic_bind"
        key_dir = f'./data/keys_casf'
        affinity_file = f'./data/pdb_to_affinity_scoring.csv'

        train_keys, test_keys, id_to_y = utils_dynamic_bind.read_data(affinity_file, key_dir, use_generalset=self.model_cfg.use_generalset)
        test_dataset, test_dataloader = get_dataset_dataloader_train(
            self.model_cfg,
                                test_keys, data_dir, md_data_dir, id_to_y, 
                                self.model_cfg.batch_size, self.model_cfg.num_workers, 
                                use_md_data=self.model_cfg.use_md_data, train=False

        )
        return test_dataloader, None, "casf-2016"

        cfg_valid = self.cfg_valid

        screening_list = []
        screening_md_list = []

        coreset_lig_dir = os.path.join(cfg_valid.casf2016_coreset_path, 'coreset_rdkit')
        coreset_pocket_dir = os.path.join(cfg_valid.casf2016_coreset_path, 'coreset_pocket5')
        coreset_lig_path_list = glob.glob(coreset_lig_dir+'/*.sdf')

        for lig_path in tqdm(coreset_lig_path_list):
            pdb_id = os.path.basename(lig_path).split('_')[0]
            pocket_path = os.path.join(coreset_pocket_dir, f'{pdb_id}_pocket5.pdb')
            ref_mol, tar_mol, ref_name, tar_name = load(lig_path, pocket_path)

            if isinstance(ref_mol, Mol) and isinstance(tar_mol, Mol):
                screening_list.append((ref_mol, tar_mol, pdb_id, ''))
            else:
                print(f'rdkit sanitazation error: Crystal PDB {pdb_id}')

            md_data_pdb_dir = f'{cfg_valid.md_data_path}/{pdb_id}'
            if os.path.exists(md_data_pdb_dir):
                for i in range(100):
                    md_lig_path = f'{md_data_pdb_dir}/frame_{i}/ligand_frame_{i}.sdf'
                    md_pocket_path = f'{md_data_pdb_dir}/frame_{i}/protein_frame_{i}_ligand_frame_{i}_pocket5.pdb'
                    ref_mol_md, tar_mol_md, ref_name, tar_name = load(md_lig_path, md_pocket_path)
                    if isinstance(ref_mol_md, Mol) and isinstance(tar_mol_md, Mol):
                        screening_md_list.append((ref_mol_md, tar_mol_md, pdb_id, str(i)))
                    else:
                        print(f'rdkit sanitazation error: Misato-MD PDB {pdb_id} Frame {i}')
            else:
                print(f'rdkit sanitazation error: Misato-MD PDB {pdb_id} does not exist')

        __, test_dataloader = get_dataset_dataloader(
                    screening_list, {}, cfg_valid.batch_size, self.model_cfg.num_workers, False,
                    )
        test_md_dataloader = None
        if len(screening_md_list) > 0:
             __, test_md_dataloader = get_dataset_dataloader(
                    screening_md_list, {}, cfg_valid.batch_size, self.model_cfg.num_workers, False,
                    )
        return test_dataloader, test_md_dataloader, "casf-2016"


def inference(args):
    if args.train_dir is not None:
        config_path = osp.join(args.train_dir, "configs.yml")
    else:
        config_path = args.config_file

    cfg_valid: DynamicBindConfig = OmegaConf.structured(DynamicBindConfig)
    cfg_valid = OmegaConf.merge(cfg_valid, OmegaConf.load(config_path))

    if args.train_dir is not None:
        cfg_valid.model_weight_dir = args.train_dir
        cfg_valid.batch_size = 1

    model_conf_path = os.path.join(cfg_valid.model_weight_dir, 'configs.yml')
    model_weight = os.path.join(cfg_valid.model_weight_dir , 'save_best.pt')
    if not os.path.exists(model_weight):
        model_weight = os.path.join(cfg_valid.model_weight_dir , 'save_last.pt')
    if cfg_valid.output_dir == "AUTO":
        cfg_valid.output_dir = os.path.join(cfg_valid.model_weight_dir, "test_scores")
    os.makedirs(cfg_valid.output_dir, exist_ok=True)
    model_cfg: DynamicBindConfig = OmegaConf.structured(DynamicBindConfig)
    model_cfg = OmegaConf.merge(model_cfg, OmegaConf.load(model_conf_path))

    dl_factory = DataLoaderFactory(cfg_valid, model_cfg)
    test_data_loader, test_md_data_loader, save_name = dl_factory.get_screening_list()

    device = torch.device(f'cuda:{cfg_valid.ngpu}' if torch.cuda.is_available() else 'cpu')
    model = DynamicBind(model_cfg)
    model = initialize_model(model, device, model_weight, strict=True)

    test_pred = dict()
    nb = len(test_data_loader)
    pbar = enumerate(test_data_loader)
    pbar = tqdm(pbar, total = nb)

    model.eval()
    for i_batch, sample in pbar:
        model.zero_grad()
        if sample[0] is None:
            continue
        sample = dic_to_device(sample[0], device)
        keys = sample["key"]
        affinity = sample["affinity"]

        with torch.no_grad():
            pred = model(sample)[0]
            pred = pred.data.cpu().numpy()
        affinity = affinity.data.cpu().numpy()

        for idx in range(len(keys)):
            key = keys[idx]
            test_pred[key] = pred[idx]
        
    """Write prediction results in CSV"""
    header = ['#code', 'score', 'vdw', 'hbond', 'metal', 'hydrophobic'] 
    header = ['#code', 'score']
    body = []
    for key in sorted(test_pred.keys()):
        # body.append([ key, test_pred[key].sum(),
        #               test_pred[key][0], test_pred[key][1],
        #               test_pred[key][2], test_pred[key][3]])
        body.append([ key, test_pred[key].sum()])

    model_version_name = os.path.basename(cfg_valid.model_weight_dir)
    with open(os.path.join(cfg_valid.output_dir , f'scores_{save_name}.csv'),
                                     'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)
    
    if test_md_data_loader is not None:
        test_pred = dict()
        nb = len(test_data_loader)
        pbar = enumerate(test_data_loader)
        pbar = tqdm(pbar, total = nb)

        model.eval()
        for i_batch, sample in pbar:
            model.zero_grad()
            if sample[0] is None:
                continue
            sample = dic_to_device(sample[0], device)
            keys = sample["key"]
            affinity = sample["affinity"]

            with torch.no_grad():
                pred = model(sample)[0]
                pred = pred.data.cpu().numpy()
            affinity = affinity.data.cpu().numpy()

            for idx in range(len(keys)):
                key = keys[idx]
                test_pred[key] = pred[idx]
            
        """Write prediction results in CSV"""
        header = ['#code', 'score', 'vdw', 'hbond', 'metal', 'hydrophobic'] 
        body = []
        for key in sorted(test_pred.keys()):
            body.append([ key, test_pred[key].sum(),
                        test_pred[key][0], test_pred[key][1],
                        test_pred[key][2], test_pred[key][3]])
        with open(os.path.join(cfg_valid.output_dir , f'md_scores_{save_name}.csv'),
                                     'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(body)
    
    # plotting figures
    if cfg_valid.casf2016_val or cfg_valid.lp_pdbbind_test:
        pred_df = pd.read_csv(os.path.join(cfg_valid.output_dir , f'scores_{save_name}.csv'))
        affinity_file = f'{model_cfg.data_path}/pdb_to_affinity_scoring.csv'
        with open(affinity_file) as f:
            lines = f.readlines()
            lines = [l.split(',') for l in lines]
            id_to_y = {l[0]: float(l[1].split('\n')[0]) for l in lines}
        exp_affinity = []
        for code in pred_df["#code"]:
            exp_affinity.append(id_to_y[code])
        exp_affinity = np.asarray(exp_affinity)
        pred_affinity = pred_df["score"].values / -1.36
        plot_scatter_info(exp_affinity, pred_affinity, cfg_valid.output_dir, f"{save_name}.png", "Model Performance on Test Set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="trained directory")
    # deprecated
    parser.add_argument('-c', '--config_file', type=str, 
                        default=None, help='config_file')
    
    args = parser.parse_args()
    inference(args)