import logging
import os
import shutil
import sys
import time
import yaml
from typing import Dict, Iterable, List, Tuple, Union
import random

import numpy as np
import torch
import torch.nn as nn

from scipy import stats
import argparse

import models
import utils_dynamic_bind as utils
from omegaconf import OmegaConf
from dataset import get_dataset_dataloader_train
from utils_dynamic_bind import DynamicBindConfig, get_model_state_dict

# 1. Set Random Seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2. Configure PyTorch for Deterministic Behavior
def configure_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 3. Worker Initialization Function for DataLoader
def worker_init_fn(worker_id):
    seed = 1 + worker_id
    np.random.seed(seed)
    random.seed(seed)

# Set the seed and configure deterministic behavior
set_seed(1)
configure_deterministic()

parser = argparse.ArgumentParser()
parser.add_argument("config")
args, unknown = parser.parse_known_args()
config_file = args.config
args: DynamicBindConfig = OmegaConf.structured(DynamicBindConfig) 
args: DynamicBindConfig = OmegaConf.merge(args, OmegaConf.load(config_file))
cli_cfg = OmegaConf.from_cli(unknown)
args = OmegaConf.merge(args, cli_cfg)

data_dir = f'./data/pdbbind2020_clean'
md_data_dir = "./data/dynamic_bind"
key_dir = f'./data/keys_casf'
if args.use_lp_pdbbind:
    key_dir = "./data/keys_lppdbbind"
affinity_file = f'./data/pdb_to_affinity_scoring.csv'

def run(model: nn.Module, data_iter: Iterable, train_mode: bool, use_md_data=True,
        use_md_rank_loss=False) -> Tuple[Union[float, Dict[str, List[float]]]]:
    model.train() if train_mode else model.eval()
    losses, losses_der1, losses_der2, losses_md = ([], [], [], [])

    save_pred, save_true = dict(), dict()
    while True:
        model.zero_grad()
        sample_score, sample_md = next(data_iter, (None, None))

        if sample_score is None:
            break
        
        """ Scoring"""
        sample_score = utils.dic_to_device(sample_score, device)
        keys, affinity = sample_score["key"], sample_score["affinity"]

        loss_all = 0.0
        cal_der_loss = False
        if args.loss_der1_ratio > 0.0 or args.loss_der2_ratio > 0.0:
            cal_der_loss = True

        pred, loss_der1, loss_der2, _ = model(sample_score, cal_der_loss=cal_der_loss)
        
        """Loss functions"""
        loss = loss_fn(pred.sum(-1), affinity)

        loss_all += loss
        loss_all += loss_der1.mean() * args.loss_der1_ratio
        loss_all += loss_der2.mean() * args.loss_der2_ratio

        del sample_score

        """ MD data augmentation """
        if use_md_data is True and sample_md is not None:
            sample_md = utils.dic_to_device(sample_md, device)
            keys_md, affinity_md = sample_md["key"], sample_md["affinity"]
            pred_md, _, _, _ = model(sample_md)
            
            if use_md_rank_loss is True:
                # loss_md = torch.clamp(affinity_md+1-pred_md.sum(-1), min=0.0).mean()
                loss_md = torch.clamp(pred_md.sum(-1)-affinity_md-1, min=0.0).mean()
            else:
                loss_md = loss_fn(pred_md.sum(-1), affinity_md)
            loss_all += loss_md * args.loss_md_ratio

            losses_md.append(loss_md.data.cpu().numpy())

            del sample_md

        if train_mode:
            loss_all.backward()
            optimizer.step()
            
        losses.append(loss.data.cpu().numpy())
        losses_der1.append(loss_der1.data.cpu().numpy())
        losses_der2.append(loss_der2.data.cpu().numpy())

        affinity = affinity.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        for i in range(len(keys)):
            save_pred[keys[i]] = pred[i]
            save_true[keys[i]] = affinity[i]

    losses = np.mean(np.array(losses))
    losses_der1 = np.mean(np.array(losses_der1))
    losses_der2 = np.mean(np.array(losses_der2))
    if use_md_data is True:
        losses_md = np.mean(np.array(losses_md))
    else:
        losses_md = 0.0

    return (losses, losses_der1, losses_der2, losses_md, 
            save_pred, save_true)


"""Make directory for save files"""
os.makedirs(args.save_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(args.save_dir, "training.log"),
                                format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

with open(f"{args.save_dir}/configs.yml", "w") as f:
    yaml_string = OmegaConf.to_yaml(args)
    f.write(yaml_string)

"""Save Reproducibility Settings"""
with open(os.path.join(args.save_dir, 'reproducibility_settings.txt'), 'w') as f:
    f.write(f"Seed: 1\n")
    f.write("Deterministic: True\n")
    f.write(f"CUDNN Deterministic: {torch.backends.cudnn.deterministic}\n")
    f.write(f"CUDNN Benchmark: {torch.backends.cudnn.benchmark}\n")

"""Read labels"""
if args.use_lp_pdbbind:
    train_keys, test_keys, id_to_y = utils.read_data_lp_pdbbind(affinity_file, key_dir, True)
else:
    train_keys, test_keys, id_to_y = utils.read_data(affinity_file, key_dir, use_generalset=args.use_generalset)

"""Model"""
model = models.DynamicBind(args)
device = torch.device(f'cuda:{args.ngpu}' if torch.cuda.is_available() else 'cpu')
model = utils.initialize_model(model, device, args.restart_file)
if args.freeze_non_rbf_layers:
    logging.info("Freezing non-RBF layers")
    for name, param in model.named_parameters():
        if not name.startswith("rbf_attention_layer."):
            param.requires_grad = False

# SWA training
if args.swa_weight is not None:
    model = torch.optim.swa_utils.AveragedModel(model,
            avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.swa_weight), use_buffers=True)

print('Initiation Done!!!')

"""Dataloader"""
train_dataset, train_dataloader = get_dataset_dataloader_train(
    args,
                        train_keys, data_dir, md_data_dir, id_to_y, 
                        args.batch_size, args.num_workers, 
                        use_md_data=args.use_md_data,
                        worker_init_fn=worker_init_fn,  # Pass the worker_init_fn,
)

test_dataset, test_dataloader = get_dataset_dataloader_train(
    args,
                        test_keys, data_dir, md_data_dir, id_to_y, 
                        args.batch_size, args.num_workers, 
                        use_md_data=args.use_md_data, train=False,
                        worker_init_fn=worker_init_fn,  # Pass the worker_init_fn

)

if args.use_md_data:
    logging.info(f"training data loaded with {len(train_dataset.pdb2_traj_data.keys())} MD traj data")
    logging.info(f"valid data loaded with {len(test_dataset.pdb2_traj_data.keys())} MD traj data")

"""Optimizer and loss"""
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.MSELoss()

"""Start training"""
start_time = time.time()

if args.restart_file:
    print(f'Resuming training: {args.restart_file}')

for epoch in range(args.num_epochs):
    (train_losses, train_losses_der1, train_losses_der2) = ([], [], [])

    (test_losses, test_losses_der1, test_losses_der2) = ([], [], [])

    (train_pred, train_true, test_pred, test_true) = (dict(), dict(), dict(), dict())

    "iterator"
    train_data_iter, test_data_iter = (iter(train_dataloader),
                                        iter(test_dataloader))
    
    "Forward and Backpropagation on training set"
    train_time_0 = time.time()
    (train_losses, train_losses_der1, train_losses_der2, train_losses_md, train_pred, train_true
    ) = run(model, train_data_iter, True, use_md_data=args.use_md_data, use_md_rank_loss=args.use_md_rank_loss)

    "Validation on Coreset"
    (test_losses, test_losses_der1, test_losses_der2, test_losses_md, test_pred, test_true
    ) = run(model, test_data_iter, False, use_md_data=args.use_md_data, use_md_rank_loss=args.use_md_rank_loss)

    train_time_1 = time.time()
    
    "Scoring power calculation"
    scores_predicted, scores_gt = [],[]
    for key in test_true.keys():
        scores_gt.append(test_true[key])
        scores_predicted.append(test_pred[key].sum())

    pearsonr=stats.pearsonr(scores_predicted, scores_gt)[0]
    spearmanr = stats.spearmanr(scores_predicted, scores_gt)[0]

    test_loss_all = test_losses

    hbond_coeff = get_model_state_dict(model)['hbond_coeff'].item()
    metal_coeff= get_model_state_dict(model)['metal_coeff'].item()
    hydrophobic_coeff = get_model_state_dict(model)['hydrophobic_coeff'].item()

    logger.info(f"Epoch {epoch} with {train_time_1-train_time_0:.1f} s")
    logger.info(f"Score {train_losses:.3f}, Test_Score {test_losses:.3f}, Pearson {pearsonr:.3f}, Rank {spearmanr:.3f}")
    logger.info(f"Score_MD {train_losses_md:.3f}, Test_Score_MD {test_losses_md:.3f}")
    logger.info(f"hbond_coeff {hbond_coeff:.3f}, "
        + f"metal_coeff {metal_coeff:.3f}, hydrophobic_coeff {hydrophobic_coeff:.3f}")
    logger.info(f"Der1 {train_losses_der1:.3f}, Der2 {train_losses_der2:.3f}, "
        + f"Test_Der1 {test_losses_der1:.3f}, Test_Der2 {test_losses_der2:.3f}")

    if epoch == 0:
        test_loss_track = test_loss_all
        best_epoch = epoch
    elif args.num_epochs - epoch < args.best_saving_epochs:
        if test_loss_track >= test_loss_all:
            test_loss_track = test_loss_all
            best_epoch = epoch
            torch.save(get_model_state_dict(model), os.path.join(args.save_dir, "save_best.pt"))
    torch.save(get_model_state_dict(model), os.path.join(args.save_dir, "save_last.pt"))

    lr = args.lr * (args.lr_decay ** (epoch // args.lr_decay_period))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

finish_time = time.time()
total_time = (finish_time - start_time)/3600
logger.info(f'Training of epoch {epoch} completed with best epoch saved {best_epoch}')
logger.info(f'Total time: {total_time:.2f} hours')
