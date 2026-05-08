import math
import time
from argparse import Namespace
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from Networks.SharedLayers.RBFAttentionLayer import TransformerLayer
from layers import InteractionNet, GatedGAT
from Networks.PhysLayers.Interaction_module import InteractionModule
from Networks.PhysLayers.PhysModule import PhysModule
from utils.utils_functions import gaussian_rbf, softplus_inverse
from utils_dynamic_bind import DynamicBindConfig

class DynamicBind(nn.Module):
    def __init__(self, args: DynamicBindConfig):
        super().__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GatedGAT(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.interaction_net:
            self.interaction_net = nn.ModuleList(
                [InteractionNet(args.dim_gnn) for _ in range(args.n_gnn)]
            )

        if args.sphysnet_interaction_net:
            self.sphysnet_interaction_net = nn.ModuleList(
                [InteractionModule(args.dim_gnn, 64, 1, "ssp", False, False, False, {}, "vi", "P") for _ in range(args.n_gnn)]
            )
        if args.sphysnet_module:
            self.sphysnet_modules = nn.ModuleList(
                [PhysModule(args.dim_gnn, 64, 1, 1, 1, 1, "ssp", "", 1, False, False, False, True, {"preserve_prot_embed": False}, "P") for _ in range(args.n_gnn)]
            )
        
        if args.whole_traj_learning:
            self.rbf_attention_layer = TransformerLayer(64, args.rbf_num_heads, 0.1, args.rbf_pos_encoding_max_len)
        if args.rbf_sigmoid:
            self.rbf_sigmoid = nn.Sigmoid()
        self.register_rbf()

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.cal_distance_correction = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Tanh(),
        )

        self.hbond_enhancement = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )      

        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.metal_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def register_rbf(self):
        if not self.args.sphysnet_interaction_net and not self.args.sphysnet_module:
            return
        
        n_rbf = 64
        feature_dist = self.args.sphysnet_cutoff
        feature_dist = torch.as_tensor(feature_dist)
        self.register_parameter('rbf_cutoff', torch.nn.Parameter(feature_dist, False))
        expansion_coe = torch.as_tensor([1.])
        self.register_parameter('rbf_expansion_coe', torch.nn.Parameter(expansion_coe, False))
        # Centers are params for Gaussian RBF expansion in PhysNet
        dens_min = 0.
        centers = softplus_inverse(torch.linspace(math.exp(-dens_min), math.exp(-feature_dist * expansion_coe), n_rbf))
        centers = torch.nn.functional.softplus(centers)
        self.register_parameter('rbf_centers', torch.nn.Parameter(centers, False))

        # Widths are params for Gaussian RBF expansion in PhysNet
        widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / n_rbf)) ** 2)] * n_rbf
        widths = torch.as_tensor(widths)
        widths = torch.nn.functional.softplus(widths)
        self.register_parameter('rbf_widths', torch.nn.Parameter(widths, False))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        vdws_sum: Tensor,
        A: Tensor,
    ) -> Tensor:

        dm = dm - vdws_sum
        retval = dm / -0.7
        retval = retval.clamp(min=0.0, max=1.0)

        hbond_coeff = self.hbond_coeff * self.hbond_coeff
        if self.args.hbond_enhancement:
            H = self.hbond_enhancement(h).squeeze(-1)*(self.args.hbond_enhancement_max_ratio
                -self.args.hbond_enhancement_min_ratio)+self.args.hbond_enhancement_min_ratio
            hbond_coeff = hbond_coeff * H

        retval = -hbond_coeff * retval * A
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval
    
    def cal_metal(
                  self,
                  dm: Tensor,
                  h: Tensor,
                  vdws_sum: Tensor,
                  A: Tensor,
                  ) -> Tensor:

        dm_0 = vdws_sum
        dm = dm - dm_0

        retval = dm / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = A * retval * -(self.metal_coeff * self.metal_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)

        return retval

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        vdws_sum: Tensor,
        A: Tensor,
    ) -> Tensor:

        dm = dm - vdws_sum
        retval = (-dm + 1.5)
        retval = retval.clamp(min=0.0, max=1.0)

        retval = A * retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        vdws_sum: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(1, 1, target_valid.size(1))
        target_valid_ = target_valid.unsqueeze(1).repeat(1, ligand_valid.size(1), 1)

        dm_0 = vdws_sum
        dm_0[dm_0 < 0.0001] = 1


        vdw_term1 = torch.pow(dm_0 / dm, 2 * self.args.vdw_N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, self.args.vdw_N)
        energy = vdw_term1 + vdw_term2

        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.args.max_vdw_interaction - self.args.min_vdw_interaction)
        A = A + self.args.min_vdw_interaction
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm
    
    def gaussian(self, x, mu, sig):
        return 1./(torch.sqrt(2.*torch.tensor(math.pi))*sig)*torch.exp(-torch.pow((x - mu)/sig, 2.)/2)
    
    def gaussian_vina(self, x, mu, sig):
        return torch.exp(-torch.pow((x - mu)/sig, 2.))

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5, cal_der_loss: bool = False,
        dm_correction: bool = True) -> Tuple[Tensor]:
        ligand_h = sample["ligand_h"]
        ligand_adj = sample["ligand_adj"]
        target_h = sample["target_h"]
        target_adj = sample["target_adj"]
        interaction_indice = sample["interaction_indice"]
        ligand_pos = sample["ligand_pos"]
        target_pos = sample["target_pos"]
        rotor = sample["rotor"]
        ligand_vdw_radii = sample["ligand_vdw_radii"]
        target_vdw_radii = sample["target_vdw_radii"]
        ligand_valid = sample["ligand_valid"]
        target_valid = sample["target_valid"]

        # feature embedding
        ligand_h = self.node_embedding(ligand_h)
        target_h = self.node_embedding(target_h)

        # distance matrix
        ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(ligand_pos, target_pos, DM_min)

        # GatedGAT propagation
        for idx in range(len(self.gconv)):
            ligand_h = self.gconv[idx](ligand_h, ligand_adj)
            target_h = self.gconv[idx](target_h, target_adj)
            ligand_h = F.dropout(
                ligand_h, training=self.training, p=self.args.dropout_rate
            )
            target_h = F.dropout(
                target_h, training=self.training, p=self.args.dropout_rate
            )

        if self.args.sphysnet_interaction_net or self.args.sphysnet_module:
            pl_feature_concat = torch.concat([ligand_h, target_h], dim=1)
            n_atoms_per_batch = pl_feature_concat.shape[1]
            n_ligand_atoms = ligand_h.shape[1]
            n_prot_atoms = target_h.shape[1]
            pl_feature_flat = pl_feature_concat.view(-1, pl_feature_concat.shape[-1])

            with torch.no_grad():
                edge_index = (dm < self.args.sphysnet_cutoff).nonzero()
                pair_dist = dm[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
                
                batch_edge_index_incre = edge_index[:, [0]] * n_atoms_per_batch
                ligand_index_incre = n_ligand_atoms
                edge_index = batch_edge_index_incre + edge_index[:, [1, 2]]
                edge_index[:, 1] = edge_index[:, 1] + ligand_index_incre

                # [2, num_edges]
                edge_index = edge_index.T
                # two ways
                edge_index = torch.concat([edge_index, edge_index[[1,0], :]], dim=-1)
                pair_dist = torch.concat([pair_dist, pair_dist], dim=0)
                if self.args.whole_traj_learning:
                    pl_pos_traj_combined = torch.concat([sample["lig_traj_coords"], sample["target_traj_coords"]], dim=-2)
                    pl_pos_traj_combined = pl_pos_traj_combined.transpose(0, 1)
                    # [n_frames, n_all_atoms, 3]
                    pl_pos_traj_combined = pl_pos_traj_combined.reshape(pl_pos_traj_combined.shape[0], -1, pl_pos_traj_combined.shape[-1])
                    traj_pair_dist = torch.sqrt(((pl_pos_traj_combined[:, edge_index[0], :]-pl_pos_traj_combined[:, edge_index[1], :])**2).sum(dim=-1))
                    pair_dist = torch.concat([pair_dist.view(1, -1), traj_pair_dist], dim=0)

            expansions = gaussian_rbf(pair_dist, getattr(self, 'rbf_centers'),
                                                        getattr(self, 'rbf_widths'),
                                                        getattr(self, 'rbf_cutoff'),
                                                        getattr(self, 'rbf_expansion_coe'),
                                                        return_dict=True, linear=False)
            if self.args.whole_traj_learning:
                # [n_frames, n_pairs, 64]
                rbf = expansions["rbf"].view(pair_dist.shape[0], -1, expansions["rbf"].shape[-1])
                # [n_pairs, n_frames, 64]
                rbf = rbf.transpose(0, 1)
                rbf_atten, __ = self.rbf_attention_layer(rbf)

                fin_rbf = rbf_atten[:, 0, :]
                if self.args.rbf_residual_weight is not None:
                    # adding a residual connection
                    fin_rbf = self.args.rbf_residual_weight * rbf_atten[:, 0, :] + rbf[:, 0, :]
                expansions["rbf"] = fin_rbf
            if self.args.rbf_sigmoid:
                assert self.args.whole_traj_learning
                expansions["rbf"] = self.rbf_sigmoid(expansions["rbf"])
            runtime_vars = {"vi": pl_feature_flat, "edge_index": edge_index, "edge_attr": expansions}
        
        if self.args.sphysnet_interaction_net:
            for layer in self.sphysnet_interaction_net:
                interacted_x, _ = layer(runtime_vars)
                runtime_vars["vi"] = interacted_x
            interacted_x = interacted_x.view(-1, n_atoms_per_batch, interacted_x.shape[-1])
            ligand_h, target_h = interacted_x[:, :n_ligand_atoms, :], interacted_x[:, n_ligand_atoms:, :]
        if self.args.sphysnet_module:
            for layer in self.sphysnet_modules:
                runtime_vars = layer(runtime_vars)
            interacted_x = runtime_vars["vi"].view(-1, n_atoms_per_batch, runtime_vars["vi"].shape[-1])
            ligand_h, target_h = interacted_x[:, :n_ligand_atoms, :], interacted_x[:, n_ligand_atoms:, :]
            if self.args.use_sphysnet_out:
                assert not cal_der_loss
                # [n_all, 1]
                out = runtime_vars["out"].reshape(-1, n_atoms_per_batch, runtime_vars["out"].shape[-1])
                with torch.no_grad():
                    pl_valid = torch.concat([ligand_valid, target_valid], dim=-1)
                out = torch.where(pl_valid.unsqueeze(-1) > 0.9, out, torch.zeros_like(out))
                out = torch.sum(out, dim=1)
                der1 = torch.zeros_like(out).sum()
                der2 = torch.zeros_like(out).sum()
                return out, der1, der2, out

        # InteractionNet propagation
        if self.args.interaction_net:
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for idx in range(len(self.interaction_net)):
                new_ligand_h = self.interaction_net[idx](
                    ligand_h,
                    target_h,
                    adj12,
                )
                new_target_h = self.interaction_net[idx](
                    target_h,
                    ligand_h,
                    adj12.permute(0, 2, 1),
                )
                ligand_h, target_h = new_ligand_h, new_target_h
                ligand_h = F.dropout(
                    ligand_h, training=self.training, p=self.args.dropout_rate
                )
                target_h = F.dropout(
                    target_h, training=self.training, p=self.args.dropout_rate
                )

        # concat features
        h1_ = ligand_h.unsqueeze(2).repeat(1, 1, target_h.size(1), 1)
        h2_ = target_h.unsqueeze(1).repeat(1, ligand_h.size(1), 1, 1)
        h_cat = torch.cat([h1_, h2_], -1)

        "Sum of vdw radii"
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1))
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1)
        vdws_sum = ligand_vdw_radii_ + target_vdw_radii_


        "distance correction"
        if self.args.dm_correction and dm_correction:
            B = self.cal_distance_correction(h_cat).squeeze(-1) * self.args.dev_vdw_radius
            dm = dm + B

        "compute energy component"
        energies_component = []
        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
                                            dm,
                                            h_cat,
                                            vdws_sum,
                                            ligand_valid,
                                            target_valid,
                                            )
        energies_component.append(vdw_energy)

        # hbond interaction
        hbond = self.cal_hbond(
                                dm,
                                h_cat,
                                vdws_sum,
                                interaction_indice[:, 0]
                               )
        energies_component.append(hbond)

        # metal interaction
        metal = self.cal_metal(
                                dm,
                                h_cat,
                                vdws_sum,
                                interaction_indice[:, 1],
                                )
        energies_component.append(metal)

        # hydrophobic interaction
        hydrophobic = self.cal_hydrophobic(
                                        dm,
                                        h_cat,
                                        vdws_sum,
                                        interaction_indice[:, 2]
                                        )
        energies_component.append(hydrophobic)


        energies_component = torch.cat(energies_component, -1)
        # rotor penalty
        if not self.args.no_rotor_penalty:
            energies = energies_component / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        # derivatives
        if cal_der_loss:
            gradient = torch.autograd.grad(
                energies.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]
            der1 = torch.pow(gradient.sum(1), 2).mean()
            der2 = torch.autograd.grad(
                gradient.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]

            der2 = (- der2.sum(1)).clamp(min=self.args.min_loss_der2).mean()
        else:
            der1 = torch.zeros_like(energies).sum()
            der2 = torch.zeros_like(energies).sum()

        return energies, der1, der2, energies_component


