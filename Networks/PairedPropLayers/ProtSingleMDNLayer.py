from collections import defaultdict
from Networks.PairedPropLayers.MDNLayer import get_prot_dim
from Networks.PairedPropLayers.NotEnoughMDNLayers import GeneralMDNLayer


import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
import torch.nn.functional as F
import torch.nn as nn


from typing import Optional, Tuple, Union

from utils.data.data_utils import parse_hetero_edge

BOND2IDX = {"min_dist": 0, "max_dist": 1, "c_alpha_dist": 2, "c_beta_dist": 3}
IDX2BOND = {v: k for k, v in BOND2IDX.items()}

class ProtSingleMDNLayer(GeneralMDNLayer):
    # Add protein intra-residue interaction.
    def __init__(self, ltype="non-local", **kwargs) -> None:
        super().__init__("prot_embed", "prot_embed", **kwargs)
        # exlcude close interactions. For example, protprot_exclude_edge==1 means exlcude 1-2 interaction
        # protprot_exclude_edge==2 means exclude 1-2 and 1-3 interaction.
        self.seq_separation_cutoff: Optional[int] = kwargs["cfg"].model.mdn["protprot_exclude_edge"]
        self.ltype = ltype
        self.ignore_pair_batch = True
        self.current_edge_name = None

        param_dim_modify = 0
        self.add_seq_distance = self.cfg.model.mdn.add_seq_distance
        if self.add_seq_distance: param_dim_modify = 1
        self.seq_distance_dim = self.cfg.model.mdn.seq_distance_dim
        self.seq_distance_cap = self.cfg.model.mdn.seq_distance_cap
        if self.add_seq_distance and self.seq_distance_dim is not None:
            self.seq_distance_embedding = nn.Embedding(self.seq_distance_cap, self.seq_distance_dim)
            param_dim_modify = self.seq_distance_dim

        if "+" in self.mdn_edge_name:
            for bond_type in self.mdn_edge_name.split("+"):
                bond_type_short = parse_hetero_edge(bond_type)[1]
                setattr(self, f"z_pi_{bond_type_short}", nn.Linear(self.hidden_dim + param_dim_modify, self.n_gaussians))
                setattr(self, f"z_sigma_{bond_type_short}", nn.Linear(self.hidden_dim + param_dim_modify, self.n_gaussians))
                setattr(self, f"z_mu_{bond_type_short}", nn.Linear(self.hidden_dim + param_dim_modify, self.n_gaussians))


    def forward(self, runtime_vars: dict):
        nmdn_params = defaultdict(list)
        for bond_type in self.mdn_edge_name.split("+"):
            self.current_edge_name = bond_type
            runtime_vars = self.forward_ext_prot_embed(runtime_vars)
            bond_type_short = parse_hetero_edge(bond_type)[1]
            nmdn_params["edge_type"].append(torch.zeros_like(runtime_vars["C_batch"]).fill_(BOND2IDX[bond_type_short]))
            for key in ["pi", "sigma", "mu", "dist", "C_batch", "pl_edge_index_used"]:
                nmdn_params[key].append(runtime_vars.pop(key))
        for key in nmdn_params.keys():
            cat_dim = 1 if "edge_index" in key else 0
            nmdn_params[key] = torch.concat(nmdn_params[key], dim=cat_dim)
        runtime_vars.update(nmdn_params)
        return runtime_vars

    def pair_embed_modify(self, pair_embed, h_1, h_2, h_1_i, h_2_j, edge_index, dist, data_batch, pair_batch):
        if not self.add_seq_distance:
            return pair_embed
        
        seq_dist: torch.LongTensor = (edge_index[0] - edge_index[1]).abs()
        if self.seq_distance_dim is None:
            seq_dist = seq_dist.type_as(pair_embed)
            pair_embed = torch.concat([pair_embed, seq_dist.view(-1, 1)], dim=-1)
            return pair_embed
        
        # cap seq dist
        seq_dist[seq_dist >= self.seq_distance_cap] = self.seq_distance_cap-1
        seq_dist_embedding = self.seq_distance_embedding(seq_dist)
        pair_embed = torch.concat([pair_embed, seq_dist_embedding], dim=-1)
        return pair_embed
    
    def predict_gaussian_params(self, pair_embeds: Tensor) -> Tuple[Tensor]:
        if self.current_edge_name is None:
            return super().predict_gaussian_params(pair_embeds)
    
        bond_type_short = parse_hetero_edge(self.current_edge_name)[1]
        z_pi = getattr(self, f"z_pi_{bond_type_short}")
        z_sigma = getattr(self, f"z_sigma_{bond_type_short}")
        z_mu = getattr(self, f"z_mu_{bond_type_short}")
        pi = F.softmax(z_pi(pair_embeds), -1)
        sigma = F.elu(z_sigma(pair_embeds))+1.1
        mu = F.elu(z_mu(pair_embeds))+1
        return pi, sigma, mu

    def overwrite_lig_dim(self) -> Optional[int]:
        return get_prot_dim(self.cfg)

    def unpack_pl_info(self, runtime_vars: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Data]:
        h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, __ = super().unpack_pl_info(runtime_vars)
        pair_batch = data_batch["protein"].batch[pp_edge[0, :]]
        # exlcude close interactions. For example, seq_separation_cutoff==1 means exlcude 1-2 interaction
        # seq_separation_cutoff==2 means exclude 1-2 and 1-3 interaction.
        if self.seq_separation_cutoff is not None:
            seq_dist: torch.LongTensor = (pp_edge[0] - pp_edge[1]).abs()
            if self.ltype == "non-local":
                edge_mask: torch.BoolTensor = seq_dist > self.seq_separation_cutoff
            else:
                assert self.ltype == "local"
                edge_mask: torch.BoolTensor = seq_dist <= self.seq_separation_cutoff
            pp_edge = pp_edge[:, edge_mask]
            pp_dist = pp_dist[edge_mask]
            h_1_i = h_1_i[edge_mask, :]
            h_2_j = h_2_j[edge_mask, :]
            pair_batch = pair_batch[edge_mask]
        # only obtain needed pl_edges to avoid un-needed calculation
        if self.cutoff_needed is not None:
            this_pl_dist_mask = (pp_dist <= self.cutoff_needed).view(-1)
            pp_edge = pp_edge[:, this_pl_dist_mask]
            pp_dist = pp_dist[this_pl_dist_mask]

        return h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, pair_batch
    
    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        bond_type = self.current_edge_name
        assert bond_type is not None
        bond_type_parsed = parse_hetero_edge(bond_type)
        edge_store = data_batch[bond_type_parsed]
        pp_dist = edge_store.dist
        edge_index = edge_store.edge_index
        return edge_index, pp_dist
    
