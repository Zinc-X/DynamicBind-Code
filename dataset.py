import pickle
import random
from typing import Any, Dict, List, Tuple, Union
from copy import deepcopy

import numpy as np
import torch
import os, glob
import os.path as osp
from rdkit import Chem, RDLogger
from rdkit.Chem import Atom, Mol, rdFMCS
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import rdFreeSASA
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix

from typing import Optional, Callable

from tqdm import tqdm

from utils_dynamic_bind import DynamicBindConfig

RDLogger.DisableLog("rdApp.*")
random.seed(0)

INTERACTION_TYPES = [
    "saltbridge",
    "hbonds",
    "pication",
    "pistack",
    "halogen",
    "waterbridge",
    "hydrophobic",
    "metal_complexes",
]
pt = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""
PERIODIC_TABLE = dict()
for i, per in enumerate(pt.split()):
    for j, ele in enumerate(per.split(",")):
        PERIODIC_TABLE[ele] = (i, j)
PERIODS = [0, 1, 2, 3, 4, 5]
GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
FORMALCHARGES = [-2, -1, 0, 1, 2, 3, 4]
METALS = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
HYDROPHOBICS = ["F", "CL", "BR", "I"]
VDWRADII = {
    6: 1.90,
    7: 1.8,
    8: 1.7,
    16: 2.0,
    15: 2.1,
    9: 1.5,
    17: 1.8,
    35: 2.0,
    53: 2.2,
    30: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    12: 1.2,
    28: 1.2,
    20: 1.2,
    29: 1.2,
}
HBOND_DONOR_INDICES = ["[!#6;!H0]"]
HBOND_ACCEPPTOR_SMARTS = [
    "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]
HBond_DONOR_Protein_N_Main = ["[N;!H0]-C-C=O"]
HBond_ACCEPPTOR_Protein_O_Main = ["[O;H0]=C-C-N"]
VdW_AlphaCarbon = ["C(-N)-C=O"]

def get_period_group(atom: Atom) -> List[bool]:
    period, group = PERIODIC_TABLE[atom.GetSymbol().upper()]
    return one_of_k_encoding(period, PERIODS) + one_of_k_encoding(group, GROUPS)


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: Any, allowable_set: List[Any]) -> List[bool]:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(mol: Mol, atom_index: int) -> np.ndarray:
    atom = mol.GetAtomWithIdx(atom_index)
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), DEGREES)
        + one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + one_of_k_encoding_unk(atom.GetFormalCharge(), FORMALCHARGES)
        + get_period_group(atom)
        + [atom.GetIsAromatic()]
    )  # (9, 6, 7, 7, 24, 1) --> total 54


def get_atom_feature(mol: Mol) -> np.ndarray:
    natoms = mol.GetNumAtoms()
    H = []
    for idx in range(natoms):
        H.append(atom_feature(mol, idx))
    H = np.array(H)
    return H


def get_vdw_radius(atom: Atom) -> float:
    atomic_num = atom.GetAtomicNum()
    if VDWRADII.get(atomic_num):
        return VDWRADII[atomic_num]
    return Chem.GetPeriodicTable().GetRvdw(atomic_num)


def get_hydrophobic_atom(mol: Mol) -> np.ndarray:
    natoms = mol.GetNumAtoms()
    hydrophobic_indice = np.zeros((natoms,))
    for atom_idx in range(natoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol.upper() in HYDROPHOBICS:
            hydrophobic_indice[atom_idx] = 1
        elif symbol.upper() in ["C"]:
            neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
            neighbors_wo_c = list(set(neighbors) - set(["C"]))
            if len(neighbors_wo_c) == 0:
                hydrophobic_indice[atom_idx] = 1
    return hydrophobic_indice


def get_A_hydrophobic(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_indice = get_hydrophobic_atom(ligand_mol)
    target_indice = get_hydrophobic_atom(target_mol)
    return np.outer(ligand_indice, target_indice)


def get_hbond_atom_indices(mol: Mol, smarts_list: List[str]) -> np.ndarray:
    indice = []
    for smarts in smarts_list:
        smarts = Chem.MolFromSmarts(smarts)
        indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
    indice = np.array(indice)
    return indice

def get_A_hbond(ligand_mol: Mol, target_mol: Mol, main_chain_ratio=1.0) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(ligand_mol, HBOND_ACCEPPTOR_SMARTS)
    target_h_acc_indice = get_hbond_atom_indices(target_mol, HBOND_ACCEPPTOR_SMARTS)
    ligand_h_donor_indice = get_hbond_atom_indices(ligand_mol, HBOND_DONOR_INDICES)
    target_h_donor_indice = get_hbond_atom_indices(target_mol, HBOND_DONOR_INDICES)

    target_h_acc_O_indice = list(get_hbond_atom_indices(target_mol, HBond_ACCEPPTOR_Protein_O_Main))
    target_h_donor_N_indice = list(get_hbond_atom_indices(target_mol, HBond_DONOR_Protein_N_Main))

    hbond_indice = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
    hbond_indice_lig_acc = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
    hbond_indice_lig_donor = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))

    for i in ligand_h_acc_indice:
        for j in target_h_donor_indice:
            if j in  target_h_donor_N_indice:
                hbond_indice[i, j] = main_chain_ratio
                hbond_indice_lig_acc[i,j] = main_chain_ratio
            else:
                hbond_indice[i, j] = 1
                hbond_indice_lig_acc[i,j] = 1
    for i in ligand_h_donor_indice:
        for j in target_h_acc_indice:
            if j in target_h_acc_O_indice:
                hbond_indice[i, j] = main_chain_ratio
                hbond_indice_lig_donor[i, j] = main_chain_ratio
            else:    
                hbond_indice[i, j] = 1
                hbond_indice_lig_donor[i, j] = 1
    return hbond_indice, hbond_indice_lig_acc, hbond_indice_lig_donor

def get_alphaCarbon_indices(mol: Mol, smarts_list: List[str]) -> np.ndarray:
    indice = []
    for smarts in smarts_list:
        smarts = Chem.MolFromSmarts(smarts)
        indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
    indice = np.array(indice)
    return indice


def get_A_metal_complexes(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
    ligand_h_acc_indice = get_hbond_atom_indices(ligand_mol, HBOND_ACCEPPTOR_SMARTS)
    target_h_acc_indice = get_hbond_atom_indices(target_mol, HBOND_ACCEPPTOR_SMARTS)
    ligand_metal_indice = np.array(
        [
            idx
            for idx in range(ligand_mol.GetNumAtoms())
            if ligand_mol.GetAtomWithIdx(idx).GetSymbol() in METALS
        ]
    )
    target_metal_indice = np.array(
        [
            idx
            for idx in range(target_mol.GetNumAtoms())
            if target_mol.GetAtomWithIdx(idx).GetSymbol() in METALS
        ]
    )

    metal_indice = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
    for ligand_idx in ligand_h_acc_indice:
        for target_idx in target_metal_indice:
            metal_indice[ligand_idx, target_idx] = 1
    for ligand_idx in ligand_metal_indice:
        for target_idx in target_h_acc_indice:
            metal_indice[ligand_idx, target_idx] = 1
    return metal_indice

# def rmsd_cal3(mol, jmol):
#     r = rdFMCS.FindMCS([mol, jmol])
#     # Atom map for reference and target              
#     a = mol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
#     b = jmol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
#     # Atom map generation     
#     amap = list(zip(a,b))
#     # distance calculation per atom pair
#     distances = []
#     for atomA, atomB in amap:
#         pos_A = mol.GetConformer().GetAtomPosition (atomA)
#         pos_B = jmol.GetConformer().GetAtomPosition (atomB)
#         coord_A = np.array((pos_A.x,pos_A.y,pos_A.z))
#         coord_B = np.array ((pos_B.x,pos_B.y,pos_B.z))
#         dist_numpy = np.linalg.norm(coord_A-coord_B)        
#         distances.append(dist_numpy)

#     # This is the RMSD formula from wikipedia
#     rmsd = np.sqrt(1/len(distances)*sum([i*i for i in distances]))
#     return rmsd



def mol_to_feature(ligand_mol: Mol, target_mol: Mol, prot_hbond_main_chain_ratio=1.0,
                                                     prot_alphaC_ratio=1.0) -> Dict[str, Any]:
    # Remove hydrogens
    ligand_mol = Chem.RemoveHs(ligand_mol)
    target_mol = Chem.RemoveHs(target_mol)

    
    # prepare ligand
    ligand_natoms = ligand_mol.GetNumAtoms()
    ligand_pos = np.array(ligand_mol.GetConformers()[0].GetPositions())
    ligand_adj = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_natoms)
    ligand_h = get_atom_feature(ligand_mol)
    
    # prepare protein
    target_natoms = target_mol.GetNumAtoms()
    target_pos = np.array(target_mol.GetConformers()[0].GetPositions())
    target_adj = GetAdjacencyMatrix(target_mol) + np.eye(target_natoms)
    target_h = get_atom_feature(target_mol)


    interaction_indice = np.zeros(
        (3, ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
    )
    hbond_indice = get_A_hbond(ligand_mol, target_mol, main_chain_ratio=prot_hbond_main_chain_ratio)[0]
    interaction_indice[0] = hbond_indice
    interaction_indice[1] = get_A_metal_complexes(ligand_mol, target_mol)
    interaction_indice[0, interaction_indice[1]==1] = 0 #correction
    interaction_indice[2] = get_A_hydrophobic(ligand_mol, target_mol)

    # count rotatable bonds
    rotor = CalcNumRotatableBonds(ligand_mol)

    # valid
    ligand_valid = np.ones((ligand_natoms,))
    target_valid = np.ones((target_natoms,))

    # no metal
    ligand_non_metal = np.array(
        [1 if atom.GetSymbol() not in METALS else 0 for atom in ligand_mol.GetAtoms()]
    )
    target_non_metal = np.array(
        [1 if atom.GetSymbol() not in METALS else 0 for atom in target_mol.GetAtoms()]
    )

    protein_alpha_carbon_index_list = get_alphaCarbon_indices(target_mol, VdW_AlphaCarbon) 
    target_non_metal = target_non_metal.astype('float32')
    if len(protein_alpha_carbon_index_list) > 0: 
        target_non_metal[protein_alpha_carbon_index_list]=prot_alphaC_ratio

    # vdw radius
    ligand_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in ligand_mol.GetAtoms()]
    )
    target_vdw_radii = np.array(
        [get_vdw_radius(atom) for atom in target_mol.GetAtoms()]
    )

    sample = {
        "ligand_h": ligand_h,
        "ligand_adj": ligand_adj,
        "target_h": target_h,
        "target_adj": target_adj,
        "interaction_indice": interaction_indice,
        "ligand_pos": ligand_pos,
        "target_pos": target_pos,
        "rotor": rotor,
        "ligand_vdw_radii": ligand_vdw_radii,
        "target_vdw_radii": target_vdw_radii,
        "ligand_valid": ligand_valid,
        "target_valid": target_valid,
    }
    return sample

# class docking_features():
#     def __init__(self, init_ref_mol: Mol, target_mol: Mol, prot_hbond_main_chain_ratio,
#                                                 prot_alphaC_ratio):
#         # Target features are fixed when docking
#         self.prot_hbond_main_chain_ratio = prot_hbond_main_chain_ratio
#         self.prot_alphaC_ratio = prot_alphaC_ratio
#         self.target_mol = Chem.RemoveHs(target_mol)

#         self.target_natoms = self.target_mol.GetNumAtoms()
#         self.target_pos = np.array(self.target_mol.GetConformers()[0].GetPositions())
#         self.target_adj = GetAdjacencyMatrix(self.target_mol) + np.eye(self.target_natoms)
#         self.target_h = get_atom_feature(self.target_mol)
#         self.target_valid = np.ones((self.target_natoms,))

#         # Parts of referance features are fixed when docking 
#         self.ligand_natoms = init_ref_mol.GetNumAtoms()
#         self.ligand_adj = GetAdjacencyMatrix(init_ref_mol) + np.eye(self.ligand_natoms)

#         self.interaction_indice = np.zeros(
#             (3, init_ref_mol.GetNumAtoms(), self.target_mol.GetNumAtoms())
#         )
#         hbond_indice = get_A_hbond(init_ref_mol, self.target_mol, main_chain_ratio=self.prot_hbond_main_chain_ratio)[0]
#         self.interaction_indice[0] = hbond_indice
#         self.interaction_indice[1] = get_A_metal_complexes(init_ref_mol, self.target_mol)
#         self.interaction_indice[0, self.interaction_indice[1]==1] = 0 #correction
#         self.interaction_indice[2] = get_A_hydrophobic(init_ref_mol, self.target_mol)

#         self.rotor = CalcNumRotatableBonds(init_ref_mol)

#         # valid
#         self.ligand_valid = np.ones((self.ligand_natoms,))
#         self.ligand_vdw_radii = np.array(
#             [get_vdw_radius(atom) for atom in init_ref_mol.GetAtoms()]
#         )
#         self.target_vdw_radii = np.array(
#             [get_vdw_radius(atom) for atom in self.target_mol.GetAtoms()]
#         )
        
        
#     def mol_to_feature(self, ligand_mol: Mol) -> Dict[str, Any]:
#         ligand_pos = np.array(ligand_mol.GetConformers()[0].GetPositions())
#         ligand_h = get_atom_feature(ligand_mol)
#         sample = {
#             "ligand_h": ligand_h,
#             "ligand_adj": self.ligand_adj,
#             "target_h": self.target_h,
#             "target_adj": self.target_adj,
#             "interaction_indice": self.interaction_indice,
#             "ligand_pos": ligand_pos,
#             "target_pos": self.target_pos,
#             "rotor": self.rotor,
#             "ligand_vdw_radii": self.ligand_vdw_radii,
#             "target_vdw_radii": self.target_vdw_radii,
#             "ligand_valid": self.ligand_valid,
#             "target_valid": self.target_valid,
#         }
#         return sample


def load(ref_path, tar_path):
    ref_name, ref_ext  = os.path.basename(ref_path).rsplit('.', 1)
    tar_name, tar_ext = os.path.basename(tar_path).rsplit('.', 1)

    if ref_ext=='mol2':
        ref_mol = Chem.MolFromMol2File(ref_path, sanitize=True, removeHs=True)
    elif ref_ext=='sdf':    
        ref_mol = Chem.SDMolSupplier(ref_path, sanitize=True, removeHs=True)[0]
    elif ref_ext=='pdb': 
        ref_mol = Chem.MolFromPDBFile(ref_path, sanitize=True, removeHs=True)
    else:
        print('reference file format is is not supported. (supported formats: .sdf, .mol2, .pdb)')
        exit()

    if tar_ext=='mol2':
        tar_mol = Chem.MolFromMol2File(tar_path, sanitize=True, removeHs=True)
    elif tar_ext=='sdf':    
        tar_mol = Chem.SDMolSupplier(tar_path, sanitize=True, removeHs=True)[0]
    elif tar_ext=='pdb':  
        tar_mol = Chem.MolFromPDBFile(tar_path, sanitize=True, removeHs=True)
    else:
        print('target file format is is not supported. (supported formats: .sdf, .mol2, .pdb)')
        exit()        
    return ref_mol, tar_mol, ref_name, tar_name

class ComplexDatasetTrainMD(Dataset):
    """
    Implemented by Song Xia
    Pick a random frame from a MD trajectory as input.
    """
    def __init__(self, keys: List[str], data_dir: str, md_data_dir: str,
                 id_to_y: Dict[str, float], cfg: DynamicBindConfig):
        self.cfg = cfg
        self.keys = keys
        self.data_dir = data_dir
        self.md_data_dir = md_data_dir
        self.id_to_y = id_to_y

        self.preload_md_meta_info()

        self.n_total_frames = cfg.n_total_frames
        traj_total_frames = 500
        if cfg.md_use_first_n_frames is not None:
            traj_total_frames = cfg.md_use_first_n_frames
        self.every = traj_total_frames // self.n_total_frames

    def preload_md_meta_info(self):
        pdb2_md_folder = {}
        pdb2_traj_data = {}
        all_ligand_sdf_files = glob.glob(osp.join(self.md_data_dir, "DynamicBinding_*", "????", "*", "ligand.sdf"))
        if self.cfg.md_first_n is not None:
            all_ligand_sdf_files = all_ligand_sdf_files[:self.cfg.md_first_n]

        for ligand_sdf in tqdm(all_ligand_sdf_files, desc="Prep traj data"):
            pdb_id = osp.basename(osp.dirname(osp.dirname(ligand_sdf)))
            if pdb_id not in self.keys:
                continue
            pdb2_md_folder[pdb_id] = osp.dirname(ligand_sdf)
            prot_mol = Chem.MolFromPDBFile(osp.join(pdb2_md_folder[pdb_id], "protein_pocket5.pdb"), removeHs=False, sanitize=True)
            lig_mol = Chem.SDMolSupplier(ligand_sdf, removeHs=False, sanitize=True)[0]
            if prot_mol is None or lig_mol is None:
                continue

            traj_data_npz = osp.join(pdb2_md_folder[pdb_id], "traj.npz")
            traj_data = np.load(traj_data_npz)

            prot_numbers = np.asarray([atom.GetAtomicNum() for atom in prot_mol.GetAtoms()])
            lig_numbers = np.asarray([atom.GetAtomicNum() for atom in lig_mol.GetAtoms()])
            ligand_coords = traj_data["ligand_coords"][:, lig_numbers!=1]
            protein_pocket_coords = traj_data["protein_pocket_coords"][:, prot_numbers!=1]

            # sanity check
            lig_mol_noH = Chem.RemoveHs(lig_mol)
            prot_mol_noH = Chem.RemoveHs(prot_mol)
            if ligand_coords.shape[1] != lig_mol_noH.GetNumAtoms() or protein_pocket_coords.shape[1] != prot_mol_noH.GetNumAtoms():
                continue

            pdb2_traj_data[pdb_id] = {
                "ligand_coords": ligand_coords,
                "protein_pocket_coords": protein_pocket_coords
            }
        self.pdb2_md_folder = pdb2_md_folder
        self.pdb2_traj_data = pdb2_traj_data

    def __len__(self) -> int:
        return len(self.keys)
    
    def load_static_sample(self, idx: int, l_file_path: Optional[str] = None, p_file_path: Optional[str] = None):
        key = self.keys[idx]
        "Loading scoring data"
        score_path = f'{self.data_dir}/{key}/{key}'
        # they should be specified at the same time
        if l_file_path is not None:
            assert p_file_path is not None
        if p_file_path is not None:
            assert l_file_path is not None

        if l_file_path is None or p_file_path is None:
            l_file_path = f'{score_path}_lig.sdf'
            p_file_path = f'{score_path}_pocket5.pdb'

        native_lig, prot, _, _= load(l_file_path, p_file_path)
        if native_lig is None or prot is None:
            return self.load_static_sample(idx)
        
        sample = mol_to_feature(native_lig, prot)
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample, None

    def load_dynamic_sample(self, idx: int):
        key = self.keys[idx]
        if key not in self.pdb2_traj_data:
            sample, __ = self.load_static_sample(idx)
            ligand_pos = sample["ligand_pos"]
            target_pos = sample["target_pos"]
            sample["lig_traj_coords"] = np.stack([ligand_pos for __ in range(self.n_total_frames)], axis=0)
            sample["target_traj_coords"] = np.stack([target_pos for __ in range(self.n_total_frames)], axis=0)
            return sample, None

        md_folder = self.pdb2_md_folder[key]
        lig_sdf = osp.join(md_folder, "ligand.sdf")
        prot_pdb = osp.join(md_folder, "protein_pocket5.pdb")
        sample, __ = self.load_static_sample(idx, lig_sdf, prot_pdb)

        traj_data = self.pdb2_traj_data[key]
        lig_coords = traj_data["ligand_coords"][:self.cfg.md_use_first_n_frames][::self.every]
        prot_coords = traj_data["protein_pocket_coords"][:self.cfg.md_use_first_n_frames][::self.every]
        sample["lig_traj_coords"] = lig_coords
        sample["target_traj_coords"] = prot_coords
        # check the consistency between structure and traj coordinates
        assert sample["ligand_pos"].shape[0] == lig_coords.shape[1], f"{key}: {sample['ligand_pos'].shape[0]} != {lig_coords.shape[1]}"
        assert sample["target_pos"].shape[0] == prot_coords.shape[1], f"{key}: {sample['target_pos'].shape[0]} != {prot_coords.shape[1]}"
        return sample, None

    
    def __getitem__(self, idx: int):
        if self.cfg.whole_traj_learning:
            return self.load_dynamic_sample(idx)
        
        key = self.keys[idx]
        if key not in self.pdb2_md_folder:
            return self.load_static_sample(idx)

        # data augmentation with a random frame
        if self.cfg.static_sample_weight is not None:
            static_prob = (self.cfg.static_sample_weight) / (1 + self.cfg.static_sample_weight)
            rd_num = np.random.rand()
            if rd_num < static_prob:
                return self.load_static_sample(idx)
        
        md_folder = self.pdb2_md_folder[key]
        lig_sdf = osp.join(md_folder, "ligand.sdf")
        prot_pdb = osp.join(md_folder, "protein_pocket5.pdb")
        traj_data = self.pdb2_traj_data[key]
        lig_coords = traj_data["ligand_coords"]
        prot_coords = traj_data["protein_pocket_coords"]

        native_lig, prot, _, _= load(lig_sdf, prot_pdb)
        if native_lig is None or prot is None:
            return self.load_static_sample(idx)

        frame_num = np.random.randint(0, lig_coords.shape[0])
        conformer_lig = native_lig.GetConformer()
        for i in range(native_lig.GetNumAtoms()):
            x,y,z = lig_coords[frame_num][i]
            conformer_lig.SetAtomPosition(i,Point3D(x.item(),y.item(),z.item()))

        conformer_prot = prot.GetConformer()
        for i in range(prot.GetNumAtoms()):
            x,y,z = prot_coords[frame_num][i]
            conformer_prot.SetAtomPosition(i,Point3D(x.item(),y.item(),z.item()))

        sample = mol_to_feature(native_lig, prot)
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        # sample["frame_num"] = frame_num
        return sample, None

class ComplexDataset_train(Dataset):
    def __init__(self, keys: List[str], data_dir: str, md_data_dir: str,
                 id_to_y: Dict[str, float], use_md_data: bool):
        # use_md_data is over-written in ComplexDatasetTrainMD
        assert not use_md_data
        self.keys = keys
        self.data_dir = data_dir
        self.md_data_dir = md_data_dir
        self.id_to_y = id_to_y
        self.use_md_data = use_md_data
        
    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]

        "Loading scoring data"
        score_path = f'{self.data_dir}/{key}/{key}'
        l_file_path = f'{score_path}_lig.sdf'
        p_file_path = f'{score_path}_pocket5.pdb'

        native_lig, prot, _, _= load(l_file_path, p_file_path)
        sample = mol_to_feature(native_lig, prot)
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key

        "Loading MD data"
        if self.use_md_data is True:
            random_frame_num = random.randint(0, 99)
            md_path = f'{self.md_data_dir}/{key}/frame_{random_frame_num}'

            md_l_file_path = os.path.join(md_path, f'ligand_*_frame_{random_frame_num}.sdf')
            matching_files = glob.glob(md_l_file_path)
            md_l_file_path = matching_files[0] if matching_files else os.path.join(md_path, f'ligand_frame_{random_frame_num}.sdf')

            md_p_file_path_pattern = os.path.join(md_path, 
                f'protein_frame_{random_frame_num}_ligand_*_frame_{random_frame_num}_pocket5.pdb')
            matching_files = glob.glob(md_p_file_path_pattern)
            md_p_file_path = matching_files[0] if matching_files else os.path.join(md_path, 
                f'protein_frame_{random_frame_num}_ligand_frame_{random_frame_num}_pocket5.pdb')
                
            # md_l_file_path = os.path.join(md_path, f'ligand_frame_{random_frame_num}.sdf')
            # md_p_file_path = os.path.join(md_path, 
            #      f'protein_frame_{random_frame_num}_ligand_frame_{random_frame_num}_pocket5.pdb')
            try:
                md_lig, md_prot, _, _= load(md_l_file_path, md_p_file_path)
                sample_md = mol_to_feature(md_lig, md_prot)
                sample_md["affinity"] = self.id_to_y[key] * -1.36
                sample_md["key"] = f'{key}_md_{random_frame_num}'
            except:
                return sample, sample
            return sample, sample_md
        else:
            return sample, None
        

def get_dataset_dataloader_train(cfg: DynamicBindConfig, keys: List[str], data_dir: str, md_data_dir: str,
                                 id_to_y: Dict[str, float], batch_size: int,
                                 num_workers: int,
                                 use_md_data: bool = False,
                                 train: bool = True,
                                 worker_init_fn: Optional[Callable] = None
                                ) -> Tuple[Dataset, DataLoader]:

    if use_md_data:
        dataset = ComplexDatasetTrainMD(keys, data_dir, md_data_dir, id_to_y, cfg)
    else:
        dataset = ComplexDataset_train(keys, data_dir, md_data_dir, id_to_y, use_md_data)
    dataloader = DataLoader(dataset, batch_size, 
                            num_workers=num_workers,
                            collate_fn=tensor_collate_fn,
                            shuffle=train,
                            worker_init_fn=worker_init_fn)
    return dataset, dataloader

def get_dataset_dataloader(
    screening_list: List[str],
    id_to_y: Dict[str, float],
    batch_size: int,
    num_workers: int,
    train: bool = True,
    worker_init_fn: Optional[Callable] = None
) -> Tuple[Dataset, DataLoader]:

    dataset = ComplexDataset(screening_list, id_to_y)
    dataloader = DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        shuffle=train,
        worker_init_fn=worker_init_fn,
    )
    return dataset, dataloader


class ComplexDataset(Dataset):
    def __init__(self, screening_list: List[str], id_to_y: Dict[str, float]):
        self.screening_list = screening_list
        self.id_to_y = id_to_y
    def __len__(self) -> int:
        return len(self.screening_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref_mol, tar_mol, ref_name, tar_name = self.screening_list[idx]

        sample = mol_to_feature(ref_mol, tar_mol)
        sample["affinity"] = 0
        if tar_name=='':
            sample["key"] = f'{ref_name}'
        else:
            sample["key"] = f'{ref_name}_{tar_name}'
        return sample, None
    
def check_dimension(tensors: List[Any]) -> Any:
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)

    return np.max(size, 0)


def collate_tensor(tensor: Any, max_tensor: Any, batch_idx: int) -> Any:
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        max_dims = max_tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor

    return max_tensor

def reorder_batch_tensor(batch):
    score_batch, md_batch = [], []
    for min_batch in batch:
        score_batch.append(min_batch[0])
        md_batch.append(min_batch[1])
    return score_batch, md_batch

def batch_fn(batch):
    if batch[0] is None:
        return None
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value) if j % n_element == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(np.array([batch_size, *check_dimension(value_list)]))
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    ret_dict = {}
    for j in range(batch_size):
        if batch[j] == None:
            continue
        keys = []
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value
    return ret_dict

def tensor_collate_fn(batch: List[Any]) -> Dict[str, Any]:
    score_batch,  md_batch= reorder_batch_tensor(batch)

    score_batch_final = batch_fn(score_batch)
    md_batch_final = batch_fn(md_batch)

    return score_batch_final, md_batch_final