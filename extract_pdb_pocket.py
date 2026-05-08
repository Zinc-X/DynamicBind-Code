import numpy as np
import warnings
import argparse
import os

from rdkit import Chem
from Bio.PDB import PDBParser
import Bio.PDB as bpdb
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial.distance import cdist

biopython_parser = PDBParser()


d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def find_pdb_pocket(rec_path, lig_coords, rec_pocket_dist=10, keep_water=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]

    valid_chain_ids = []
    fasta_list=[]
    
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        invalid_res_ids = []
        seq = ''
        for res_idx, residue in enumerate(chain):
            residue_is_water = False
            if residue.get_resname() == 'HOH' and not keep_water:
                residue_is_water = True
            elif residue.get_resname() in d3to1:
                seq+=d3to1[residue.resname]

            residue_coords = []
            res_in_pocket = False
            for atom in residue:
                atom_coord = np.array(list(atom.get_vector()))
                if cdist(atom_coord.reshape((1,-1)), lig_coords).min() < rec_pocket_dist: # Check if one of the atoms of the residue is in the pocket
                    res_in_pocket = True
                residue_coords.append(list(atom.get_vector()))    

            if res_in_pocket is True and residue_is_water is not True:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_coords.append(np.array(residue_coords))
            else:
                invalid_res_ids.append(residue.get_id()) 
        if seq!='':
            fasta_list.append([f'chain_{chain.get_id()}', seq])

        for res_id in invalid_res_ids:
            chain.detach_child(res_id)

        if len(chain_coords) > 0: #and not chain_is_water:
            valid_chain_ids.append(chain.get_id())

    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() not in valid_chain_ids:
            invalid_chain_ids.append(chain.get_id())

    for invalid_id in invalid_chain_ids: # Remove invalid chains
        rec.detach_child(invalid_id)

    return rec, fasta_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--p_file", type=str, default='v8_gwt1_ai_rescore/pro_gwt1-min.pdb',
                        help=" input protein pdb file")
    parser.add_argument("-l", "--l_file", type=str, default='v8_gwt1_ai_rescore/ref_lig.sdf',
                        help="reference ligand sdf file")
    
    parser.add_argument("-w", "--keep_water", action="store_true", default=False, help="Keep water molecules")
    parser.add_argument("-d", "--pocket_dist", type=float, default=5, help="pocket distance from input ligand")                    
    parser.add_argument("-c", "--out_dir", type=str, default='./', help="output directory")

    args = parser.parse_args()

    p_file_path = args.p_file
    l_file_path = args.l_file
    pocket_dist= args.pocket_dist
    keep_water = args.keep_water
    out_dir = args.out_dir

    supplier = Chem.SDMolSupplier(l_file_path, sanitize=True)
    lig = supplier[0]
    lig_coords = lig.GetConformer().GetPositions()

    pocket_rec, _ = find_pdb_pocket(p_file_path, lig_coords, rec_pocket_dist=pocket_dist, keep_water=keep_water)

    p_basename = os.path.basename(p_file_path).rsplit('.')[0]
    l_basename = os.path.basename(l_file_path).rsplit('.')[0]

    io=bpdb.PDBIO()
    io.set_structure(pocket_rec)
    io.save(os.path.join(out_dir, f'{p_basename}_{l_basename}_pocket{int(pocket_dist)}.pdb'))
