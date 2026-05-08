import MDAnalysis as mda
import numpy as np
import os.path as osp
import subprocess
from MDAnalysis.topology.guessers import guess_types
import subprocess
from glob import glob
from tqdm import tqdm
import os
import os.path as osp
from tqdm.contrib.concurrent import process_map

def proc_one_gro(gro_file):
    save_dir = osp.dirname(gro_file)
    if osp.exists(osp.join(save_dir, 'ligand.sdf')):
        return
    
    universe = mda.Universe(gro_file, gro_file.replace(".gro", ".xtc"))

    # Select the ligand using the specified residue name 'LIG'
    ligand = universe.select_atoms('resname LIG')
    # Select protein pocket using residues within 5 Å of the ligand
    protein_pocket = universe.select_atoms(f'same residue as around 5 resname LIG')
    # Save protein pocket (from the last frame) as protein_pocket5.pdb
    protein_pocket.write(osp.join(save_dir, 'protein_pocket5.pdb'))

    if len(ligand) == 0:
        raise RuntimeError("No ligand found with resname LIG.")

    # Save the ligand as ligand.pdb
    # elements = guess_types(universe.atoms.names)
    # universe.add_TopologyAttr('elements', elements)
    # Save the ligand as ligand.pdb
    ligand.write(osp.join(save_dir, 'ligand.pdb'))

    def try_proc_ligand():
        try:
            subprocess.run("pdb2sdf ligand.pdb ligand.sdf", shell=True, check=True, cwd=save_dir)
        except Exception as e:
            subprocess.run("pdb2sdf ligand.pdb ligand_conv.pdb", shell=True, check=True, cwd=save_dir)
            subprocess.run("pdb2sdf ligand_conv.pdb ligand.sdf", shell=True, check=True, cwd=save_dir)
    try:
        try_proc_ligand()
    except Exception as e:
        # Save the ligand as ligand.pdb
        elements = guess_types(universe.atoms.names)
        universe.add_TopologyAttr('elements', elements)
        # Save the ligand as ligand.pdb
        ligand.write(osp.join(save_dir, 'ligand.pdb')) 
        try:
            try_proc_ligand()
        except Exception as e:
            print(f"Giving up on {gro_file}: {e}")

    # Analyze trajectory
    ligand_coords = []
    protein_pocket_coords = []

    for ts in universe.trajectory:
        # Update positions
        current_ligand_positions = ligand.positions.copy()
        # Append coordinates
        ligand_coords.append(current_ligand_positions)
        protein_pocket_coords.append(protein_pocket.positions.copy())

    # Convert lists to numpy arrays
    ligand_coords = np.array(ligand_coords)
    protein_pocket_coords = np.array(protein_pocket_coords)

    # Save coordinates as traj.npz
    np.savez(osp.join(save_dir, 'traj.npz'), ligand_coords=ligand_coords, protein_pocket_coords=protein_pocket_coords)

# Load the universe with the GRO and XTC files
gro_files = glob("/ai/share/workspace/sxia/smart_logic_data/DynamicBinding_*/????/*/nowat.gro")
process_map(proc_one_gro, gro_files)
# proc_one_gro("/ai/share/workspace/sxia/smart_logic_data/DynamicBinding_N1000_20251107/3ztx/PDB_Complex_3ztx_NPT_10W_lxl/nowat.gro")