import os.path as osp
import signal
import subprocess
from tempfile import TemporaryDirectory


SCRIPT_ROOT = "/ai/moleculeos/software"
PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite-1.0/bin/prepare_receptor"
PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite-1.0/bin/prepare_ligand"
LINF9_PATH = "/ai/share/workspace/songxia/Lin_F9"
LINF9 = f"{LINF9_PATH}/smina.static"

def handler(signum, frame):
    raise RuntimeError("TIMEOUT running PREPARE_LIG")

class LinF9LocalOptimizer:
    def __init__(self, protein_pdb=None, ligand_file=None, ligand_linf9_opt=None, 
                 ligand_mol2=None, protein_pdbqt=None, ligand_pdbqt=None) -> None:
        self.protein_pdb = protein_pdb
        self.ligand_mol2 = ligand_mol2
        self.ligand_file = ligand_file

        self.protein_pdbqt = protein_pdbqt
        self.ligand_pdbqt = ligand_pdbqt
        self.ligand_linf9_opt = ligand_linf9_opt

    def run(self, protdir=None):
        temp_dir = TemporaryDirectory()
        if self.protein_pdbqt is None:
            self.protein_pdbqt = osp.join(temp_dir.name if protdir is None else protdir, 
                                          osp.basename(self.protein_pdb).replace(".pdb", ".pdbqt"))
        if self.ligand_pdbqt is None:
            self.ligand_pdbqt = osp.join(temp_dir.name, "ligand.pdbqt")
        if not osp.exists(self.protein_pdbqt):
            assert self.protein_pdb is not None
            subprocess.run(f"{PREPARE_PROT} -r {self.protein_pdb} -U nphs_lps -A 'checkhydrogens' -o {self.protein_pdbqt} ", shell=True, check=True)
        if not osp.exists(self.ligand_pdbqt):
            ligand_mol2 = self.ligand_mol2
            if self.ligand_mol2 is None:
                assert self.ligand_file is not None
                mol2_file = osp.join(temp_dir.name, "tmp.mol2")
                lig_fmt = self.ligand_file.split(".")[-1]
                conv_cmd = f"obabel -i{lig_fmt} {self.ligand_file} -omol2 -O {mol2_file}"
                print(conv_cmd)
                subprocess.run(conv_cmd, shell=True, check=True)
                ligand_mol2 = mol2_file
            ligand_dir = osp.dirname(ligand_mol2)
            lig_cmd = f"{PREPARE_LIG} -l {ligand_mol2} -U nphs_lps -A 'checkhydrogens' -o {self.ligand_pdbqt} "
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)
            subprocess.run(lig_cmd, shell=True, check=True, cwd=ligand_dir)
            signal.alarm(0)
        linf9_cmd = f"{LINF9} -r {self.protein_pdbqt} -l {self.ligand_pdbqt} --local_only --scoring Lin_F9 -o {self.ligand_linf9_opt} "
        subprocess.run(linf9_cmd, shell=True, check=True)
        temp_dir.cleanup()

