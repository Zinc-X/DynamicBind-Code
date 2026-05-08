from typing import Union, Optional
import prody
from prody import AtomGroup, parsePDB, parsePDBStream
import torch as th
import numpy as np
from Bio import pairwise2

INF_DIST = 1000000.

def get_resnum_mapping(native_prot_pdb: str, pred_prot_pdb: str):
    mapper = {}
    native_ag = prody.parsePDB(native_prot_pdb).protein.toAtomGroup()
    pred_ag = prody.parsePDB(pred_prot_pdb).protein.toAtomGroup()
    native_seq = pdb2seq(prot_ag=native_ag)
    pred_seq = pdb2seq(prot_ag=pred_ag)
    pred2native_n = map_sequences(pred_seq, native_seq)

    native_resi_getter = [res.getResnum() for res in native_ag.iterResidues()]
    for res, native_n in zip(pred_ag.iterResidues(), pred2native_n):
        if native_n is None:
            continue
        pred_resnum = res.getResnum()
        mapper[pred_resnum] = native_resi_getter[native_n]
    return mapper

def map_sequences(seq1: str, seq2: str):
    # Align sequences using Biopython's pairwise2
    alignments = pairwise2.align.globalxx(seq1, seq2)
    alignment = alignments[0]  # Take the best alignment

    # Map residues from seq1 to seq2 based on the alignment
    seq1_aligned, seq2_aligned = alignment[0], alignment[1]
    mapping = []
    seq2_index = 0
    for i, (res1, res2) in enumerate(zip(seq1_aligned, seq2_aligned)):
        if res1 != '-' and res2 != '-':
            mapping.append(seq2_index)
        elif res1 != '-' and res2 == '-':
            mapping.append(None)
        if res2 != "-":
            seq2_index += 1
    return mapping

def pdb2seq(pdb_f: Optional[str] = None, pdb_stream = None, prot_ag: AtomGroup = None, chain: str = None, add_gap: bool = False):
    if pdb_f is not None:
        assert prot_ag is None
        prot_ag = parsePDB(pdb_f).protein.toAtomGroup()
    if pdb_stream is not None:
        assert prot_ag is None
        prot_ag = parsePDBStream(pdb_stream).protein.toAtomGroup()
    if chain is not None:
        prot_ag = prot_ag.select(f"chain {chain}").toAtomGroup()
    out = ""
    prev_resnum = 0
    for res in prot_ag.iterResidues():
        resnum = res.getResnum()
        if add_gap and resnum - prev_resnum > 1:
            out += "X" * (resnum - prev_resnum - 1)
        prev_resnum = resnum
        
        out += res.getSequence()[0]
    return out

def pdb2res_info(pdb_f: str):
    prot_ag = parsePDB(pdb_f).protein.toAtomGroup()
    out = []
    for res in prot_ag.iterResidues():
        res = res.toAtomGroup()
        chid = res.getChids()[0]
        resnum = res.getResnums()[0]
        resname = res.getResnames()[0]
        out.append(f"{chid};{resnum};{resname}")
    return out

def pdb2chain_seqs(pdb_f: str):
    prot_ag = parsePDB(pdb_f).protein.toAtomGroup()
    seqs = []
    for chain in prot_ag.iterChains():
        seqs.append(pdb2seq(prot_ag=chain.toAtomGroup()))
    return seqs

def pl_min_dist_matrix(lig_pos, prot_pos, device: str = "cpu"):
    """
    Calculate ligand-atom to protein residue minimum distance matrix.

    Input should be either numpy arrays or torch tensors;

    lig_pos should be [N_lig, 3] or [batch_size, N_lig, 3];
    prot_pos should be [N_aa, N_max_atom_per_aa, 3] or 
        [batch_size, N_aa, N_max_atom_per_aa, 3], respectively.
    """
    if isinstance(lig_pos, np.ndarray):
        lig_pos: th.Tensor = th.as_tensor(lig_pos)
    if isinstance(prot_pos, np.ndarray):
        prot_pos: th.Tensor = th.as_tensor(prot_pos)
    lig_pos = lig_pos.double().to(device)
    prot_pos = prot_pos.double().to(device)
    assert lig_pos.dim() == prot_pos.dim() - 1, f"error sizes: {lig_pos.shape}, {prot_pos.shape}"
    if lig_pos.dim() == 2:
        lig_pos = lig_pos.unsqueeze(0)
        prot_pos = prot_pos.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = lig_pos.shape[0]

    # based on torch.cdist
    pairdist = th.cdist(lig_pos, prot_pos)
    pairdist = th.nan_to_num(pairdist, INF_DIST)
    mindist1 = pairdist.min(axis=-1)[0].permute(0, 2, 1)
    return mindist1

    # old script discarded -- switching to torch.cdist
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY

    N_l = lig_pos.shape[-2]
    N_max = prot_pos.shape[-2]

    prot_pos = prot_pos.view(batch_size, -1, 3)
    
    dists = -2 * th.bmm(lig_pos, prot_pos.permute(0, 2, 1)) + th.sum(prot_pos**2,    axis=-1).unsqueeze(1) + th.sum(lig_pos**2, axis=-1).unsqueeze(-1)	
    pairdist = th.nan_to_num((dists**0.5).view(batch_size, N_l,-1,N_max), INF_DIST)
    mindist = pairdist.min(axis=-1)[0]
    breakpoint()
    return mindist

def pp_min_dist_matrix_vec(prot_pos):
    """
    Calculate protein residue-residue minimum distance matrix.
    prot_pos should be [N_aa, N_max_atom_per_aa, 3] or [batch_size, N_aa, N_max_atom_per_aa, 3].

    return: [batch_size, N_aa, N_aa]
    """
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    if isinstance(prot_pos, np.ndarray):
        prot_pos = th.as_tensor(prot_pos)
    prot_pos = prot_pos.double()

    if prot_pos.dim() == 3:
        prot_pos = prot_pos.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = prot_pos.shape[0]

    N_max = prot_pos.shape[-2]
    N_aa = prot_pos.shape[-3]
    prot_pos = prot_pos.view(batch_size, -1, 3)
    dists = -2 * th.bmm(prot_pos, prot_pos.permute(0, 2, 1)) + th.sum(prot_pos**2,    axis=-1).unsqueeze(1) + th.sum(prot_pos**2, axis=-1).unsqueeze(-1)	
    dists = th.nan_to_num((dists**0.5).view(batch_size, N_aa, N_max, N_aa, N_max), INF_DIST)
    min_dist = dists.min(axis=-1)[0].min(axis=-2)[0]
    return min_dist

def pp_min_dist_matrix_vec_mem(prot1_pos: Union[np.ndarray, th.Tensor], prot2_pos: Union[np.ndarray, th.Tensor] = None, return_max=False) -> th.Tensor:
    """
    WINNER!!!
    
    A memory efficient (but slower) implementation of PP min dist calculation.

    WARNING: only works for proteins
    """
    if isinstance(prot1_pos, np.ndarray):
        prot1_pos = th.as_tensor(prot1_pos)
    if isinstance(prot2_pos, np.ndarray):
        prot2_pos = th.as_tensor(prot2_pos)
    # self-self interaction
    if prot2_pos is None:
        prot2_pos = prot1_pos

    # the input shape should be: [N_aa, N_max, 3]
    assert prot1_pos.dim() == 3, prot1_pos.shape
    assert prot2_pos.dim() == 3, prot2_pos.shape

    n_aas = prot1_pos.shape[0]
    pp_min_martix = []
    pp_max_matrix = []
    for aa_id in range(n_aas):
        selec_aa = prot1_pos[aa_id, :, :]
        selec_pp = pl_min_dist_matrix(selec_aa, prot2_pos)
        this_pp_min = selec_pp.min(dim=1)[0]
        pp_min_martix.append(this_pp_min)
        if return_max:
            # for corrent .max() behaviour
            selec_pp[selec_pp > INF_DIST -1] = -INF_DIST
            pp_max_matrix.append(selec_pp.max(dim=1)[0])
    # output dimension should be [N_aa_prot1, N_aa_prot2]
    out = th.concat(pp_min_martix, dim=0)
    if return_max: out = (out, th.concat(pp_max_matrix, dim=0))
    return out

def pp_min_dist_naive(atomgroup):
    from prody import buildDistMatrix
    residues = [res for res in atomgroup.iterResidues()]
    min_dist = []
    for res1 in residues:
        this_min_dist = []
        for res2 in residues:
            this_matrix = buildDistMatrix(res1, res2)
            this_min_dist.append(np.min(this_matrix))
        min_dist.append(this_min_dist)
    return np.asarray(min_dist)

def pp_min_dist_oneway(atomgroup):
    from prody import buildDistMatrix
    residues = [res for res in atomgroup.iterResidues()]
    n_res = len(residues)
    min_dist = []
    for res1_id in range(n_res):
        this_min_dist = []
        for res2_id in range(n_res):
            # since it is a symetric matrix, we only calculate half of them
            # since we want to remove self-interaction, I set the diagnol elements to INF as well
            if res2_id <= res1_id:
                this_min_dist.append(INF_DIST)
                continue
            this_matrix = buildDistMatrix(residues[res1_id], residues[res2_id])
            this_min_dist.append(np.min(this_matrix))
        min_dist.append(this_min_dist)
    return np.asarray(min_dist)

def test_pp_min_dist():
    import time
    from geometry_processors.pl_dataset.ConfReader import PDBReader
    pdb_reader = PDBReader("/vast/sx801/geometries/PDBBind2020_OG/RenumPDBs/10gs.renum.pdb")
    protein_pad_dict = pdb_reader.get_padding_style_dict()

    # tik = time.time()
    # min_dist_naive = pp_min_dist_naive(pdb_reader.prody_parser)
    # tok = time.time()
    # t_naive = tok - tik
    # min_dist_naive = th.as_tensor(min_dist_naive)

    tik = time.time()
    min_dist_naive = pp_min_dist_matrix_vec_mem(protein_pad_dict["R"])
    tok = time.time()
    t_naive = tok - tik
    min_dist_naive = th.as_tensor(min_dist_naive)

    tik = time.time()
    min_dist = pp_min_dist_matrix_vec(protein_pad_dict["R"])
    tok = time.time()
    t_vec = tok - tik
    min_dist = min_dist.squeeze(0)

    diff = min_dist - min_dist_naive
    print(f"Diff: {diff}")
    print(f"Error: {diff.abs().sum()}")
    print(f"mem time: {t_naive}")
    print(f"Vec time: {t_vec}")

if __name__ == "__main__":
    seqs = pdb2chain_seqs("/scratch/sx801/temp/2ymd_protein.polar.pdb")
    print(len("".join(seqs)))
    seq_long = pdb2seq("/scratch/sx801/temp/2ymd_protein.polar.pdb")
    print(len(seq_long))
