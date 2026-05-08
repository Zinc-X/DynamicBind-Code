import MDAnalysis as mda

tpr_file = "/ai/share/workspace/sxia/smart_logic_data/DynamicBinding_N2000_20251105/10gs/43976-PDB_Complex_10gs_NPT_10W_lxl/nowat.gro"
xtc_file = "/ai/share/workspace/sxia/smart_logic_data/DynamicBinding_N2000_20251105/10gs/43976-PDB_Complex_10gs_NPT_10W_lxl/nowat.xtc"

u = mda.Universe(tpr_file, xtc_file)
print(u)
breakpoint()
print("finished")
