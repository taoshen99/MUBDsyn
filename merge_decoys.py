#!/usr/bin/env python
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen

import pandas as pd
import os

def main(diverse_ligands_len_path, diverse_ligands_PS_maxmin_path, output):
    ULS_dir = "output/ULS"
    UDS_dir = "output/UDS"
    with open(os.path.join(ULS_dir, diverse_ligands_len_path), 'r') as f:
        ct = f.read()

    length = int(ct.rstrip())
    all_list = []

    for i in range(0, length):
        filename = os.path.join(UDS_dir, 'auto_train', 'ligand_' + str(i) + '/results/cluster_smi.csv')
        df = pd.read_csv(filename)
        s_list = list(df['SMILES'])
        all_list.extend(s_list[:39])

    length_Decoy = len(all_list)
    Suppl_Decoy = []
    for i in range(0, length_Decoy):
        m = Chem.MolFromSmiles(all_list[i])
        Suppl_Decoy.append(m)

    def Normalization(x, Max, Min):
        if Max == Min:
            x = 1
            return x
        else:
            x = float(x - Min) / float(Max - Min)
            return x

    df_PS_maxmin = pd.read_csv(os.path.join(ULS_dir, diverse_ligands_PS_maxmin_path), index_col=0)
    MW_Ligand_Max = df_PS_maxmin.iloc[0, 0]
    MW_Ligand_Min = df_PS_maxmin.iloc[1, 0]
    NR_Ligand_Max = df_PS_maxmin.iloc[2, 0]
    NR_Ligand_Min = df_PS_maxmin.iloc[3, 0]
    HD_Ligand_Max = df_PS_maxmin.iloc[4, 0]
    HD_Ligand_Min = df_PS_maxmin.iloc[5, 0]
    HA_Ligand_Max = df_PS_maxmin.iloc[6, 0]
    HA_Ligand_Min = df_PS_maxmin.iloc[7, 0]
    FC_Ligand_Max = df_PS_maxmin.iloc[8, 0]
    FC_Ligand_Min = df_PS_maxmin.iloc[9, 0]
    LogP_Ligand_Max = df_PS_maxmin.iloc[10, 0]
    LogP_Ligand_Min = df_PS_maxmin.iloc[11, 0]

    MW_list = []
    NR_list = []
    HD_list = []
    HA_list = []
    FC_list = []
    LogP_list = []

    for i in range(0, length_Decoy):
        MW = Descriptors.MolWt(Suppl_Decoy[i])
        NR = Lipinski.NumRotatableBonds(Suppl_Decoy[i])
        HD = Lipinski.NumHDonors(Suppl_Decoy[i])
        HA = Lipinski.NumHAcceptors(Suppl_Decoy[i])
        FC = rdmolops.GetFormalCharge(Suppl_Decoy[i])
        LogP = Crippen.MolLogP(Suppl_Decoy[i])
        MW_list.append(MW)
        NR_list.append(NR)
        HD_list.append(HD)
        HA_list.append(HA)
        FC_list.append(FC)
        LogP_list.append(LogP)

    MW_Decoy = MW_list
    LogP_Decoy = LogP_list
    HD_Decoy = HD_list
    HA_Decoy = HA_list
    NR_Decoy = NR_list
    FC_Decoy = FC_list

    MW_MCS_Decoy = []
    NR_MCS_Decoy = []
    HD_MCS_Decoy = []
    HA_MCS_Decoy = []
    FC_MCS_Decoy = []
    LogP_MCS_Decoy = []

    for i in range(0, length_Decoy):
        MW_MCS_Decoy.append(
            Normalization(
                MW_Decoy[i],
                MW_Ligand_Max,
                MW_Ligand_Min))
        NR_MCS_Decoy.append(
            Normalization(
                NR_Decoy[i],
                NR_Ligand_Max,
                NR_Ligand_Min))
        HD_MCS_Decoy.append(
            Normalization(
                HD_Decoy[i],
                HD_Ligand_Max,
                HD_Ligand_Min))
        HA_MCS_Decoy.append(
            Normalization(
                HA_Decoy[i],
                HA_Ligand_Max,
                HA_Ligand_Min))
        FC_MCS_Decoy.append(
            Normalization(
                FC_Decoy[i],
                FC_Ligand_Max,
                FC_Ligand_Min))
        LogP_MCS_Decoy.append(
            Normalization(
                LogP_Decoy[i],
                LogP_Ligand_Max,
                LogP_Ligand_Min))

    dic = {'SMILES': all_list,
           'MW': MW_list,
           'MW_MCS': MW_MCS_Decoy,
           'NR': NR_list,
           'NR_MCS': NR_MCS_Decoy,
           'HD': HD_list,
           'HD_MCS': HD_MCS_Decoy,
           'HA': HA_list,
           'HA_MCS': HA_MCS_Decoy,
           'FC': FC_list,
           'FC_MCS': FC_MCS_Decoy,
           'LogP': LogP_list,
           'LogP_MCS': LogP_MCS_Decoy
           }

    df_Decoy = pd.DataFrame(dic)

    df_Decoy.to_csv(os.path.join(UDS_dir, output), index=False)


if __name__ == "__main__":
    main(
        "Diverse_ligands_len.txt",
        "Diverse_ligands_PS_maxmin.csv",
        "Final_decoys.csv")
