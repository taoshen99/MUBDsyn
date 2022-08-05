#!/usr/bin/env python
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs


def NLBScore(diverse_ligands_PS, final_decoys):
    df_Ligand = pd.read_csv(diverse_ligands_PS)
    df_Decoy = pd.read_csv(final_decoys)
    length_Ligand = len(df_Ligand)
    length_Decoy = len(df_Decoy)

    S_Ligand = list(df_Ligand['SMILES'])
    Suppl_Ligand = []
    for i in range(0, length_Ligand):
        m = Chem.MolFromSmiles(S_Ligand[i])
        Suppl_Ligand.append(m)

    S_Decoy = list(df_Decoy['SMILES'])
    Suppl_Decoy = []
    for i in range(0, length_Decoy):
        m = Chem.MolFromSmiles(S_Decoy[i])
        Suppl_Decoy.append(m)

    Fps_Ligand = [MACCSkeys.GenMACCSKeys(x) for x in Suppl_Ligand]
    Fps_Decoy = [MACCSkeys.GenMACCSKeys(x) for x in Suppl_Decoy]

    def sims_LL(t, r):
        sims = DataStructs.FingerprintSimilarity(Fps_Ligand[t], Fps_Ligand[r])
        return sims

    def sims_LD(t, r):
        sims = DataStructs.FingerprintSimilarity(Fps_Ligand[t], Fps_Decoy[r])
        return sims

    def Get_NLB(k):
        Sims_LD_list = []
        for i in range(0, length_Decoy):
            Sims_LD_list.append(sims_LD(k, i))

        Sims_LD_Max = max(Sims_LD_list[i] for i in range(0, len(Sims_LD_list)))

        Sims_LL_list = []
        for i in range(0, length_Ligand):
            if i != k:
                Sims_LL_list.append(sims_LL(k, i))

        count = 0
        for i in range(0, len(Sims_LL_list)):
            Sims_LL = Sims_LL_list[i]
            if Sims_LL > Sims_LD_Max:
                count = count + 1
        Pcount = float(count) / float(len(Sims_LL_list))

        return Pcount

    NLB_Sum = 0
    for k in range(0, length_Ligand):
        Pcount = Get_NLB(k)
        NLB_Sum = NLB_Sum + Pcount
    score = float(NLB_Sum) / float(length_Ligand)

    NLBScore = format(score, '.4f')

    print("NLBScore: ", NLBScore)
