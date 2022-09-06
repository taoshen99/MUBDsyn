#!/usr/bin/env python
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdmolops
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize

from dimorphite_dl import DimorphiteDL
import pandas as pd
import os
import argparse


class Curate():
    def __init__(self):
        self.lfc = rdMolStandardize.LargestFragmentChooser()
        self.charge = DimorphiteDL(
            min_ph=7.3,
            max_ph=7.5
        )

    def __call__(self, smi):
        mol = Chem.MolFromSmiles(smi)
        mol = rdMolStandardize.Cleanup(mol)
        mol = self.lfc.choose(mol)
        smi_charged = self.charge.protonate(Chem.MolToSmiles(mol))
        return smi_charged


def main(
        Ligand_data,
        Diverse_ligands,
        Diverse_ligands_PS,
        Diverse_ligands_PS_maxmin,
        Diverse_ligands_sims_maxmin,
        Diverse_ligands_len,
        **kwargs):
    
    ULS_dir = "output/ULS"
    
    try:
        os.makedirs(ULS_dir)
    except FileExistsError:
        pass
    
    filename, filetype = os.path.splitext(Ligand_data)
    if filetype == ".smi":
        with open(Ligand_data, 'r') as f:
            List_Smile = [smi.rstrip() for smi in f]
    elif filetype == ".csv":
        df_smi = pd.read_csv(Ligand_data)
        List_Smile = list(df_smi["SMILES"])
    else:
        raise Exception("supported input file format: .smi, .csv")

    smi_preprocessed = []

    if kwargs['curate']:
        preprocess = Curate()
        for i, smi in enumerate(List_Smile):
            if Chem.MolFromSmiles(smi):
                smi_list = preprocess(smi)
                smi_preprocessed.extend(smi_list)
            else:
                print("SMILES failed in MolFromSmiles")
                print(List_Smile[i])
                continue
    else:
        smi_preprocessed = List_Smile

    suppl = [Chem.MolFromSmiles(smi) for smi in smi_preprocessed]

    List_SMN = []

    for i in range(0, len(suppl)):
        MW = Descriptors.MolWt(suppl[i])
        NR = Lipinski.NumRotatableBonds(suppl[i])
        list_smn = []
        list_smn.append(smi_preprocessed[i])
        list_smn.append(MW)
        list_smn.append(NR)
        List_SMN.append(list_smn)

    df_SMN = pd.DataFrame(List_SMN, columns=['SMILES', 'MW', 'NR'])
    df_filter = df_SMN[(df_SMN['MW'] < 600) & (df_SMN['NR'] <= 20)]

    df_Smile = pd.DataFrame(list(df_filter['SMILES']), columns=['SMILES'])
    df_container = df_Smile.copy()

    Diverse_Ligands = []

    while True:
        df_start = df_container
        if len(df_start) != 0:
            list_start = list(df_start['SMILES'])
            Diverse_Ligands.append(list_start[0])

            mol = []
            for i in range(0, len(list_start)):
                m = Chem.MolFromSmiles(list_start[i])
                mol.append(m)

            fps = [MACCSkeys.GenMACCSKeys(x) for x in mol]
            list_ssf = []
            for j in range(0, len(list_start)):
                x = DataStructs.FingerprintSimilarity(fps[0], fps[j])
                list_fps = []
                list_fps.append(df_start.iloc[0, 0])
                list_fps.append(df_start.iloc[j, 0])
                list_fps.append(x)
                list_ssf.append(list_fps)
            df_SSF = pd.DataFrame(
                list_ssf, columns=[
                    'SMILES1', 'SMILES2', 'Fps'])

            df_filtered = df_SSF[df_SSF['Fps'] < 0.75]['SMILES2']
            df_final = pd.DataFrame(list(df_filtered), columns=['SMILES'])

            df_container = df_final

        else:
            break
    print('Total Number of Diverse Ligands:' + str(len(Diverse_Ligands)))

    df_diverseligands = pd.DataFrame(Diverse_Ligands, columns=['SMILES'])
    df_diverseligands.to_csv(os.path.join(ULS_dir, Diverse_ligands), index=None)

    list_MW = []
    list_NR = []
    list_HD = []
    list_HA = []
    list_FC = []
    list_LogP = []

    List_Smiles_diverseligands = list(df_diverseligands['SMILES'])
    suppl_diverseligands = []

    for i in range(0, len(List_Smiles_diverseligands)):
        m = Chem.MolFromSmiles(List_Smiles_diverseligands[i])
        suppl_diverseligands.append(m)

    for i in range(0, len(suppl_diverseligands)):
        MW = Descriptors.MolWt(suppl_diverseligands[i])
        NR = Lipinski.NumRotatableBonds(suppl_diverseligands[i])
        HD = Lipinski.NumHDonors(suppl_diverseligands[i])
        HA = Lipinski.NumHAcceptors(suppl_diverseligands[i])
        FC = rdmolops.GetFormalCharge(suppl_diverseligands[i])
        LogP = Crippen.MolLogP(suppl_diverseligands[i])

        list_MW.append(MW)
        list_NR.append(NR)
        list_HD.append(HD)
        list_HA.append(HA)
        list_FC.append(FC)
        list_LogP.append(LogP)

    MW_Max = max(list_MW[i] for i in range(0, len(list_MW)))
    MW_Min = min(list_MW[i] for i in range(0, len(list_MW)))
    NR_Max = max(list_NR[i] for i in range(0, len(list_NR)))
    NR_Min = min(list_NR[i] for i in range(0, len(list_NR)))
    HD_Max = max(list_HD[i] for i in range(0, len(list_HD)))
    HD_Min = min(list_HD[i] for i in range(0, len(list_HD)))
    HA_Max = max(list_HA[i] for i in range(0, len(list_HA)))
    HA_Min = min(list_HA[i] for i in range(0, len(list_HA)))
    FC_Max = max(list_FC[i] for i in range(0, len(list_FC)))
    FC_Min = min(list_FC[i] for i in range(0, len(list_FC)))
    LogP_Max = max(list_LogP[i] for i in range(0, len(list_LogP)))
    LogP_Min = min(list_LogP[i] for i in range(0, len(list_LogP)))

    a = [
        MW_Max,
        MW_Min,
        NR_Max,
        NR_Min,
        HD_Max,
        HD_Min,
        HA_Max,
        HA_Min,
        FC_Max,
        FC_Min,
        LogP_Max,
        LogP_Min]
    b = [
        'MW_Max',
        'MW_Min',
        'NR_Max',
        'NR_Min',
        'HD_Max',
        'HD_Min',
        'HA_Max',
        'HA_Min',
        'FC_Max',
        'FC_Min',
        'LogP_Max',
        'LogP_Min']

    df_PS_maxmin = pd.DataFrame(a, index=b, columns=['values'])
    df_PS_maxmin.to_csv(os.path.join(ULS_dir, Diverse_ligands_PS_maxmin))

    def Normalization(x, Max, Min):
        if Max == Min:
            x = 1
            return x
        else:
            x = float(x - Min) / float(Max - Min)
            return x

    MW_MCS = []
    NR_MCS = []
    HD_MCS = []
    HA_MCS = []
    FC_MCS = []
    LogP_MCS = []

    for i in range(0, len(List_Smiles_diverseligands)):
        mw = Normalization(list_MW[i], MW_Max, MW_Min)
        MW_MCS.append(mw)
        nr = Normalization(list_NR[i], NR_Max, NR_Min)
        NR_MCS.append(nr)
        hd = Normalization(list_HD[i], HD_Max, HD_Min)
        HD_MCS.append(hd)
        ha = Normalization(list_HA[i], HA_Max, HA_Min)
        HA_MCS.append(ha)
        fc = Normalization(list_FC[i], FC_Max, FC_Min)
        FC_MCS.append(fc)
        logp = Normalization(list_LogP[i], LogP_Max, LogP_Min)
        LogP_MCS.append(logp)

    df_diverseligands.insert(1, 'MW', list_MW)
    df_diverseligands.insert(2, 'NR', list_NR)
    df_diverseligands.insert(3, 'HD', list_HD)
    df_diverseligands.insert(4, 'HA', list_HA)
    df_diverseligands.insert(5, 'FC', list_FC)
    df_diverseligands.insert(6, 'LogP', list_LogP)
    df_diverseligands.insert(7, 'MW_MCS', MW_MCS)
    df_diverseligands.insert(8, 'NR_MCS', NR_MCS)
    df_diverseligands.insert(9, 'HD_MCS', HD_MCS)
    df_diverseligands.insert(10, 'HA_MCS', HA_MCS)
    df_diverseligands.insert(11, 'FC_MCS', FC_MCS)
    df_diverseligands.insert(12, 'LogP_MCS', LogP_MCS)
    df_Diverseligands_Ps = df_diverseligands[['SMILES',
                                              'MW',
                                              'MW_MCS',
                                              'NR',
                                              'NR_MCS',
                                              'HD',
                                              'HD_MCS',
                                              'HA',
                                              'HA_MCS',
                                              'FC',
                                              'FC_MCS',
                                              'LogP',
                                              'LogP_MCS']]
    df_Diverseligands_Ps.to_csv(os.path.join(ULS_dir, Diverse_ligands_PS), index=None)

    LL_Smis = []
    fps = [MACCSkeys.GenMACCSKeys(x) for x in suppl_diverseligands]
    for i in range(0, len(suppl_diverseligands)):
        for j in range(0, len(suppl_diverseligands)):
            if i != j:
                x = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                LL_Smis.append(x)

    LL_Sims_Max = max(LL_Smis[i] for i in range(0, len(LL_Smis)))
    LL_Sims_Min = min(LL_Smis[i] for i in range(0, len(LL_Smis)))

    with open(os.path.join(ULS_dir, Diverse_ligands_sims_maxmin), 'w') as fp:
        fp.write(str(LL_Sims_Max) + '\t')
        fp.write(str(LL_Sims_Min))

    df_DL = pd.read_csv(os.path.join(ULS_dir, Diverse_ligands))
    s_list = list(df_DL['SMILES'])
    len_s_list = len(s_list)
    with open(os.path.join(ULS_dir, Diverse_ligands_len), 'w') as fl:
        fl.write(str(len_s_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get unbiased ligand set')
    parser.add_argument('--i', help='input actives', default='raw_actives.smi')
    parser.add_argument(
        '--o1',
        help='output diverse ligands',
        default='Diverse_ligands.csv')
    parser.add_argument(
        '--o2',
        help='output diverse ligands with property',
        default='Diverse_ligands_PS.csv')
    parser.add_argument(
        '--o3',
        help='output max/min property',
        default='Diverse_ligands_PS_maxmin.csv')
    parser.add_argument(
        '--o4',
        help='output max/min similarity',
        default='Diverse_ligands_sims_maxmin.txt')
    parser.add_argument(
        '--o5',
        help='output number of diverse ligands',
        default='Diverse_ligands_len.txt')
    parser.add_argument('--curate',
                        action='store_true',
                        help='curate raw SMILES')
    args = parser.parse_args()
    main(args.i, args.o1, args.o2, args.o3, args.o4, args.o5, curate=args.curate)
