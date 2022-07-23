from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumRotatableBonds
from rdkit.Chem.rdmolops import GetFormalCharge

import pandas as pd
import numpy as np
import re


class Cal_Simsdiff():
    """ get simsdiff """
    def __init__(self, diverse_ligands_path, active_index):
        self.diverse_ligands_path = diverse_ligands_path
        self.query_idx = [int(s) for s in re.findall(r'\d+', active_index)][0]

    def simsdiff_qr(self, mol) -> float:
        df_DL = pd.read_csv(self.diverse_ligands_path)
        s_list = list(df_DL['SMILES'])
        suppl2 = [Chem.MolFromSmiles(smi) for smi in s_list]
        fps2 = [MACCSkeys.GenMACCSKeys(y) for y in suppl2]

        sum = 0
        for j in range(0, len(suppl2)):
            if self.query_idx != j:
                sim1 = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol), fps2[j])
                sim2 = DataStructs.FingerprintSimilarity(fps2[self.query_idx], fps2[j])
                sum += abs(sim1 - sim2)
        sims_mean = sum / (len(suppl2) - 1)
        return sims_mean