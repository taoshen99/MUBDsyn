from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

import pandas as pd


class Cal_Max():
    """ get max similarity between query decoy and all actives """
    def __init__(self, diverse_ligands_path):
        self.diverse_ligands_path = diverse_ligands_path

    def cal_max_sim(self, mol) -> float:
        df_1  = pd.read_csv(self.diverse_ligands_path)
        s1 = list(df_1['SMILES'])
        suppl_L = []
        for i in range(0,len(s1)):
            m = Chem.MolFromSmiles(s1[i])
            suppl_L.append(m)
        
        fps = MACCSkeys.GenMACCSKeys(mol) 
        fps_L = [MACCSkeys.GenMACCSKeys(x) for x in suppl_L]

        Fps=[]                  
        for j in range(0,len(suppl_L)):         
            x=DataStructs.FingerprintSimilarity(fps,fps_L[j])  
            Fps.append(x)
        len_Fps = len(Fps)
        Sim_Max=max(Fps[i] for i in range(0,len_Fps))

        return Sim_Max  

class Cal_Min(Cal_Max):
    """ get min similarity between query decoy and all actives """
    def __init__(self, diverse_ligands_path):
        super().__init__(diverse_ligands_path)

    def cal_min_sim(self, mol) -> float:
        df_1  = pd.read_csv(self.diverse_ligands_path)
        s1 = list(df_1['SMILES'])
        suppl_L = []
        for i in range(0,len(s1)):
            m = Chem.MolFromSmiles(s1[i])
            suppl_L.append(m)
        
        fps = MACCSkeys.GenMACCSKeys(mol) 
        fps_L = [MACCSkeys.GenMACCSKeys(x) for x in suppl_L]

        Fps=[]                  
        for j in range(0,len(suppl_L)):         
            x=DataStructs.FingerprintSimilarity(fps,fps_L[j])  
            Fps.append(x)
        len_Fps = len(Fps)
        Sim_Min=min(Fps[i] for i in range(0,len_Fps))

        return Sim_Min 
           