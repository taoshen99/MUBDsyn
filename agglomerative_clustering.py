#!/usr/bin/env python
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import os
import warnings


def main(input_f, output_f):
    df_ = pd.read_csv(input_f)
    smi_list_raw = list(df_['SMILES'])
    score_list_raw = list(df_['total_score'])
    smi_list_canonical = []
    score_list_prep = []
    for i,smi in enumerate(smi_list_raw):
        if Chem.MolFromSmiles(smi) is not None:
            smi_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            if smi_canonical not in smi_list_canonical:
                smi_list_canonical.append(smi_canonical)
                score_list_prep.append(score_list_raw[i])

    smi_list_10000 = smi_list_canonical[:10000]
    score_list_10000 = score_list_prep[:10000]

    dict_ = dict(zip(smi_list_10000, score_list_10000))

    mols_ = []

    for smi in dict_.keys():
        m = Chem.MolFromSmiles(smi)
        mols_.append(m)

    n_mols_ = len(mols_)
    np.random.seed(13)
    np.random.shuffle(mols_)

    MACCS_fp = [MACCSkeys.GenMACCSKeys(x) for x in mols_]
    sim_matrix = np.array([DataStructs.BulkTanimotoSimilarity(MACCS_fp[i], MACCS_fp[:n_mols_],returnDistance=True) for i in range(n_mols_)])

    clustering = AgglomerativeClustering(n_clusters=39, linkage='ward', affinity='euclidean')
    clustering.fit(sim_matrix)

    clustering_library = {i: [] for i in range(39)}
    for n,j in enumerate(clustering.labels_):
        clustering_library[j].append(Chem.MolToSmiles(mols_[n]))

    tar_smi_l = []
    for smis in clustering_library.values():
        score_dict = {}
        for smi in smis:
            for smi_fromdict in dict_.keys():
                if smi == smi_fromdict:
                    score_dict[smi] = dict_[smi_fromdict]
        tar_smi = max(score_dict, key=score_dict.get)
        tar_smi_l.append(tar_smi)

    smi_fin = {'SMILES':tar_smi_l}
    df_fin = pd.DataFrame(smi_fin)
    df_fin.to_csv(output_f, index=None)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    s = os.environ["a"]
    input_ = "output/ligand_" + s + "/results/scaffold_memory.csv"
    output_ = "output/ligand_" + s + "/results/cluster_smi.csv"
    main(input_, output_)
