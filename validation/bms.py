#!/usr/bin/env python
import pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def BMSratio(final_decoys):
    df_1 = pd.read_csv(final_decoys)
    s_list = list(df_1['SMILES'])
    m_list = []

    for smi in s_list:
        m_list.append(MurckoScaffoldSmiles(smi))

    murcko = list(set(m_list))
    len_murcko = len(murcko)
    ratio = len_murcko / len(s_list)

    print("number of unique Bemis-Murcko Atomic Frameworks: ", len_murcko)
    print("number of final decoys: ", len(s_list))
    print("scaffold / decoy ratio:", round(ratio, 2))
