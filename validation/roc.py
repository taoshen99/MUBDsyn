import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import os


def compute_ROC(diverse_ligand_PS, final_decoys):

    df_Ligand = pd.read_csv(diverse_ligand_PS)
    df_Decoy = pd.read_csv(final_decoys)
    length_Ligand = len(df_Ligand)
    length_Decoy = len(df_Decoy)

    MW_MCS_Ligand = list(df_Ligand['MW_MCS'])
    NR_MCS_Ligand = list(df_Ligand['NR_MCS'])
    HD_MCS_Ligand = list(df_Ligand['HD_MCS'])
    HA_MCS_Ligand = list(df_Ligand['HA_MCS'])
    FC_MCS_Ligand = list(df_Ligand['FC_MCS'])
    LogP_MCS_Ligand = list(df_Ligand['LogP_MCS'])

    MW_MCS_Decoy = list(df_Decoy['MW_MCS'])
    NR_MCS_Decoy = list(df_Decoy['NR_MCS'])
    HD_MCS_Decoy = list(df_Decoy['HD_MCS'])
    HA_MCS_Decoy = list(df_Decoy['HA_MCS'])
    FC_MCS_Decoy = list(df_Decoy['FC_MCS'])
    LogP_MCS_Decoy = list(df_Decoy['LogP_MCS'])

    def Middle(PIT, PIR):
        x = PIT - PIR
        y = numpy.square(x)
        return(y)

    def simp_LL(t, r):
        simp1 = Middle(MW_MCS_Ligand[t], MW_MCS_Ligand[r])
        simp2 = Middle(NR_MCS_Ligand[t], NR_MCS_Ligand[r])
        simp3 = Middle(HD_MCS_Ligand[t], HD_MCS_Ligand[r])
        simp4 = Middle(HA_MCS_Ligand[t], HA_MCS_Ligand[r])
        simp5 = Middle(FC_MCS_Ligand[t], FC_MCS_Ligand[r])
        simp6 = Middle(LogP_MCS_Ligand[t], LogP_MCS_Ligand[r])
        simp = 1 - numpy.sqrt((simp1 + simp2 + simp3 +
                              simp4 + simp5 + simp6) / 6)
        return simp

    def simp_LD(t, r):
        simp1 = Middle(MW_MCS_Ligand[t], MW_MCS_Decoy[r])
        simp2 = Middle(NR_MCS_Ligand[t], NR_MCS_Decoy[r])
        simp3 = Middle(HD_MCS_Ligand[t], HD_MCS_Decoy[r])
        simp4 = Middle(HA_MCS_Ligand[t], HA_MCS_Decoy[r])
        simp5 = Middle(FC_MCS_Ligand[t], FC_MCS_Decoy[r])
        simp6 = Middle(LogP_MCS_Ligand[t], LogP_MCS_Decoy[r])
        simp = 1 - numpy.sqrt((simp1 + simp2 + simp3 +
                              simp4 + simp5 + simp6) / 6)
        return simp

    def Get_simp(k):
        Simp = []

        for i in range(0, length_Ligand):
            if i != k:
                Simp.append(1)
                Simp.append(simp_LL(k, i))

        for j in range(0, length_Decoy):
            if not ((39 * k <= j) & (j < 39 * (k + 1))):
                Simp.append(0)
                Simp.append(simp_LD(k, j))
        return Simp

    def Get_Roc_Arg_simp(k):
        Simp_list = Get_simp(k)
        data = numpy.array(Simp_list).reshape(-1, 2)
        df_data = pd.DataFrame(data, columns=['sort', 'simp'])
        df = df_data.sort_values(by="simp", ascending=False)
        sort = list(df['sort'])
        simp = list(df['simp'])

        fpr, tpr, thresholds = roc_curve(sort, simp)
        roc_auc = auc(fpr, tpr)

        return (fpr, tpr, roc_auc)

    Fpr_list_simp = []
    Tpr_list_simp = []
    Auc_list_simp = []

    for k in range(0, length_Ligand):
        Roc_Arg = Get_Roc_Arg_simp(k)
        Fpr_list_simp.append(Roc_Arg[0])
        Tpr_list_simp.append(Roc_Arg[1])
        Auc_list_simp.append(Roc_Arg[2])

    Auc_sum_simp = 0
    for i in range(0, length_Ligand):
        Auc = Auc_list_simp[i]
        Auc_sum_simp = Auc_sum_simp + Auc
    mean_aucs_simp = float(Auc_sum_simp) / float(length_Ligand)

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

    def Get_sims(k):
        Sims = []

        for i in range(0, length_Ligand):
            if i != k:
                Sims.append(1)
                Sims.append(sims_LL(k, i))
        for j in range(0, length_Decoy):
            if not ((39 * k <= j) & (j < 39 * (k + 1))):
                Sims.append(0)
                Sims.append(sims_LD(k, j))
        return Sims

    def Get_Roc_Arg_sims(k):
        Sims_list = Get_sims(k)
        data = numpy.array(Sims_list).reshape(-1, 2)
        df_data = pd.DataFrame(data, columns=['sort', 'sims'])
        df = df_data.sort_values(by="sims", ascending=False)
        sort = list(df['sort'])
        sims = list(df['sims'])

        fpr, tpr, thresholds = roc_curve(sort, sims)
        roc_auc = auc(fpr, tpr)

        return (fpr, tpr, roc_auc)

    Fpr_list_sims = []
    Tpr_list_sims = []
    Auc_list_sims = []

    for k in range(0, length_Ligand):
        Roc_Arg = Get_Roc_Arg_sims(k)
        Fpr_list_sims.append(Roc_Arg[0])
        Tpr_list_sims.append(Roc_Arg[1])
        Auc_list_sims.append(Roc_Arg[2])

    Auc_sum_sims = 0
    for i in range(0, length_Ligand):
        Auc = Auc_list_sims[i]
        Auc_sum_sims = Auc_sum_sims + Auc

    mean_aucs_sims = float(Auc_sum_sims) / float(length_Ligand)

    Roc_files_path = 'ROC_files'
    if not os.path.exists(Roc_files_path):
        os.makedirs(Roc_files_path)

    df_Fprp = pd.DataFrame(Fpr_list_simp)
    df_Tprp = pd.DataFrame(Tpr_list_simp)
    df_Fprs = pd.DataFrame(Fpr_list_sims)
    df_Tprs = pd.DataFrame(Tpr_list_sims)

    df_Fprp.to_csv('ROC_files/Fprp.csv', index=0)
    df_Tprp.to_csv('ROC_files/Tprp.csv', index=0)
    df_Fprs.to_csv('ROC_files/Fprs.csv', index=0)
    df_Tprs.to_csv('ROC_files/Tprs.csv', index=0)

    with open('ROC_files/mean_aucs.txt', 'w') as fp:
        fp.write(str(mean_aucs_simp) + ' ')
        fp.write(str(mean_aucs_sims))


def plot_ROC():
    df_Fprp = pd.read_csv('ROC_files/Fprp.csv')
    df_Tprp = pd.read_csv('ROC_files/Tprp.csv')
    df_Fprs = pd.read_csv('ROC_files/Fprs.csv')
    df_Tprs = pd.read_csv('ROC_files/Tprs.csv')
    mean_aucs = pd.read_csv('ROC_files/mean_aucs.txt', sep=' ', header=None)

    mean_aucs_simp = mean_aucs.iloc[0, 0]
    mean_aucs_sims = mean_aucs.iloc[0, 1]

    Fpr_list_simp = df_Fprp.values.tolist()
    Tpr_list_simp = df_Tprp.values.tolist()
    Fpr_list_sims = df_Fprs.values.tolist()
    Tpr_list_sims = df_Tprs.values.tolist()

    Curve_number = len(df_Fprp)
    plt.figure('Roc Curves', figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    lw = 2

    plt.sca(ax1)
    plt.text(0.5, 0.05, 'Mean(ROC AUCs) = %0.3f' % mean_aucs_simp)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(Simp)')
    for i in range(0, Curve_number):
        plt.plot(Fpr_list_simp[i], Tpr_list_simp[i], color=('red'), lw=lw)
    #plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], color=('k'), lw=lw, linestyle='--')

    plt.sca(ax2)
    plt.text(0.5, 0.05, 'Mean(ROC AUCs) = %0.3f' % mean_aucs_sims)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(Sims)')
    for i in range(0, Curve_number):
        plt.plot(
            Fpr_list_sims[i],
            Tpr_list_sims[i],
            color=('deepskyblue'),
            lw=lw)
    #plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], color=('k'), lw=lw, linestyle='--')

    plt.tight_layout()
