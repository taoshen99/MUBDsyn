import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


def compute_PDC(diverse_ligand_PS, final_decoys):
    df_Ligand = pd.read_csv(diverse_ligand_PS)
    df_Decoy = pd.read_csv(final_decoys)

    def Draw_property(Pname, length):

        Ligand_P_Max = df_Ligand[Pname].max()
        Ligand_P_Min = df_Ligand[Pname].min()

        Decoy_P_Max = df_Decoy[Pname].max()
        Decoy_P_Min = df_Decoy[Pname].min()

        Ceil = max(int(Ligand_P_Max), int(Decoy_P_Max))
        Flo = min(int(Ligand_P_Min), int(Decoy_P_Min))

        Xstart = (int(Flo / length) - 1) * length
        Xend = (int(Ceil / length) + 2) * length

        total_count_Ligand = len(df_Ligand)
        P_Ligand_count = []
        for i in range(Xstart, Xend, length):
            count = len(df_Ligand[(df_Ligand[Pname] >= i)
                        & (df_Ligand[Pname] < (i + length))])
            pcount = float(count) / float(total_count_Ligand)
            P_Ligand_count.append(pcount)

        total_count_Decoy = len(df_Decoy)
        P_Decoy_count = []
        for i in range(Xstart, Xend, length):
            count = len(df_Decoy[(df_Decoy[Pname] >= i) &
                        (df_Decoy[Pname] < (i + length))])
            pcount = float(count) / float(total_count_Decoy)
            P_Decoy_count.append(pcount)

        X = [Xstart]
        for i in range(Xstart, Xend, length):
            X.append(i + float(length) / 2)
        X.append(Xend)

        P_Ligand_count[0:0] = [0]
        P_Decoy_count[0:0] = [0]

        P_Ligand_count.append(0)
        P_Decoy_count.append(0)

        y_Ligand = P_Ligand_count
        y_Decoy = P_Decoy_count
        x = X

        Draw = [y_Ligand, y_Decoy, x, [Xstart], [Xend]]
        return Draw

    LogP = Draw_property('LogP', 1)
    MW = Draw_property('MW', 100)
    NR = Draw_property('NR', 1)
    NHD = Draw_property('HD', 1)
    NHA = Draw_property('HA', 1)
    FC = Draw_property('FC', 1)

    PDC_files_path = 'PDC_files'
    if not os.path.exists(PDC_files_path):
        os.makedirs(PDC_files_path)

    with open('PDC_files/LogP.csv', 'w', newline='') as csv1:
        w = csv.writer(csv1)
        w.writerows(LogP)

    with open('PDC_files/MW.csv', 'w', newline='') as csv2:
        w = csv.writer(csv2)
        w.writerows(MW)

    with open('PDC_files/NR.csv', 'w', newline='') as csv3:
        w = csv.writer(csv3)
        w.writerows(NR)

    with open('PDC_files/NHD.csv', 'w', newline='') as csv4:
        w = csv.writer(csv4)
        w.writerows(NHD)

    with open('PDC_files/NHA.csv', 'w', newline='') as csv5:
        w = csv.writer(csv5)
        w.writerows(NHA)

    with open('PDC_files/FC.csv', 'w', newline='') as csv6:
        w = csv.writer(csv6)
        w.writerows(FC)


def plot_PDC():

    df_LogP = pd.read_csv('PDC_files/LogP.csv', header=None)
    LogP_y_Ligand = df_LogP.iloc[0]
    LogP_y_Decoy = df_LogP.iloc[1]
    LogP_x = df_LogP.iloc[2]
    LogP_Xstart = df_LogP.iloc[3, 0]
    LogP_Xend = df_LogP.iloc[4, 0]

    df_MW = pd.read_csv('PDC_files/MW.csv', header=None)
    MW_y_Ligand = df_MW.iloc[0]
    MW_y_Decoy = df_MW.iloc[1]
    MW_x = df_MW.iloc[2]
    MW_Xstart = df_MW.iloc[3, 0]
    MW_Xend = df_MW.iloc[4, 0]

    df_NR = pd.read_csv('PDC_files/NR.csv', header=None)
    NR_y_Ligand = df_NR.iloc[0]
    NR_y_Decoy = df_NR.iloc[1]
    NR_x = df_NR.iloc[2]
    NR_Xstart = df_NR.iloc[3, 0]
    NR_Xend = df_NR.iloc[4, 0]

    df_NHD = pd.read_csv('PDC_files/NHD.csv', header=None)
    NHD_y_Ligand = df_NHD.iloc[0]
    NHD_y_Decoy = df_NHD.iloc[1]
    NHD_x = df_NHD.iloc[2]
    NHD_Xstart = df_NHD.iloc[3, 0]
    NHD_Xend = df_NHD.iloc[4, 0]

    df_NHA = pd.read_csv('PDC_files/NHA.csv', header=None)
    NHA_y_Ligand = df_NHA.iloc[0]
    NHA_y_Decoy = df_NHA.iloc[1]
    NHA_x = df_NHA.iloc[2]
    NHA_Xstart = df_NHA.iloc[3, 0]
    NHA_Xend = df_NHA.iloc[4, 0]

    df_FC = pd.read_csv('PDC_files/FC.csv', header=None)
    FC_y_Ligand = df_FC.iloc[0]
    FC_y_Decoy = df_FC.iloc[1]
    FC_x = df_FC.iloc[2]
    FC_Xstart = df_FC.iloc[3, 0]
    FC_Xend = df_FC.iloc[4, 0]

    plt.figure('Property Distribution Curve', figsize=(12, 8))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)

    plt.sca(ax1)
    plt.xlabel('LogP')
    plt.ylabel('Fraction')
    plt.xlim(LogP_Xstart, LogP_Xend)
    plt.ylim(0, 1)
    plt.plot(LogP_x, LogP_y_Ligand, "-", label="LogP_Ligand")
    plt.plot(LogP_x, LogP_y_Decoy, "-", label="LogP_Decoy")
    plt.legend(fontsize=8, loc="upper left")

    plt.sca(ax2)
    plt.xlabel('Molecular_Weight')
    plt.ylabel('Fraction')
    plt.xlim(MW_Xstart, MW_Xend)
    plt.ylim(0, 1)
    plt.plot(MW_x, MW_y_Ligand, "-", label="Molecular_Weight_Ligand")
    plt.plot(MW_x, MW_y_Decoy, "-", label="Molecular_Weight_Decoy")
    plt.legend(fontsize=8, loc="upper left")

    plt.sca(ax3)
    plt.xlabel('Num_RotatableBonds')
    plt.ylabel('Fraction')
    plt.xlim(NR_Xstart, NR_Xend)
    plt.ylim(0, 1)
    plt.plot(NR_x, NR_y_Ligand, "-", label="Num_RotatableBonds_Ligand")
    plt.plot(NR_x, NR_y_Decoy, "-", label="Num_RotatableBonds_Decoy")
    plt.legend(fontsize=8, loc="upper left")

    plt.sca(ax4)
    plt.xlabel('Num_H_Donors')
    plt.ylabel('Fraction')
    plt.xlim(NHD_Xstart, NHD_Xend)
    plt.ylim(0, 1)
    plt.plot(NHD_x, NHD_y_Ligand, "-", label="Num_H_Donors_Ligand")
    plt.plot(NHD_x, NHD_y_Decoy, "-", label="Num_H_Donors_Decoy")
    plt.legend(fontsize=8, loc="upper left")

    plt.sca(ax5)
    plt.xlabel('Num_H_Acceptors')
    plt.ylabel('Fraction')
    plt.xlim(NHA_Xstart, NHA_Xend)
    plt.ylim(0, 1)
    plt.plot(NHA_x, NHA_y_Ligand, "-", label="Num_H_Acceptors_Ligand")
    plt.plot(NHA_x, NHA_y_Decoy, "-", label="Num_H_Acceptors_Decoy")
    plt.legend(fontsize=8, loc="upper left")

    plt.sca(ax6)
    plt.xlabel('FormalCharge')
    plt.ylabel('Fraction')
    plt.xlim(FC_Xstart, FC_Xend)
    plt.ylim(0, 1)
    plt.plot(FC_x, FC_y_Ligand, "-", label="FormalCharge_Ligand")
    plt.plot(FC_x, FC_y_Decoy, "-", label="FormalCharge_Decoy")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()

    for i in os.listdir("PDC_files"):
        f_d = os.path.join("PDC_files", i)
        os.remove(f_d)
    os.rmdir("PDC_files")