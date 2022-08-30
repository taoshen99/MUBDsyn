#!/usr/bin/env python
from validation.bms import BMSratio
from validation.nlbscore import NLBScore
from validation.pdc import compute_PDC, plot_PDC
from validation.roc import compute_ROC, plot_ROC
import matplotlib.pyplot as plt
import os

val_dir = "validation/results"
try:
    os.mkdir(val_dir)
except FileExistsError:
    pass

diverse_ligands_PS = "Diverse_ligands_PS.csv"
final_decoys = "Final_decoys.csv"

num_murcko, num_comp, ratio = BMSratio(final_decoys)
with open(os.path.join(val_dir, "BMSratio.txt"), "a") as f:
    f.write(f"number of unique Bemis-Murcko Atomic Frameworks: {num_murcko}\n")
    f.write(f"number of final decoys: {num_comp}\n")
    f.write(f"scaffold / decoy ratio: {ratio}\n")

nlb = NLBScore(diverse_ligands_PS, final_decoys)
with open(os.path.join(val_dir, "NLBScore.txt"), "w") as f:
    f.write(f"NLBScore: {nlb}")

compute_PDC(diverse_ligands_PS, final_decoys)
plot_PDC()
plt.savefig(os.path.join(val_dir,"Property Distribution Curve.png"))

compute_ROC(diverse_ligands_PS, final_decoys)
plot_ROC()
plt.savefig(os.path.join(val_dir,"ROC Curve.png"))