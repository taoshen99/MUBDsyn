#!/usr/bin/env python
#
# Copyright 2017 Atomwise Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import KFold

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

RDLogger.DisableLog('rdApp.*')

def readMols(file):
    fileName, fileExtension = os.path.splitext(file)
    mols = []
    if (fileExtension == ".smi"):
        f = open(file, 'r')
        l = f.readline()
        f.close()
        if '\t' in l:
            mols = Chem.SmilesMolSupplier(
                file, delimiter='\t', titleLine=False)
        else:
            mols = Chem.SmilesMolSupplier(file, delimiter=' ', titleLine=False)
    elif (fileExtension == ".sdf"):
        mols = Chem.SDMolSupplier(file)
    else:
        raise Exception("un-supported input file format: " +
                        fileExtension + " . ")
    smis_noiso = []
    for mol in mols:
        if mol:
            smis_noiso.append(Chem.MolToSmiles(mol, isomericSmiles=False))
    mols_ = [Chem.MolFromSmiles(smi) for smi in smis_noiso]
    return mols_

def get_fp(mols, fpType="MACCS"):
    fps = []
    if(fpType == 'ECFP4'):
        for x in mols:
            if(x):
                z = AllChem.GetMorganFingerprintAsBitVect(x, 2)
                fps.append(z)
    if(fpType == 'ECFP6'):
        for x in mols:
            if(x):
                z = AllChem.GetMorganFingerprintAsBitVect(x, 3)
                fps.append(z)
    if(fpType == 'ECFP12'):
        for x in mols:
            if(x):
                z = AllChem.GetMorganFingerprintAsBitVect(x, 6)
                fps.append(z)
    if(fpType == 'MACCS'):
        for x in mols:
            if(x):
                z = Chem.MACCSkeys.GenMACCSKeys(x)
                fps.append(z)
    if(fpType == 'Daylight'):
        for x in mols:
            if(x):
                z = FingerprintMols.FingerprintMol(x)
                fps.append(z)
    if (fpType == 'AP'):
        for x in mols:
            if (x):
                z = GetHashedAtomPairFingerprintAsBitVect(x, nBits=4096)
                fps.append(z)
    return fps

def gen_eval(train_data, train_labels, test_data, test_labels, method="knn1", metric="jaccard"):
    if method[:3] == "knn":
        k = int(method[3:])
        classifier = neighbors.KNeighborsClassifier(
            k, metric=metric, algorithm='brute')
        classifier.fit(train_data, train_labels)
        pred_labels = classifier.predict(test_data)
    elif method == "lr":
        classifier = LogisticRegression()
        classifier.fit(train_data, train_labels)
        pred_labels = classifier.predict(test_data)
    elif method == "rf":
        classifier = RandomForestClassifier(class_weight='balanced')
        classifier.fit(train_data, train_labels)
        pred_labels = classifier.predict(test_data)
    elif method == "svm":
        classifier = svm.SVC(class_weight='balanced')
        classifier.fit(train_data, train_labels)
        pred_labels = classifier.predict(test_data)

    mcc = matthews_corrcoef(test_labels, pred_labels)
    return mcc

def Cdist(params):
    return cdist(params[0], params[1], params[2])

def calcDistMat(fp1, fp2, distType="jaccard"):
        return cdist(fp1, fp2, distType)

def ave_val(fps, labels, target):
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    k = 0
    aaad, ddda, ave = [], [], []
    knn_metric, lr_metric, rf_metric, svm_metric = [], [], [], []
    for train_index, test_index in kf.split(fps):
        fps_train, fps_test = fps[train_index], fps[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        actives_fps_train, decoys_fps_train = [], []
        for i, label in enumerate(labels_train):
            if label == 1:
                actives_fps_train.append(fps_train[i])
            else:
                decoys_fps_train.append(fps_train[i])
        actives_fps_train, decoys_fps_train = np.array(actives_fps_train), np.array(decoys_fps_train)

        actives_fps_test, decoys_fps_test = [], []
        for i, label in enumerate(labels_test):
            if label == 1:
                actives_fps_test.append(fps_test[i])
            else:
                decoys_fps_test.append(fps_test[i])
        actives_fps_test, decoys_fps_test = np.array(actives_fps_test), np.array(decoys_fps_test)
    
        aTest_aTrain_D = calcDistMat(actives_fps_test, actives_fps_train)
        aTest_dTrain_D = calcDistMat(actives_fps_test, decoys_fps_train)
        dTest_dTrain_D = calcDistMat(decoys_fps_test, decoys_fps_train)
        dTest_aTrain_D = calcDistMat(decoys_fps_test, actives_fps_train)

        aTest_aTrain_S = np.mean(
            [np.mean(np.any(aTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
        aTest_dTrain_S = np.mean(
            [np.mean(np.any(aTest_dTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
        dTest_dTrain_S = np.mean(
            [np.mean(np.any(dTest_dTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
        dTest_aTrain_S = np.mean(
            [np.mean(np.any(dTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
    
        aaad.append((aTest_aTrain_S-aTest_dTrain_S))
        ddda.append((dTest_dTrain_S-dTest_aTrain_S))
        ave.append((aaad[k] + ddda[k]))

        knn_ = gen_eval(fps_train, labels_train, fps_test, labels_test, "knn1")
        knn_metric.append(knn_)
        lr_ = gen_eval(fps_train, labels_train, fps_test, labels_test, "lr")
        lr_metric.append(lr_)
        rf_ = gen_eval(fps_train, labels_train, fps_test, labels_test, "rf")
        rf_metric.append(rf_)
        svm_ = gen_eval(fps_train, labels_train, fps_test, labels_test, "svm")
        svm_metric.append(svm_)
    
        k += 1
    print(f"{target} done")
    return {"Target":target, "AA-AD":round(np.array(aaad).mean(), 2), "DD-DA":round(np.array(ddda).mean(), 2),
            "AVE bias":round(np.array(ave).mean(), 2), "KNN":round(np.array(knn_metric).mean(), 2), 
            "LR":round(np.array(lr_metric).mean(), 2), "RF":round(np.array(rf_metric).mean(), 2), 
            "SVM":round(np.array(svm_metric).mean(), 2)}

def get_path(cur_dir, dataset_tp="MUBDreal"):
    data_paths = []
    for it in os.listdir(os.path.join(cur_dir, "datasets_ext_val_ML_VS")):
        if os.path.exists(os.path.join("./datasets_ext_val_ML_VS/", it, "agonists")):
            active_path = os.path.join(cur_dir, "datasets_ext_val_ML_VS", it, "agonists", dataset_tp, "actives.smi")
            decoy_path = os.path.join(cur_dir, "datasets_ext_val_ML_VS", it, "agonists", dataset_tp, "decoys.smi")
            data_path = [it + "_agonists", active_path, decoy_path]
            data_paths.append(data_path)

        if os.path.exists(os.path.join("./datasets_ext_val_ML_VS/", it, "antagonists")):
            active_path = os.path.join(cur_dir, "datasets_ext_val_ML_VS", it, "antagonists", dataset_tp, "actives.smi")
            decoy_path = os.path.join(cur_dir, "datasets_ext_val_ML_VS", it, "antagonists", dataset_tp, "decoys.smi")
            data_path = [it + "_antagonists", active_path, decoy_path]
            data_paths.append(data_path)
    return data_paths

if __name__ == "__main__":
    pts_real = get_path(os.getcwd(), dataset_tp="MUBDreal")
    pts_syn = get_path(os.getcwd(), dataset_tp="MUBDsyn")

    rows_real = []
    for data_path in pts_real:
        actives = readMols(data_path[1])
        decoys = readMols(data_path[2])
        actives_fp = get_fp(actives)
        decoys_fp = get_fp(decoys)
        fps = np.concatenate((np.array(actives_fp), np.array(decoys_fp)), axis=0)
        labels = np.concatenate((np.ones(len(actives), dtype=int), np.zeros(len(decoys), dtype=int)), axis=0)

        rows_real.append(ave_val(fps, labels, data_path[0]))
    df_real = pd.DataFrame(rows_real)
    df_real.to_csv("MUBDreal_ML_validation.csv", index=None)
    
    rows_syn = []
    for data_path in pts_syn:
        actives = readMols(data_path[1])
        decoys = readMols(data_path[2])
        actives_fp = get_fp(actives)
        decoys_fp = get_fp(decoys)
        fps = np.concatenate((np.array(actives_fp), np.array(decoys_fp)), axis=0)
        labels = np.concatenate((np.ones(len(actives), dtype=int), np.zeros(len(decoys), dtype=int)), axis=0)

        rows_syn.append(ave_val(fps, labels, data_path[0]))
    df_syn = pd.DataFrame(rows_syn)
    df_syn.to_csv("MUBDsyn_ML_validation.csv", index=None)