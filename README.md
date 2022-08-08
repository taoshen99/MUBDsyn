# MUBD-DecoyMaker 3.0: Making Maximal Unbiased Benchmarking Data Sets with Deep Reinforcement Learning

## Introduction

MUBD-DecoyMaker 3.0 is a brand-new computational software to make Maximal Unbiased Benchmarking Data Sets (MUBD) for in silico screening. Compared with our earlier two versions, i.e. MUBD-DECOYMAKER (Pipeline Pilot-based version, or MUBD-DecoyMaker 1.0) and MUBD-DecoyMaker 2.0, MUBD-DecoyMaker 3.0 has two noteworthy features:

1. Virtual molecules generated by recurrent neural netwrok (RNN)-based molecular generator with reinforcement learning (RL), instead of chemical library molecules, constitue the unbiased decoy set (UDS) component of MUBD. 

2. The criteria (or rule) for an ideal decoy previously defined in the earlier versions are integrated into a new scoring function for RL to fine-tune the generator.


Below is how to implement and run MUBD-DecoyMaker3.0.

![Figure from manuscript](figures/model.png)

## Requirements

As [REINVENT](https://github.com/MolecularAI/Reinvent) is used to make virtual decoys of MUBD 3.0, users are required to install this tool as instructed. The corresponding `conda` environment named `reinvent.v3.2` is created for virtual decoy generation. Please note we have modified the [PyPI](pypi.org) packages `reinvent_chemistry` and `reinvent_scoring` here in order to include our scoring functions specific for MUBD. Another `conda` environment named `MUBD3.0` is also created for preprocessing and postprocessing.

1) Install [REINVENT](https://github.com/MolecularAI/Reinvent).

2) Clone this repository and navigate to it:
```bash
$ git clone https://github.com/Sooooooap/MUBD3.0.git
$ cd MUBD3.0
```

3) Copy the modified packages `reinvent_chemistry` and `reinevnt_scoring` to replace the original ones:
```bash
$ conda activate reinvent.v3.2 
$ pip show reinvent_chemistry # Location: ~/anaconda3/envs/reinvent.v3.2/lib/python3.7/site-packages
$ cp -r reinvent_chemistry/ reinvent_scoring/ ~/anaconda3/envs/reinvent.v3.2/lib/python3.7/site-packages
```

4) Create the `conda` environment called `MUBD3.0`:
```bash
$ conda env create -f MUBD3.0.yml
```

## Usage

`ACM Agonists` is used as a test case to demonstrate how to build MUBD-ACM-AGO data set with MUBD-DecoyMaker 3.0. All the test files are included in the directory of `resources`. 

### Get unbiased ligand set (ULS)
Run `get_ligands.py` to process the raw ligand set. This script takes raw ligands in the representation of SMILES `raw_actives.smi` as input and outputs unbiased ligand set `Diverse_ligands.csv`. Another four property profiles `Diverse_ligands_PS.csv`, `Diverse_ligands_PS_maxmin.csv`, `Diverse_ligands_sims_maxmin.txt` and `Diverse_ligands_len.txt` are also recorded.

IMPORTANT: Ligand curation including standardizing molecule, stripping salts and charging at a specific range of pH (implemented by [Dimorphite-DL](https://github.com/Sulstice/dimorphite_dl)) is required if no curation is performed before. Please use `--cure` option of `get_ligands.py` to realize this (raw ligands in this test case have been cured before). Please use `--help` option to show all available options.
```bash
$ conda activate MUBD3.0
(MUBD3.0) $ python get_ligands.py
```

### Generate potential decoy set

`mk_config.py` writes out the configuration for MUBD3.0 virtual decoy generation. In order to automatically set up the configuration for each ligand and proceed to the next ligand, we provide `gen_decoys.sh`. Please replace the `</path/to/REINVENT>` and `</path/to/MUBD3.0>` in the scripts with user-defined directories.
```
$ mkdir output
$ chmod +x ./gen_decoys.sh
$ conda activate reinvent.v3.2
(reinvent.v3.2) $ ./gen_decoys.sh
```

### Get unbiased decoy set (UDS)
The file `output/ligand_$idx/results/scaffold_memory.csv` contains the potential decoy set for `ligand_$idx`. To get the unbiased decoy set `Final_decoys.csv`, potential decoys are refined by SMILES curation, structural clustering and pooling all decoys annotated with their property profiles. We provide `process_decoys.sh` which automatically runs `curing_clustering.py` and `pool_decoys.py` as the realization.
```
$ chmod +x ./process_decoys.sh
$ conda activate MUBD3.0
(MUBD3.0) $ ./process_decoys.sh
```

## Validation
Basically, The MUBD is validated and measured with four metrics. Please go through the notebook `basic_validation.ipynb` for more details.
```
$ conda activate MUBD3.0
(MUBD3.0) $ jupyter notebook
```
