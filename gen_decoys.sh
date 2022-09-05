#!/usr/bin/bash
idx=0
len=$(cat output/ULS/Diverse_ligands_len.txt)
while [ $idx -lt $len ]
do
    echo Generating decoys \for ligand_$idx, $[$len-$idx-1] ligands left
    export idx
    python mk_config.py
    python <path/to/REINVENT>/input.py output/UDS/auto_train/ligand_$idx/ligand_$idx.json
    let "idx++"
    echo ligand_$[$idx-1] finished
done