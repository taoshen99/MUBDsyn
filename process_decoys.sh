#!/usr/bin/bash
idx=0
len=$(cat Diverse_ligands_len.txt)
echo start molecular clustering

while [ $idx -lt $len ]
do
    echo clustering potential decoys \for ligand_$idx, $[$len-$idx-1] ligands left
    export idx
    python agglomerative_clustering.py
    let "idx++"
    echo ligand_$[$idx-1] finished
done

echo molecular clustering finished
echo pooling all decoys

python pool_decoys.py

echo pooling finished, postprocessing finished