#!/usr/bin/bash
idx=0
len=$(cat output/ULS/Diverse_ligands_len.txt)
echo start molecular curating and clustering

while [ $idx -lt $len ]
do
    echo curating and clustering potential decoys \for ligand_$idx, $[$len-$idx-1] ligands left
    export idx
    python curating_clustering.py
    let "idx++"
    echo ligand_$[$idx-1] finished
done

echo molecular curating and clustering finished
echo merging all decoys

python merge_decoys.py

echo merging finished, refining finished