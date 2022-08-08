#!/usr/bin/bash
idx=0
len=$(cat Diverse_ligands_len.txt)
echo start molecular curing and clustering

while [ $idx -lt $len ]
do
    echo curing and clustering potential decoys \for ligand_$idx, $[$len-$idx-1] ligands left
    export idx
    python curing_clustering.py
    let "idx++"
    echo ligand_$[$idx-1] finished
done

echo molecular curing and clustering finished
echo pooling all decoys

python pool_decoys.py

echo pooling finished, refining finished