#!/usr/bin/bash
source activate MUBD3.0
a=0
my_var=$(cat Diverse_ligands_len.txt)
echo start molecular clustering

while [ $a -lt $my_var ]
do
    echo clustering potential decoys \for ligand_$a, $[$my_var-$a-1] ligands left
    export a
    python agglomerative_clustering.py
    let "a++"
    echo ligand_$[$a-1] finished
done

echo molecular clustering finished
echo pooling all decoys

python pool_decoys.py

echo pooling finished, processing finished