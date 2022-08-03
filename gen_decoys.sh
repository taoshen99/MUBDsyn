#!/usr/bin/bash
source activate reinvent.v3.2
a=5
my_var=$(cat Diverse_ligands_len.txt)
while [ $a -lt $my_var ]
do
    echo Generating decoys \for ligand_$a, $[$my_var-$a-1] ligands left
    export a
    python mk_config.py
    python ~/project/reinvent/Reinvent/input.py output/ligand_$a/ligand_$a.json
    let "a++"
    echo ligand_$[$a-1] finished
done