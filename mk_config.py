#!/usr/bin/env python
import os
import json
import pandas as pd

#change path variables as user defined 
reinvent_dir = os.path.expanduser(
    "</path/to/REINVENT>")
reinvent_env = os.path.expanduser("~/anaconda3/envs/reinvent.v3.2")

idx = os.environ["idx"]
output_dir = os.path.expanduser("</path/to/MUBD3.0>/output/ligand_" + idx)
output_json_filename = "ligand_" + idx + ".json"
active_idx = "active_" + idx

MUBD_dir = os.path.expanduser("</path/to/MUBD3.0>")
diverse_ligands_path = os.path.join(MUBD_dir, "Diverse_ligands.csv")
diverse_ligands_ps_path = os.path.join(MUBD_dir, "Diverse_ligands_PS.csv")
diverse_ligands_ps_max_min_path = os.path.join(
    MUBD_dir, "Diverse_ligands_PS_maxmin.csv")
diverse_ligands_sim_max_min_path = os.path.join(
    MUBD_dir, "Diverse_ligands_sims_maxmin.txt")

df_PS_maxmin = pd.read_csv(diverse_ligands_ps_max_min_path, index_col=0)
MW_Ligand_Max = df_PS_maxmin.iloc[0, 0]
MW_Ligand_Min = df_PS_maxmin.iloc[1, 0]
NR_Ligand_Max = df_PS_maxmin.iloc[2, 0]
NR_Ligand_Min = df_PS_maxmin.iloc[3, 0]
HD_Ligand_Max = df_PS_maxmin.iloc[4, 0]
HD_Ligand_Min = df_PS_maxmin.iloc[5, 0]
HA_Ligand_Max = df_PS_maxmin.iloc[6, 0]
HA_Ligand_Min = df_PS_maxmin.iloc[7, 0]
FC_Ligand_Max = df_PS_maxmin.iloc[8, 0]
FC_Ligand_Min = df_PS_maxmin.iloc[9, 0]
LogP_Ligand_Max = df_PS_maxmin.iloc[10, 0]
LogP_Ligand_Min = df_PS_maxmin.iloc[11, 0]

df_sim_maxmin = pd.read_csv(
    diverse_ligands_sim_max_min_path,
    sep='\t',
    header=None)
sim_min = df_sim_maxmin.iloc[0, 1]

try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

configuration = {
    "version": 3,
    "run_type": "reinforcement_learning"
}

configuration["logging"] = {
    "sender": "http://0.0.0.1",
    "recipient": "local",
    "logging_frequency": 1000,
    "logging_path": os.path.join(output_dir, "progress.log"),
    "result_folder": os.path.join(output_dir, "results"),
    "job_name": "MUBD Demo",
    "job_id": "Demo"
}

configuration["parameters"] = {}

configuration["parameters"]["diversity_filter"] = {
    "name": "IdenticalMurckoScaffold",
    "nbmax": 5,
    "minscore": 0.9,
    "minsimilarity": 0.4
}

configuration["parameters"]["inception"] = {
    "smiles": [],
    "memory_size": 100,
    "sample_size": 10
}

configuration["parameters"]["reinforcement_learning"] = {
    "prior": os.path.join(MUBD_dir, "models/random.prior.new"),
    "agent": os.path.join(MUBD_dir, "models/random.prior.new"),
    "n_steps": 2000,
    "sigma": 128,
    "learning_rate": 0.0001,
    "batch_size": 128,
    "reset": 0,

    "reset_score_cutoff": 0.5,
    "margin_threshold": 50
}

scoring_function = {
    "name": "custom_sum",
    "parallel": True,

    "parameters": [
            {
                "component_type": "molecular_weight",
                "name": "mw",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": MW_Ligand_Max,
                        "low": MW_Ligand_Min
                    }
                },
                "weight": 1
            },

        {
                "component_type": "num_rotatable_bonds",
                "name": "rb",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": NR_Ligand_Max,
                        "low": NR_Ligand_Min
                    }
                },
                "weight": 1},

        {
                "component_type": "num_hbd_lipinski",
                "name": "hd",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": HD_Ligand_Max,
                        "low": HD_Ligand_Min
                    }
                },
                "weight": 1},

        {
                "component_type": "num_hba_lipinski",
                "name": "ha",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": HA_Ligand_Max,
                        "low": HA_Ligand_Min
                    }
                },
                "weight": 1},

        {
                "component_type": "formal_charge",
                "name": "fc",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": FC_Ligand_Max,
                        "low": FC_Ligand_Min
                    }
                },
                "weight": 1},

        {
                "component_type": "slogp",
                "name": "logp",
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "step",
                        "high": LogP_Ligand_Max,
                        "low": LogP_Ligand_Min
                    }
                },
                "weight": 1},

        {
                "component_type": "max_sim",
                "name": "max sim",
                "weight": 1,
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "left_step",
                        "low": 0.75,
                    },
                    "diverse_ligands_path": diverse_ligands_path
                }},

        {
                "component_type": "min_sim",
                "name": "min sim",
                "weight": 1,
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "right_step",
                        "low": sim_min,
                    },
                    "diverse_ligands_path": diverse_ligands_path
                }},

        {
                "component_type": "simsdiff",
                "name": "simsdiff",
                "weight": 1,
                "specific_parameters": {
                    "transformation": {
                        "transformation_type": "reverse_sigmoid",
                        "high": 0.4,
                        "low": -0.2,
                        "k": 0.8},
                    "diverse_ligands_path": diverse_ligands_path,
                    "active_index": active_idx
                }
                },

        {
                "component_type": "simp",
                "name": "simp",
                "specific_parameters": {
                    "diverse_ligands_ps_path": diverse_ligands_ps_path,
                    "diverse_ligands_ps_max_min_path": diverse_ligands_ps_max_min_path,
                    "active_index": active_idx
                },
                "weight": 1}
    ]}
configuration["parameters"]["scoring_function"] = scoring_function


configuration_JSON_path = os.path.join(output_dir, output_json_filename)
with open(configuration_JSON_path, 'w') as f:
    json.dump(configuration, f, indent=4)
