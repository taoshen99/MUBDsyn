from typing import List
import numpy as np
import gol
from reinvent_chemistry.mubd import Cal_Simsdiff
from rdkit import DataStructs
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

class Simsdiff(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        ACTIVE_idx = self.parameters.specific_parameters.get(self.component_specific_parameters.ACTIVE_INDEX, "")
        self._calsimsdiff = Cal_Simsdiff(ACTIVE_idx)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary: 
        score, raw_score = self._calculate_simsdiff(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_simsdiff(self, query_mols) -> np.array:
        simsdiffs = []
        fps_l = gol.get_value("MACCSkeys")
        fps_q = gol.get_value("gen_MACCSkeys")
        query_idx = gol.get_value('query_idx')
        for fp in fps_q:
            sum = 0
            for j in range(0, len(fps_l)):
                if query_idx != j:
                    sim1 = DataStructs.FingerprintSimilarity(fp, fps_l[j])
                    sim2 = DataStructs.FingerprintSimilarity(fps_l[query_idx], fps_l[j])
                    sum += abs(sim1 - sim2)
            sims_mean = sum / (len(fps_l) - 1)
            simsdiffs.append(sims_mean)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_simsdiffs = self._transformation_function(simsdiffs, transform_params)
        return np.array(transformed_simsdiffs, dtype=np.float32), np.array(simsdiffs, dtype=np.float32)
