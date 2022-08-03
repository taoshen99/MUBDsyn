from typing import List
import numpy as np

from reinvent_chemistry.mubd import Cal_Max

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class MaxSim(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        DL_path = self.parameters.specific_parameters.get(self.component_specific_parameters.DIVERSE_LIGANDS_PATH, "")
        self._calmax = Cal_Max(DL_path)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary: 
        score, raw_score = self._calculate_maxsim(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_maxsim(self, query_mols) -> np.array:
        maxsims = []
        for mol in query_mols:
            try:
                maxsim = self._calmax.cal_max_sim(mol)
            except ValueError:
                maxsim = 0
            maxsims.append(maxsim)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_maxsims = self._transformation_function(maxsims, transform_params)
        return np.array(transformed_maxsims, dtype=np.float32), np.array(maxsims, dtype=np.float32)