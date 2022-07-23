from typing import List
import numpy as np

from reinvent_chemistry.simp import Cal_Simp

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

class Simp(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        DL_PS_path = self.parameters.specific_parameters.get(self.component_specific_parameters.DIVERSE_LIGANDS_PS_PATH, "")
        DL_PS_MAX_MIN_path = self.parameters.specific_parameters.get(self.component_specific_parameters.DIVERSE_LIGANDS_PS_MAX_MIN_PATH, "")
        ACTIVE_idx = self.parameters.specific_parameters.get(self.component_specific_parameters.ACTIVE_INDEX, "")
        self._calsimp = Cal_Simp(DL_PS_path, DL_PS_MAX_MIN_path, ACTIVE_idx)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary: 
        score, raw_score = self._calculate_simp(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_simp(self, query_mols) -> np.array:
        simps = []
        for mol in query_mols:
            try:
                simp = self._calsimp.simp_tr(mol)
            except ValueError:
                simp = 0
            simps.append(simp)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_simps = self._transformation_function(simps, transform_params)
        return np.array(transformed_simps, dtype=np.float32), np.array(simps, dtype=np.float32)
