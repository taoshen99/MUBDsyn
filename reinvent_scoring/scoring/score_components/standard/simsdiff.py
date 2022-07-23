from typing import List
import numpy as np

from reinvent_chemistry.simsdiff import Cal_Simsdiff

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

class Simsdiff(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        DL_path = self.parameters.specific_parameters.get(self.component_specific_parameters.DIVERSE_LIGANDS_PATH, "")
        ACTIVE_idx = self.parameters.specific_parameters.get(self.component_specific_parameters.ACTIVE_INDEX, "")
        self._calsimsdiff = Cal_Simsdiff(DL_path, ACTIVE_idx)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary: 
        score, raw_score = self._calculate_simsdiff(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_simsdiff(self, query_mols) -> np.array:
        simsdiffs = []
        for mol in query_mols:
            try:
                simsdiff = self._calsimsdiff.simsdiff_qr(mol)
            except ValueError:
                simsdiff = 0
            simsdiffs.append(simsdiff)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_simsdiffs = self._transformation_function(simsdiffs, transform_params)
        return np.array(transformed_simsdiffs, dtype=np.float32), np.array(simsdiffs, dtype=np.float32)
