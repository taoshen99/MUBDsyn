from typing import List
import numpy as np

from reinvent_chemistry.Max_Min_Sim import Cal_Min

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

class MinSim(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        DL_path = self.parameters.specific_parameters.get(self.component_specific_parameters.DIVERSE_LIGANDS_PATH, "")
        self._calmin = Cal_Min(DL_path)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary: 
        score, raw_score = self._calculate_minsim(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_minsim(self, query_mols) -> np.array:
        minsims = []
        for mol in query_mols:
            try:
                minsim = self._calmin.cal_min_sim(mol)
            except ValueError:
                minsim = 0
            minsims.append(minsim)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_minsims = self._transformation_function(minsims, transform_params)
        return np.array(transformed_minsims, dtype=np.float32), np.array(minsims, dtype=np.float32)