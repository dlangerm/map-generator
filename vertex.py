from contextlib import AsyncExitStack
import logging
import operator
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, PrivateAttr
from yaml import safe_load
from tqdm import tqdm
from collections import defaultdict

from nosie_functions import NOISE_FUNCTIONS, range_field

RANGE_t = Tuple[int, int]
logger = logging.getLogger(__name__)

class SoilType(str, Enum):
    ice = 'ice'
    grass = 'grass'
    water = 'water'


class SoilDescriptor(BaseModel):
    soil_type: SoilType
    percent: int

    def __hash__(self):
        return hash(
            self.soil_type
        )


class Biome(BaseModel):
    name: str
    temperature_range_f: RANGE_t
    soil_types: Set[
        SoilDescriptor
    ]
    light_range_lux: RANGE_t
    rainfall_in: RANGE_t


class Point:
    temperature_f: int
    soil_type: SoilType
    light_lux: int
    rainfall_in: int


class PhysicalProperty:
    value: int
    name: str


class UniformDistribution(BaseModel):
    low: int
    high: int


class GaussianDistribution(BaseModel):
    mu: int
    sigma: int


class Criteria(BaseModel):
    op: str
    value: int


class PhysicalRule(BaseModel):
    # the name of the physical property to evaluate
    property_name: Literal[
        'temperature_f',
        'soil_type',
        'light_lux',
        'rainfall_in'
    ]

    # the field to use as input to the lte/gt/ne evaluation
    vector_field: str = 'heightmap'

    # condition of the vector field for the field
    conditions: List[Criteria]

    # all or any lte, gt, ne must be satisfied
    satisfaction_mode: Literal['any', 'all'] = 'all'

    # if multiple rules evaluate on the same block,
    # what weight should this result be over some other rule
    # two rules that have the same precedent at the same
    # will either be sorted by this value or evaluated
    # by a callable which returns the precedent
    weight: Optional[
        # Union[
        float
        # UniformDistribution, GaussianDistribution
        # ]
    ] = 1.0

    # what to replace that value with
    #  if the condition is satisfied
    satisfied_value: Optional[
        # Union[
        int
        # UniformDistribution,
        # GaussianDistribution
        # ]
    ]
    unsatisfied_value: Optional[
        # Union[
        int
        # UniformDistribution,
        # GaussianDistribution
        # ]
    ]

    def _pointwise_evaluate_all(self, field: np.ndarray):
        logger.info('eval all')
        all_res = np.empty_like(field)
        all_res.fill(True)

        for condition in self.conditions:
            op = vars(operator)[condition.op]
            np.logical_and(
                op(field, condition.value),
                all_res,
                out=all_res
            )

        return all_res

    def _pointwise_evaluate_any(self, field: np.ndarray):
        logger.info('eval any')
        all_res = np.empty_like(field)
        all_res.fill(False)
        logger.error(all_res)

        for condition in self.conditions:
            op = vars(operator)[condition.op]
            np.logical_or(
                op(field, condition.value),
                all_res,
                out=all_res
            )

        return all_res

    def evaluate(
        self,
        field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        satisfied_values = np.ones_like(
            field,
            dtype=int
        ) * self.satisfied_value if self.satisfied_value is not None \
            else np.empty_like(field)

        unsatisfied_values = np.ones_like(
            field,
            dtype=int
        ) * self.unsatisfied_value if self.unsatisfied_value is not None \
            else np.empty_like(field)

        # tuple of property name, weight, and the field we evaluated
        mask = self._pointwise_evaluate_all(field) \
            if self.satisfaction_mode == 'all' \
            else self._pointwise_evaluate_any(field)

        affected_cells = mask.copy()

        # mask should only cover where satisfied
        if self.unsatisfied_value is None \
            and self.satisfied_value is None:
            raise ValueError(
                "either an unsatisfied or satisfied value must be chosen"
            )
        elif self.satisfied_value is None:
            # only unsatisfied
            affected_cells = ~affected_cells
        elif self.unsatisfied_value is not None:
            # both are affected
            affected_cells.fill(True)

        return np.where(
            mask,
            satisfied_values,
            unsatisfied_values
        ), affected_cells


class VectorField(BaseModel):
    name: str
    noise_type: str
    noise_params: Dict[str, Any]
    norm_max: int
    norm_min: int


class WorldParams(BaseModel):
    width: int
    height: int


class BiomeCollection(BaseModel):
    biomes: List[Biome]
    rules: List[PhysicalRule]
    vector_fields: List[VectorField]
    world_params: WorldParams

    _vector_fields: Dict = PrivateAttr(None)
    _parameter_fields: Dict = PrivateAttr({})

    @classmethod
    def load(cls):
        with open('./biomes.yaml', 'r', encoding='utf8') as f:
            yml = safe_load(f.read())
        return cls(**yml)

    def generate_fields(self):
        if not self._vector_fields:
            logger.error("Generating base vector fields")
            base_field = range_field(self.world_params)
            self._vector_fields = {}
            for field in tqdm(self.vector_fields):
                func = NOISE_FUNCTIONS[field.noise_type]
                vector_eval = np.vectorize(func(**field.noise_params))
                noise_map = vector_eval(base_field.copy())

                # normalize
                nm_min = noise_map.min()
                nm_range = noise_map.max() - nm_min
                noise_map -= noise_map.min()
                norm = noise_map / nm_range
                noise_map = norm

                # map
                noise_map *= (field.norm_max - field.norm_min)
                noise_map += field.norm_min

                self._vector_fields[field.name] = noise_map
        return self._vector_fields

    def evaluate(self):
        resultant_fields = defaultdict(list)
        vector_fields = self.generate_fields()

        logger.error("Evaluating all rules")
        for rule in tqdm(self.rules):
            eval_field = vector_fields[
                rule.vector_field
            ]
            result_field, mask = rule.evaluate(eval_field)

            resultant_fields[rule.property_name].append({
                "rule": rule,
                "result": result_field,
                "mask": mask,
                "field": eval_field.copy()
            })

        logger.error("Generate weighted result maps")
        for key, field_array in tqdm(resultant_fields.items()):

            final_result = np.ma.zeros(
                (
                    len(field_array),
                    self.world_params.width,
                    self.world_params.height,
                )
            )
            final_weight_mask = np.ma.zeros(
                (
                    len(field_array),
                    self.world_params.width,
                    self.world_params.height,
                )
            )

            for i, result in enumerate(field_array):
                weight = result['rule'].weight
                mask = result['mask']
                result = result['result']

                final_result[i] = np.ma.array(
                    result * weight,
                    mask=mask
                )

                final_weight_mask[i] = np.ma.array(
                    np.ones_like(result) * weight,
                    mask=mask
                )

            weighted_average = final_result.mean(
                axis=0
            )
            weighted_average /= final_weight_mask.sum(
                axis=0
            )
            assert np.all(np.isfinite(weighted_average))

            self._parameter_fields[key] = weighted_average.data

        return self._parameter_fields
