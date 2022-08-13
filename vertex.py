from enum import Enum
import operator
from typing import Any, Dict, Union, List, Literal, Optional, Set, Tuple
import numpy as np
from yaml import safe_load

from pydantic import BaseModel

RANGE_t = Tuple[int, int]

class SoilType(str, Enum):
    ice = 'ice'
    grass = 'grass'
    water = 'water'


class SoilDescriptor(BaseModel):
    soil_type: SoilType
    percent: int

    def __hash__(self):
        return hash(self.soil_type.value)


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
    vector_field: Literal[
        'heightmap_m',
        'latitude',
        'longitude',
        'random'
    ] = 'heightmap'

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
        all_res = np.empty_like(field).fill(True)

        for x in self.conditions:
            op = vars(operator)[x.op]
            np.logical_and(
                op(field, x.value),
                all_res,
                out=all_res
            )

        return all_res

    def _pointwise_evaluate_any(self, field: np.ndarray):
        all_res = np.empty_like(field).fill(False)

        for x in self.conditions:
            op = vars(operator)[x.op]
            np.logical_or(
                op(field, x.value),
                all_res,
                out=all_res
            )

        return all_res

    def evaluate(
        self,
        field: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        satisfied_values = np.ones_like(
            field,
            dtype=int
        ) * self.satisfied_value if self.satisfied_value else field
        unsatisfied_values = np.ones_like(
            field,
            dtype=int
        ) * self.unsatisfied_value if self.unsatisfied_value else field

        # tuple of property name, weight, and the field we evaluated
        return np.where(
            self._pointwise_evaluate_all(field)
            if self.satisfaction_mode == 'all'
            else self._pointwise_evaluate_any(field),
            satisfied_values,
            unsatisfied_values
        )


class VectorField(BaseModel):
    name: str
    noise_type: Literal['perlin']
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

    @classmethod
    def load(cls):
        with open('./biomes.yaml', 'r', encoding='utf8') as f:
            yml = safe_load(f.read())
        return cls(**yml)
