from perlin_noise import PerlinNoise
from typing import Tuple
from vertex import WorldParams
import numpy as np

def perlin(
    size: Tuple[int, int],
    levels: int,
    seed: int = 42,
    **kwargs
):
    xpix, ypix = size
    noise_arrs = [
        PerlinNoise(octaves=(x * 3 + 1), seed=seed + x)
        for x in range(levels)
    ]
    def apply_noise_map(x: int):
        j = x % ypix
        i = x // xpix
        noise_vals = [
            1 / (2*(idx + 1)) * noisefunc([i/xpix, j/ypix])
            for idx, noisefunc in enumerate(noise_arrs)
        ]
        return sum(noise_vals)

    return apply_noise_map


def range_field(params: WorldParams):
    return np.array([
        [
            i * params.width + j for j in range(params.width)
        ] for i in range(params.height)
    ])


NOISE_FUNCTIONS = {
    'perlin': perlin
}
