from perlin_noise import PerlinNoise
from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from vertex import WorldParams

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


def range_field(params: 'WorldParams'):
    return np.array([
        [
            i * params.width + j for j in range(params.width)
        ] for i in range(params.height)
    ])

def sin(
    size: Tuple[int, int],
    transpose: bool = False,
    amplitude: int = 90,
    frequency: float = 1,
    vertical_shift: int = 0,
    horizontal_shift: int = 0,
):
    xpix, ypix = size

    x = np.linspace(
        -2*np.pi,
        2*np.pi,
        ypix if not transpose else xpix
    )

    x = np.sin(
        (x.astype(np.float32) * frequency) - horizontal_shift
    ) * amplitude + vertical_shift

    result = np.zeros(size, dtype=np.float32)
    result[..., :] = np.expand_dims(
        x, 0
    ).repeat(xpix if not transpose else ypix, axis=0).T

    if transpose:
        result = result.T

    result = result.flatten().astype(np.int64)

    def apply(x: int):
        return result[x]

    return apply


NOISE_FUNCTIONS = {
    'perlin': perlin,
    'sin': sin
}
