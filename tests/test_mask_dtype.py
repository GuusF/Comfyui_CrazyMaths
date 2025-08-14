import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from math_alpha_mask import (
    MathAlphaMask,
    FractalNoiseAlphaMask,
    VoronoiAlphaMask,
    SierpinskiAlphaMask,
    HarmonographAlphaMask,
    AttractorAlphaMask,
    QuasicrystalAlphaMask,
    EquationAlphaMask,
)

def dummy_latent(h=16, w=16):
    return {"samples": torch.zeros(1, 4, h, w, dtype=torch.float16)}

import pytest

@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (MathAlphaMask, dict(frames=1, wave_function="sine", orientation="horizontal", frequency=1.0, phase=0.0, amplitude=1.0, bias=0.0, invert=False, radial=False)),
        (FractalNoiseAlphaMask, dict(frames=1, cells=2, octaves=1, persistence=0.5, lacunarity=2.0, speed=0.0, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
        (VoronoiAlphaMask, dict(frames=1, num_points=3, distance_power=2.0, jitter=0.0, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
        (SierpinskiAlphaMask, dict(frames=1, scale=2, shift_x=1.0, shift_y=1.0, invert=False)),
        (HarmonographAlphaMask, dict(frames=1, a1=1.0, a2=0.0, b1=1.0, b2=0.0, f1=2.0, f2=3.0, g1=2.0, g2=3.0, d1=0.0, d2=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0, phase_change=0.0, samples=500, line_width=0.0, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
        (AttractorAlphaMask, dict(frames=1, sigma=10.0, rho=28.0, beta=8.0/3.0, rho_change=0.0, step_size=0.01, num_steps=100, skip=10, x0=0.1, y0=0.0, z0=0.0, projection="xy", line_width=0.0, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
        (QuasicrystalAlphaMask, dict(frames=1, num_waves=5, frequency=5.0, phase_speed=0.0, golden_angle=True, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
        (EquationAlphaMask, dict(frames=1, expression="x", time_scale=1.0, contrast=1.0, amplitude=1.0, bias=0.0, invert=False)),
    ],
)
def test_mask_dtype(cls, kwargs):
    latent = dummy_latent()
    node = cls()
    mask, image = node.generate(latent, **kwargs)
    assert mask.dtype == torch.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask.shape[0] == kwargs["frames"]
    h, w = latent["samples"].shape[2:]
    assert mask.shape[1:] == (h, w)
