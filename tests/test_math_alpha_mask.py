import math
import os
import sys
import torch

# Ensure the module under test is importable when tests are executed
# from within the ``tests`` directory.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from math_alpha_mask import MathAlphaMask

def test_triangle_wave_correct():
    latent={'samples':torch.zeros(1,4,1,1)}
    node=MathAlphaMask()
    mask,_ = node.generate(
        latent,
        frames=5,
        wave_function='triangle',
        orientation='horizontal',
        frequency=0.5,
        phase=0.0,
        amplitude=1.0,
        bias=0.0,
        invert=False,
        radial=False,
    )
    assert mask.shape == (5,1,1)

    # Compute the expected waveform analytically using the corrected
    # triangle-wave formula implemented in the node.  This guards
    # against regressions that reintroduce the old flattened waveform.
    expected = []
    frames = 5
    frequency = 0.5
    phase = 0.0
    for i in range(frames):
        tau = i / (frames - 1) if frames > 1 else 0.0
        raw = frequency * tau + phase / (2.0 * math.pi)
        offset = 2.0 * abs(raw - math.floor(raw + 0.5))
        pattern = offset
        base = math.sin(2.0 * math.pi * frequency * pattern + phase)
        expected.append(0.5 + 0.5 * base)
    expected = torch.tensor(expected)

    assert torch.allclose(mask.view(-1), expected, atol=1e-6)
