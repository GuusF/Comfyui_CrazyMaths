from .math_alpha_mask import (
    MathAlphaMask,
    FractalNoiseAlphaMask,
    VoronoiAlphaMask,
    SierpinskiAlphaMask,
    HarmonographAlphaMask,
    AttractorAlphaMask,
    QuasicrystalAlphaMask,
    EquationAlphaMask
)

NODE_CLASS_MAPPINGS = {
    "MathAlphaMask": MathAlphaMask,
    "FractalNoiseAlphaMask": FractalNoiseAlphaMask,
    "VoronoiAlphaMask": VoronoiAlphaMask,
    "SierpinskiAlphaMask": SierpinskiAlphaMask,
    "HarmonographAlphaMask": HarmonographAlphaMask,
    "AttractorAlphaMask": AttractorAlphaMask,
    "QuasicrystalAlphaMask": QuasicrystalAlphaMask,
    "EquationAlphaMask": EquationAlphaMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {␊
    "MathAlphaMask": "Crazy Math - Waveform Mask 🤯	",
    "FractalNoiseAlphaMask": "Crazy Math - Fractal Noise 🤯",
    "VoronoiAlphaMask": "Crazy Math - Voronoi 🤯",
    "SierpinskiAlphaMask": "Crazy Math - Sierpinski 🤯",
    "HarmonographAlphaMask": "Crazy Math - Harmonograph 🤯",
    "AttractorAlphaMask": "Crazy Math - Strange Attractor 🤯",
    "QuasicrystalAlphaMask": "Crazy Math - Quasicrystal 🤯",
    "EquationAlphaMask": "Crazy Math - Equation 🤯",
}

# Expose a web directory so ComfyUI loads the custom front-end assets
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
