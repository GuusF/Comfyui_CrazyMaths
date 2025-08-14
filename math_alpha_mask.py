"""
Custom ComfyUI node: MathAlphaMask
This node generates a batch of alpha masks using simple mathematical
functions. The masks can be animated by varying the phase along the
frame index and can be oriented horizontally, vertically or radially.

Inputs
------
latent : LATENT
    A latent image whose shape is used to determine the spatial
    resolution of the generated masks. Only the spatial dimensions of
    the latent are used; the actual latent values are ignored.
frames : INT
    The number of frames to generate in the mask batch. One mask is
    produced per frame.
wave_function : ENUM["sine", "triangle", "sawtooth", "square"]
    Which base waveform to use when computing the phase offset. The
    chosen function determines the offset that will be applied along
    the orientation axis (or radius) for each frame.
orientation : ENUM["horizontal", "vertical"]
    Whether to vary the phase along the x‑axis (horizontal) or y‑axis
    (vertical). If `radial` is True this setting is ignored.
frequency : FLOAT
    Controls how many cycles of the waveform appear across the
    orientation axis. Higher frequencies result in more stripes or
    ripples.
phase : FLOAT
    A constant phase offset (in radians) applied to every frame. This
    lets you shift the entire pattern left/right or up/down.
amplitude : FLOAT
    A multiplier applied to the waveform before clamping. Lower
    amplitudes reduce the contrast of the mask.
bias : FLOAT
    A constant added to the mask after scaling by `amplitude`.
    Useful for shifting the mask range up or down. The final mask is
    always clamped to [0, 1].
invert : BOOLEAN
    If True the mask values are inverted (`mask = 1 - mask`).
radial : BOOLEAN
    If True the pattern is computed radially from the centre rather
    than along a single axis. The `orientation` parameter is ignored
    when this is selected.

Returns
-------
MASK : torch.Tensor
    A 3D tensor of shape ``(frames, H_latent, W_latent)`` where
    ``H_latent`` and ``W_latent`` are the spatial dimensions of the
    supplied latent. Each slice along the first dimension is a mask
    corresponding to that frame. Mask values lie in the [0, 1] range.

To use this node in a workflow, connect the `latent` input to an
existing latent (for example the output of a `Repeat Latent Batch`
node). Choose the number of frames and other parameters to control
how the mask evolves over time. The resulting mask can be passed
directly into a `Latent Composite Masked` node to blend two sets of
latents together frame by frame.
"""

import math
from typing import Tuple

import torch.nn.functional as F

import torch


class MathAlphaMask:
    CATEGORY = "Crazy Math 🤯"


    """Generate animatable alpha masks using simple mathematical waveforms."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 16, "min": 1, "max": 4096}),
                "wave_function": (["sine", "triangle", "sawtooth", "square"], {"default": "sine"}),
                "orientation": (["horizontal", "vertical"], {"default": "horizontal"}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": math.tau, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "invert": ("BOOLEAN", {"default": False}),
                "radial": ("BOOLEAN", {"default": False}),
            },
        }

    # Return both the single‑channel mask and a 3‑channel image version.  Many
    # ComfyUI compositing nodes accept either a mask or an image; by
    # supplying both you can preview the pattern directly or use it as
    # a mask in a latent composite.
    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def generate(
        self,
        latent,
        frames: int,
        wave_function: str,
        orientation: str,
        frequency: float,
        phase: float,
        amplitude: float,
        bias: float,
        invert: bool,
        radial: bool,
    ) -> Tuple[torch.Tensor]:
        """Create a batch of masks based on the input parameters.

        Parameters
        ----------
        latent : dict
            A dictionary containing a ``samples`` tensor of shape
            (batch, channels, height, width). Only ``height`` and
            ``width`` are used to construct the masks.
        frames : int
            Number of mask frames to generate.
        wave_function : str
            Wave type to use when computing the phase offset for each
            frame. One of ``sine``, ``triangle``, ``sawtooth`` or
            ``square``.
        orientation : str
            Either ``horizontal`` or ``vertical``. Ignored when
            ``radial`` is True.
        frequency : float
            Controls the spatial frequency of the wave pattern across
            the orientation axis.
        phase : float
            Constant phase offset (radians) applied to every mask.
        amplitude : float
            Scale factor applied to the raw sinusoid before adding bias.
        bias : float
            Constant offset added after scaling the sinusoid.
        invert : bool
            If True, invert the mask values after clamping.
        radial : bool
            If True, the pattern is computed radially from the centre.

        Returns
        -------
        mask_batch : torch.Tensor
            A tensor of shape (frames, H, W) with values in the [0, 1]
            range. Each slice along the first dimension is an
            individual mask.
        """
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device

        # Generate coordinate grids in the range [0, 1].
        x_coords = torch.linspace(0.0, 1.0, w, device=device).unsqueeze(0).expand(h, -1)
        y_coords = torch.linspace(0.0, 1.0, h, device=device).unsqueeze(1).expand(-1, w)

        masks = []

        def base_wave(t: float) -> float:
            """Compute a base waveform value in the range [0, 1] for a given frame index.

            The returned value is used as a phase offset along the spatial axis.
            """
            # Normalized time across frames
            if frames > 1:
                tau = t / (frames - 1)
            else:
                tau = 0.0
            # Compute base value according to the selected wave_function
            if wave_function == "sine":
                v = 0.5 + 0.5 * math.sin(2.0 * math.pi * frequency * tau + phase)
            elif wave_function == "triangle":
                # Triangle wave between 0 and 1
                raw = frequency * tau + phase / (2.0 * math.pi)
                v = 2.0 * abs(2.0 * (raw - math.floor(raw + 0.5)))
            elif wave_function == "sawtooth":
                raw = frequency * tau + phase / (2.0 * math.pi)
                v = raw % 1.0
            elif wave_function == "square":
                v = 1.0 if math.sin(2.0 * math.pi * frequency * tau + phase) >= 0.0 else 0.0
            else:
                v = 0.0
            return max(0.0, min(1.0, v))

        for i in range(frames):
            offset = base_wave(i)
            if radial:
                # Radial pattern: compute distance from centre and apply phase offset along x only
                dx = x_coords - 0.5 + offset
                dy = y_coords - 0.5
                radius = torch.sqrt(dx ** 2 + dy ** 2)
                base = torch.sin(2.0 * math.pi * frequency * radius + phase)
                mask = 0.5 + 0.5 * base
            else:
                if orientation == "horizontal":
                    # Vary along the x‑axis. Add offset to x coordinate to create travelling pattern
                    pattern = x_coords + offset
                else:
                    # Vary along the y‑axis. Add offset to y coordinate to create travelling pattern
                    pattern = y_coords + offset
                base = torch.sin(2.0 * math.pi * frequency * pattern + phase)
                mask = 0.5 + 0.5 * base
            # Apply amplitude and bias then clamp to [0,1]
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)

        mask_tensor = torch.stack(masks, dim=0).to(samples)
        return (mask_tensor, mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1))


class FractalNoiseAlphaMask:
    CATEGORY = "Crazy Math 🤯"


    """Generate alpha masks using fractal Perlin value noise.

    This node synthesises smooth, organic textures by combining multiple
    layers (octaves) of two‑dimensional Perlin noise. Each successive
    octave doubles the spatial frequency (controlled by the lacunarity)
    and is scaled by a persistence factor to reduce its contribution,
    producing characteristic fractal patterns【829919289628262†L70-L79】.

    Parameters allow you to control the number of grid cells for the
    base octave (``cells``), the number of octaves, persistence and
    lacunarity values, how quickly the pattern drifts over time
    (``speed``), and the contrast, amplitude and bias applied to the
    final mask. Masks can also be inverted.  Increasing ``cells`` or
    ``octaves`` adds detail while increasing ``speed`` animates the
    pattern across frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 16, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "cells": ("INT", {"default": 4, "min": 1, "max": 64, "help": "Number of grid cells along each axis for the base octave. More cells create finer patterns."}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8, "help": "Number of noise layers to combine. Each octave doubles the frequency and scales the amplitude by the persistence."}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Relative strength of each successive octave. Lower values emphasise low frequencies."}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "help": "Multiplicative factor applied to the frequency for each successive octave."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Controls how fast the noise pattern drifts over frames."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the final mask to adjust contrast (values >1 increase contrast)."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask after normalisation."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    @staticmethod
    def _perlin2d(shape: Tuple[int, int], res: int, offset_x: float, offset_y: float, device) -> torch.Tensor:
        """
        Generate a single octave of 2D Perlin value noise.

        Parameters
        ----------
        shape : tuple of int
            Height and width of the noise map to generate.
        res : int
            Number of grid cells across each axis. The grid repeats
            periodically after ``res`` cells.
        offset_x : float
            Horizontal offset applied to the noise domain. Increasing this
            value scrolls the noise pattern along the x–axis.
        offset_y : float
            Vertical offset applied to the noise domain.
        device : torch.device
            Device on which to allocate tensors.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(h, w)`` containing values roughly in
            the ``[-1, 1]`` range.
        """
        h, w = shape
        # Create random gradient vectors on a (res x res) grid.  We use a
        # unit circle projection to ensure even distribution of directions.
        grads = torch.randn(res, res, 2, device=device)
        grads = grads / (torch.norm(grads, dim=-1, keepdim=True) + 1e-8)
        # Coordinate arrays.  The domain is [0, res] in both axes.  Offsets
        # allow the pattern to translate smoothly over time.
        ys = torch.linspace(0.0, res, h, device=device) + offset_y
        xs = torch.linspace(0.0, res, w, device=device) + offset_x
        # Compute integer grid coordinates and fractional components.  We
        # apply modulo to wrap around the periodic grid.
        y0 = torch.floor(ys).long() % res  # shape (h)
        x0 = torch.floor(xs).long() % res  # shape (w)
        dy = ys - y0.float()               # shape (h)
        dx = xs - x0.float()               # shape (w)
        y1 = (y0 + 1) % res
        x1 = (x0 + 1) % res
        # Expand to 2D by broadcasting.  g00, g10, g01, g11 have shape (h, w, 2)
        g00 = grads[y0.unsqueeze(1), x0.unsqueeze(0)]
        g10 = grads[y0.unsqueeze(1), x1.unsqueeze(0)]
        g01 = grads[y1.unsqueeze(1), x0.unsqueeze(0)]
        g11 = grads[y1.unsqueeze(1), x1.unsqueeze(0)]
        # Offsets for the four corners
        dx2d = dx.unsqueeze(0).repeat(h, 1)
        dy2d = dy.unsqueeze(1).repeat(1, w)
        # Dot products between gradient vectors and distance vectors
        dot00 = g00[..., 0] * dx2d + g00[..., 1] * dy2d
        dot10 = g10[..., 0] * (dx2d - 1.0) + g10[..., 1] * dy2d
        dot01 = g01[..., 0] * dx2d + g01[..., 1] * (dy2d - 1.0)
        dot11 = g11[..., 0] * (dx2d - 1.0) + g11[..., 1] * (dy2d - 1.0)
        # Fade curves for interpolation (6t^5 - 15t^4 + 10t^3)
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        fx = fade(dx2d)
        fy = fade(dy2d)
        # Interpolate along x then y
        nx0 = dot00 + fx * (dot10 - dot00)
        nx1 = dot01 + fx * (dot11 - dot01)
        nxy = nx0 + fy * (nx1 - nx0)
        return nxy

    def generate(
        self,
        latent,
        frames: int,
        cells: int,
        octaves: int,
        persistence: float,
        lacunarity: float,
        speed: float,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device
        masks = []
        # Normalise speed relative to number of frames to obtain smooth motion
        # If frames == 1 the offset remains zero.
        for f in range(frames):
            t = f / max(frames - 1, 1)
            offset = t * speed
            noise = torch.zeros(h, w, device=device)
            total_amp = 0.0
            frequency = 1.0
            amplitude_factor = 1.0
            # Combine octaves
            for o in range(octaves):
                res = max(1, int(cells * (lacunarity ** o)))
                # Use offset scaled by current frequency so that higher octaves
                # drift faster.
                octave_noise = self._perlin2d((h, w), res, offset * frequency, offset * frequency, device)
                noise = noise + amplitude_factor * octave_noise
                total_amp += amplitude_factor
                amplitude_factor *= persistence
                frequency *= lacunarity
            # Normalise to [-1,1] then convert to [0,1]
            noise = noise / (total_amp + 1e-8)
            mask = 0.5 + 0.5 * noise
            # Apply contrast
            if contrast != 1.0:
                # Avoid negative values when raising to fractional powers
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias then clamp
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(samples)
        # Convert mask to RGB image by repeating across three channels
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class VoronoiAlphaMask:
    CATEGORY = "Crazy Math"

    """Generate alpha masks from animated Voronoi diagrams.

    Voronoi diagrams divide the plane into regions based on proximity to
    a set of seed points.  Generative artists often use Voronoi patterns
    to approximate organic cellular textures and mosaic effects
    【669393097977570†L190-L239】.  This node generates a distance
    field from random seeds and animates the seeds over time by
    jittering their positions.  The distance field is normalised so
    that cell boundaries are bright and cell centres are dark (or
    inverted when ``invert`` is true).

    You can control the number of seeds, the distance metric (via
    ``distance_power``), the amount of jitter per frame and the
    contrast of the result.  Increasing the number of seeds creates
    smaller cells while larger jitter values produce more chaotic
    animations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 16, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "num_points": ("INT", {"default": 50, "min": 1, "max": 2048, "help": "Number of Voronoi seed points. More points give finer cells."}),
                "distance_power": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1, "help": "Exponent used in the Minkowski distance. 2.0 yields Euclidean distance; 1.0 is Manhattan distance."}),
                "jitter": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005, "help": "Random displacement applied to each seed per frame. Higher values create more movement."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the normalised distance field. Values >1 emphasise boundaries."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask after contrast adjustment."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def generate(
        self,
        latent,
        frames: int,
        num_points: int,
        distance_power: float,
        jitter: float,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device
        # Initialise seed positions uniformly in [0,1]^2 for frame 0
        seeds = torch.rand(num_points, 2, device=device)
        # Prepare coordinate grid
        x_coords = torch.linspace(0.0, 1.0, w, device=device)
        y_coords = torch.linspace(0.0, 1.0, h, device=device)
        xx = x_coords.unsqueeze(0).expand(h, -1)
        yy = y_coords.unsqueeze(1).expand(-1, w)
        masks = []
        # Precompute max distance for normalisation: the largest possible
        # Minkowski distance in the unit square.  For p >= 1 this is
        # 2**(1/p); for p < 1 use the same formula to avoid division by 0.
        max_dist = (2.0) ** (1.0 / max(distance_power, 1e-6))
        for f in range(frames):
            if f > 0 and jitter > 0.0:
                # Apply Gaussian jitter to each seed for animation
                seeds = seeds + torch.randn_like(seeds) * jitter
                seeds = torch.clamp(seeds, 0.0, 1.0)
            # Compute distance from every pixel to each seed
            # Expand shapes: seeds[:,0] shape (num_points), seeds[:,1]
            dx = xx.unsqueeze(2) - seeds[:, 0].unsqueeze(0).unsqueeze(0)  # shape (h,w,num_points)
            dy = yy.unsqueeze(2) - seeds[:, 1].unsqueeze(0).unsqueeze(0)
            # Minkowski distance
            dist = (torch.abs(dx) ** distance_power + torch.abs(dy) ** distance_power) ** (1.0 / distance_power)
            min_dist, _ = torch.min(dist, dim=2)
            # Normalise: distances near boundaries are small; we want
            # bright boundaries.  Invert by subtracting from 1.
            mask = min_dist / max_dist
            mask = 1.0 - mask
            # Apply contrast
            if contrast != 1.0:
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias then clamp
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class SierpinskiAlphaMask:
    CATEGORY = "Crazy Math 🤯"

    """Generate Sierpinski triangle fractal masks using bitwise AND.

    The classic Sierpinski triangle fractal appears naturally when
    computing the bitwise AND of two integers.  If the result of
    ``x & y`` is zero at a given coordinate then that pixel belongs to
    the white part of the fractal; otherwise it lies in the dark
    region【240453763846756†L28-L44】.  This node scales the integer
    coordinates by ``scale`` to control the size of the triangles and
    animates the pattern by translating the integer coordinates at a
    user‑specified rate.

    You can adjust the ``scale`` parameter to zoom in or out on the
    fractal and set ``shift_x`` and ``shift_y`` to scroll the pattern
    horizontally and vertically over time.  Masks can also be
    inverted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 16, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "scale": ("INT", {"default": 8, "min": 1, "max": 1024, "help": "Factor used to scale the integer coordinates. Larger values create finer fractal detail."}),
                "shift_x": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1, "help": "Number of integer units the pattern shifts along the x–axis per frame."}),
                "shift_y": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1, "help": "Number of integer units the pattern shifts along the y–axis per frame."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def generate(
        self,
        latent,
        frames: int,
        scale: int,
        shift_x: float,
        shift_y: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device
        # Precompute base integer coordinates
        base_x = torch.arange(w, device=device, dtype=torch.long)
        base_y = torch.arange(h, device=device, dtype=torch.long)
        # Scale coordinates to control fractal density
        base_x = base_x * scale
        base_y = base_y * scale
        masks = []
        for f in range(frames):
            offset_x = int(round(f * shift_x))
            offset_y = int(round(f * shift_y))
            # Compute translated integer coordinates
            xi = base_x + offset_x
            yi = base_y + offset_y
            # Compute bitwise AND matrix.  Use broadcasting: xi is (w,), yi is (h,).  We
            # want bitwise_and(y, x) for every pixel coordinate.  We'll
            # expand and swap axes to get shape (h,w).
            and_matrix = torch.bitwise_and(yi.unsqueeze(1), xi.unsqueeze(0))
            mask = (and_matrix == 0).float()
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class HarmonographAlphaMask:
    CATEGORY = "Crazy Math 🤯"

    """Generate alpha masks using harmonograph (parametric pendulum) curves.

    A harmonograph is a drawing device that uses multiple damped
    pendulums to trace complex curves on paper.  The resulting
    patterns depend on the amplitudes, frequencies, phases and damping
    factors of each pendulum【777053190775737†L260-L273】.  This node
    synthesises similar patterns by sampling a parametric curve and
    rasterising it onto a 2D grid.  You can adjust the amplitudes
    (``a1``, ``a2``, ``b1``, ``b2``), frequencies (``f1``, ``f2``,
    ``g1``, ``g2``), damping factors (``d1``, ``d2``) and phases
    (``p1``–``p4``) of the two pendulums.  The ``phase_change``
    parameter adds a constant offset to all phases each frame,
    producing animation.  The ``line_width`` controls how thick the
    traced line appears when rasterised.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 8, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "a1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Amplitude of the first term in the x equation."}),
                "a2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "help": "Amplitude of the second term in the x equation."}),
                "b1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Amplitude of the first term in the y equation."}),
                "b2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "help": "Amplitude of the second term in the y equation."}),
                "f1": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "help": "Frequency of the first term in the x equation."}),
                "f2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "help": "Frequency of the second term in the x equation."}),
                "g1": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "help": "Frequency of the first term in the y equation."}),
                "g2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "help": "Frequency of the second term in the y equation."}),
                "d1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Damping factor applied to the first terms in x and y. Values >0 cause the curve to spiral inwards."}),
                "d2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Damping factor applied to the second terms in x and y."}),
                "p1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": math.tau, "step": 0.01, "help": "Initial phase of the first term in the x equation."}),
                "p2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": math.tau, "step": 0.01, "help": "Initial phase of the second term in the x equation."}),
                "p3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": math.tau, "step": 0.01, "help": "Initial phase of the first term in the y equation."}),
                "p4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": math.tau, "step": 0.01, "help": "Initial phase of the second term in the y equation."}),
                "phase_change": ("FLOAT", {"default": 0.0, "min": -math.tau, "max": math.tau, "step": 0.01, "help": "Phase offset added to all phases each frame to animate the pattern."}),
                "samples": ("INT", {"default": 5000, "min": 100, "max": 50000, "help": "Number of sample points used to trace the curve. Higher values yield smoother lines."}),
                "line_width": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.005, "help": "Relative thickness of the drawn line. 0 produces thin lines; larger values blur the line to a thicker stroke."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the normalised density map. Values >1 emphasise dark/light regions."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask after normalisation."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def _gaussian_kernel(self, kernel_size: int, sigma: float, device) -> torch.Tensor:
        """Create a 2D Gaussian kernel of size ``kernel_size`` and standard deviation ``sigma``."""
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2 + 1e-8))
        kernel = kernel / kernel.sum()
        return kernel

    def generate(
        self,
        latent,
        frames: int,
        a1: float,
        a2: float,
        b1: float,
        b2: float,
        f1: float,
        f2: float,
        g1: float,
        g2: float,
        d1: float,
        d2: float,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        phase_change: float,
        samples: int,
        line_width: float,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_samples = latent["samples"]
        _, _, h, w = latent_samples.shape
        device = latent_samples.device
        masks = []
        # Precompute time vector
        t = torch.linspace(0.0, 1.0, samples, device=device)
        # Precompute exponential damping for first and second terms
        exp1 = torch.exp(-d1 * t)
        exp2 = torch.exp(-d2 * t)
        for f in range(frames):
            # Adjust phases per frame
            dp = f * phase_change
            p1_f = p1 + dp
            p2_f = p2 + dp
            p3_f = p3 + dp
            p4_f = p4 + dp
            # Compute parametric curve
            x = a1 * torch.sin(f1 * t + p1_f) * exp1 + a2 * torch.sin(f2 * t + p2_f) * exp2
            y = b1 * torch.sin(g1 * t + p3_f) * exp1 + b2 * torch.sin(g2 * t + p4_f) * exp2
            # Normalise to [0,1]
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
            y_norm = (y - y_min) / (y_max - y_min + 1e-8)
            x_idx = torch.clamp((x_norm * (w - 1)).round().long(), 0, w - 1)
            y_idx = torch.clamp((y_norm * (h - 1)).round().long(), 0, h - 1)
            # Rasterise by accumulating counts in a histogram
            indices = y_idx * w + x_idx
            counts = torch.bincount(indices, minlength=h * w).float()
            mask = counts.reshape(h, w)
            # Blur the strokes to control line width
            if line_width > 0.0:
                # Determine kernel size proportional to image size
                radius = max(int(line_width * max(h, w)), 1)
                kernel_size = 2 * radius + 1
                sigma = line_width * max(h, w)
                kernel = self._gaussian_kernel(kernel_size, sigma, device=device)
                mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=radius)[0, 0]
            # Normalise density map to [0,1]
            if mask.max() > 0:
                mask = mask / (mask.max() + 1e-8)
            # Apply contrast
            if contrast != 1.0:
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(latent_samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class AttractorAlphaMask:
    CATEGORY = "Crazy Math 🤯"

    """Generate alpha masks from strange attractors such as the Lorenz system.

    Strange attractors are dynamical systems that exhibit chaotic
    behaviour while remaining bounded.  The Lorenz attractor is a
    classic example, producing swirling patterns when projected to
    two dimensions【777053190775737†L234-L249】.  This node integrates the
    Lorenz system for a fixed number of steps at each frame and
    projects the trajectory onto one of the three coordinate planes
    (``xy``, ``xz`` or ``yz``).  You can vary the ``rho`` parameter
    linearly over frames using ``rho_change`` to animate the
    attractor, and adjust the integration step size and number of
    steps to control detail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 8, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "sigma": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1, "help": "Sigma parameter of the Lorenz system."}),
                "rho": ("FLOAT", {"default": 28.0, "min": 0.0, "max": 50.0, "step": 0.1, "help": "Rho parameter of the Lorenz system."}),
                "beta": ("FLOAT", {"default": 8.0 / 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "help": "Beta parameter of the Lorenz system."}),
                "rho_change": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "help": "Amount added to rho each frame to animate the attractor."}),
                "step_size": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.0001, "help": "Integration time step for Euler's method. Smaller values yield smoother curves but are slower."}),
                "num_steps": ("INT", {"default": 5000, "min": 100, "max": 20000, "help": "Number of integration steps per frame."}),
                "skip": ("INT", {"default": 100, "min": 0, "max": 10000, "help": "Number of initial integration steps to discard (transient)."}),
                "x0": ("FLOAT", {"default": 0.1, "min": -10.0, "max": 10.0, "step": 0.1, "help": "Initial x coordinate."}),
                "y0": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "help": "Initial y coordinate."}),
                "z0": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "help": "Initial z coordinate."}),
                "projection": (["xy", "xz", "yz"], {"default": "xy", "help": "Which two axes to project onto the plane to form the mask."}),
                "line_width": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.005, "help": "Relative thickness of the attractor lines. Larger values blur the trajectory."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the normalised density map. Values >1 emphasise structure."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def _gaussian_blur(self, mask: torch.Tensor, line_width: float, device) -> torch.Tensor:
        """Apply Gaussian blur to a 2D mask to thicken lines."""
        h, w = mask.shape
        if line_width <= 0.0:
            return mask
        radius = max(int(line_width * max(h, w)), 1)
        kernel_size = 2 * radius + 1
        sigma = line_width * max(h, w)
        ax = torch.arange(kernel_size, device=device) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2 + 1e-8))
        kernel = kernel / kernel.sum()
        blurred = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=radius)[0, 0]
        return blurred

    def generate(
        self,
        latent,
        frames: int,
        sigma: float,
        rho: float,
        beta: float,
        rho_change: float,
        step_size: float,
        num_steps: int,
        skip: int,
        x0: float,
        y0: float,
        z0: float,
        projection: str,
        line_width: float,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_samples = latent["samples"]
        _, _, h, w = latent_samples.shape
        device = latent_samples.device
        masks = []
        # Determine which axes to project onto
        proj_map = {
            "xy": (0, 1),
            "xz": (0, 2),
            "yz": (1, 2),
        }
        axis_i, axis_j = proj_map.get(projection, (0, 1))
        # Loop over frames
        for f in range(frames):
            rho_f = rho + f * rho_change
            # Initialise state variables
            x = x0
            y = y0
            z = z0
            # Accumulate samples for histogram after skipping transients
            xs = []
            ys = []
            # Integrate using simple Euler method
            for i in range(num_steps):
                dx = sigma * (y - x)
                dy = x * (rho_f - z) - y
                dz = x * y - beta * z
                x = x + dx * step_size
                y = y + dy * step_size
                z = z + dz * step_size
                if i >= skip:
                    if axis_i == 0:
                        xi = x
                    elif axis_i == 1:
                        xi = y
                    else:
                        xi = z
                    if axis_j == 0:
                        yi_ = x
                    elif axis_j == 1:
                        yi_ = y
                    else:
                        yi_ = z
                    xs.append(xi)
                    ys.append(yi_)
            # Convert lists to tensors
            xs_t = torch.tensor(xs, device=device)
            ys_t = torch.tensor(ys, device=device)
            # Normalise to [0,1]
            x_min, x_max = xs_t.min(), xs_t.max()
            y_min, y_max = ys_t.min(), ys_t.max()
            x_norm = (xs_t - x_min) / (x_max - x_min + 1e-8)
            y_norm = (ys_t - y_min) / (y_max - y_min + 1e-8)
            x_idx = torch.clamp((x_norm * (w - 1)).round().long(), 0, w - 1)
            y_idx = torch.clamp((y_norm * (h - 1)).round().long(), 0, h - 1)
            indices = y_idx * w + x_idx
            counts = torch.bincount(indices, minlength=h * w).float()
            mask = counts.reshape(h, w)
            # Apply blur for line width
            if line_width > 0.0:
                mask = self._gaussian_blur(mask, line_width, device)
            # Normalise
            if mask.max() > 0:
                mask = mask / (mask.max() + 1e-8)
            # Apply contrast
            if contrast != 1.0:
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(latent_samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class QuasicrystalAlphaMask:
    """Generate alpha masks using sums of rotated cosine waves.

    Quasicrystals and Penrose tilings exhibit local pentagonal symmetry
    and aperiodic structure related to the golden ratio【620589028546908†L440-L459】.
    Summing a set of evenly spaced cosine waves at different angles
    produces similar non‑periodic interference patterns.  This node
    allows you to choose the number of waves, their common frequency
    and whether to use golden‑ratio angles, then animates the pattern
    by advancing a phase offset each frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 8, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "num_waves": ("INT", {"default": 5, "min": 2, "max": 12, "help": "Number of cosine waves to sum. More waves produce more complex patterns."}),
                "frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1, "help": "Base spatial frequency of the waves. Higher values produce tighter interference fringes."}),
                "phase_speed": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.1, "help": "Amount by which the phase advances each frame to animate the pattern."}),
                "golden_angle": ("BOOLEAN", {"default": False, "help": "If true, use multiples of the golden angle (2π/φ) for the wave directions; otherwise use evenly spaced angles."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the normalised interference pattern to adjust contrast."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def generate(
        self,
        latent,
        frames: int,
        num_waves: int,
        frequency: float,
        phase_speed: float,
        golden_angle: bool,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device
        # Coordinate grid centred at 0
        x_coords = torch.linspace(-0.5, 0.5, w, device=device)
        y_coords = torch.linspace(-0.5, 0.5, h, device=device)
        xx = x_coords.unsqueeze(0).expand(h, -1)
        yy = y_coords.unsqueeze(1).expand(-1, w)
        # Precompute wave angles
        if golden_angle:
            # Golden angle is 2π/φ where φ ≈ 1.618
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            angle_inc = 2.0 * math.pi / phi
            angles = torch.tensor([((i * angle_inc) % (2.0 * math.pi)) for i in range(num_waves)], device=device)
        else:
            angles = torch.linspace(0.0, 2.0 * math.pi, num_waves, device=device, endpoint=False)
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)
        masks = []
        for f in range(frames):
            phase = f * phase_speed
            # Sum rotated cosine waves
            wave_sum = torch.zeros(h, w, device=device)
            for i in range(num_waves):
                # Compute dot product of position with wave direction
                proj = frequency * (xx * cos_t[i] + yy * sin_t[i])
                wave_sum = wave_sum + torch.cos(2.0 * math.pi * proj + phase)
            # Normalise wave_sum to [0,1]
            wave_sum = wave_sum / num_waves
            mask = 0.5 + 0.5 * wave_sum
            # Apply contrast
            if contrast != 1.0:
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)


class EquationAlphaMask:
    CATEGORY = "Crazy Math 🤯"

    """Evaluate user‑supplied mathematical expressions to generate masks.

    This node allows you to enter an arbitrary mathematical expression
    involving the variables ``x``, ``y`` and ``t`` and common
    mathematical functions (``sin``, ``cos``, ``sqrt``, etc.).  The
    expression is evaluated safely over the entire pixel grid and
    optionally animated by treating the frame index as a time
    variable ``t``.  After evaluation the result is normalised to
    [0,1], scaled by ``amplitude`` and offset by ``bias``.  You can
    adjust the contrast and invert the output as well.

    For example, an expression such as ``0.5 + 0.5*sin(10*(x**2 + y**2) + t)``
    produces concentric rings that drift over time.  Functions from
    the Python ``math`` and ``torch`` libraries are exposed under
    familiar names (e.g., ``sin``, ``cos``, ``abs``, ``exp``, ``log``,
    ``sqrt``, ``floor``, ``ceil``, ``pow``, ``min``, ``max`` and
    ``clamp``).  The variables ``x`` and ``y`` are normalised to
    ``[-1, 1]``; ``t`` varies linearly from 0 to ``time_scale`` over
    the frame range; and ``r`` provides the radial distance
    ``sqrt(x**2 + y**2)``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "frames": ("INT", {"default": 8, "min": 1, "max": 4096, "help": "Number of frames in the output batch."}),
                "expression": ("STRING", {"default": "0.5 + 0.5*sin(10*(x**2 + y**2) + t)", "multiline": True, "help": "Mathematical expression involving x, y and t. Use Python syntax (e.g. ** for exponent)."}),
                "time_scale": ("FLOAT", {"default": 2.0 * math.pi, "min": 0.0, "max": 100.0, "step": 0.1, "help": "Range of the t variable over the frame sequence. t runs from 0 to time_scale."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "help": "Exponent applied to the normalised result to adjust contrast."}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "help": "Scale factor applied to the mask after normalisation."}),
                "bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "help": "Constant added to the mask after scaling."}),
                "invert": ("BOOLEAN", {"default": False, "help": "Invert the final mask (1 - mask)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "generate"
    CATEGORY = "Mask"

    def _safe_eval(self, expr: str, vars_dict: dict) -> torch.Tensor:
        """Safely evaluate an expression using only allowed names and provided variables.

        The expression is compiled and then executed with a restricted
        global namespace.  Only functions defined in ``self.allowed``
        and variables provided in ``vars_dict`` are accessible.
        """
        code = compile(expr, "<string>", "eval")
        # Ensure all names in the expression are allowed
        for name in code.co_names:
            if name not in self.allowed and name not in vars_dict:
                raise NameError(f"Name '{name}' is not allowed in the expression.")
        return eval(code, {"__builtins__": {}}, {**self.allowed, **vars_dict})

    def __init__(self):
        # Define allowed functions and constants.  We map familiar
        # mathematical names to their torch equivalents where possible so
        # expressions can operate on tensors directly.  Functions that
        # operate on scalars and tensors alike are included.  We also
        # expose pi and e for convenience.
        import torch as _torch
        self.allowed = {
            # Trigonometric functions
            'sin': _torch.sin,
            'cos': _torch.cos,
            'tan': _torch.tan,
            'asin': _torch.asin,
            'acos': _torch.acos,
            'atan': _torch.atan,
            'atan2': _torch.atan2,
            # Hyperbolic functions
            'sinh': _torch.sinh,
            'cosh': _torch.cosh,
            'tanh': _torch.tanh,
            # Exponentials and logarithms
            'exp': _torch.exp,
            'log': _torch.log,
            # Powers and roots
            'sqrt': _torch.sqrt,
            'pow': _torch.pow,
            # Absolute value
            'abs': _torch.abs,
            # Rounding
            'floor': _torch.floor,
            'ceil': _torch.ceil,
            # Min/Max and clamping
            'min': _torch.minimum,
            'max': _torch.maximum,
            'clamp': lambda x, a, b: _torch.clamp(x, min=a, max=b),
            # Constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
        }

    def generate(
        self,
        latent,
        frames: int,
        expression: str,
        time_scale: float,
        contrast: float,
        amplitude: float,
        bias: float,
        invert: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = latent["samples"]
        _, _, h, w = samples.shape
        device = samples.device
        # Precompute coordinate grids in [-1,1]
        x = torch.linspace(-1.0, 1.0, w, device=device)
        y = torch.linspace(-1.0, 1.0, h, device=device)
        xx = x.unsqueeze(0).expand(h, -1)
        yy = y.unsqueeze(1).expand(-1, w)
        # Precompute radial coordinate
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        masks = []
        for f in range(frames):
            t_val = (f / max(frames - 1, 1)) * time_scale
            # Evaluate the expression safely with vectorised tensors
            vars_dict = {
                'x': xx,
                'y': yy,
                'r': rr,
                't': t_val,
            }
            try:
                result = self._safe_eval(expression, vars_dict)
            except Exception as e:
                # On error, fall back to zeros
                print(f"Error evaluating expression: {e}")
                result = torch.zeros_like(xx)
            # Convert result to float tensor and normalise
            # Flatten across all pixels to find min and max
            result = result.float()
            # If the expression yields NaN/Inf values, replace them with zeros
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            vmin = torch.min(result)
            vmax = torch.max(result)
            if vmax - vmin != 0:
                mask = (result - vmin) / (vmax - vmin)
            else:
                mask = torch.zeros_like(result)
            # Apply contrast
            if contrast != 1.0:
                mask = torch.clamp(mask, 0.0, 1.0) ** contrast
            # Scale and bias
            mask = amplitude * mask + bias
            mask = torch.clamp(mask, 0.0, 1.0)
            if invert:
                mask = 1.0 - mask
            masks.append(mask)
        mask_tensor = torch.stack(masks, dim=0).to(samples)
        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        return (mask_tensor, image_tensor)



# Register node definitions for ComfyUI
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

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathAlphaMask": "Math Alpha Mask",
    "FractalNoiseAlphaMask": "Fractal Noise Alpha Mask",
    "VoronoiAlphaMask": "Voronoi Alpha Mask",
    "SierpinskiAlphaMask": "Sierpinski Alpha Mask",
    "HarmonographAlphaMask": "Harmonograph Alpha Mask",
    "AttractorAlphaMask": "Attractor Alpha Mask",
    "QuasicrystalAlphaMask": "Quasicrystal Alpha Mask",
    "EquationAlphaMask": "Equation Alpha Mask",
}
