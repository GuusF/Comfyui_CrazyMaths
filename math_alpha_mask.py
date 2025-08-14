diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -151,87 +151,95 @@ class MathAlphaMask:
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
-                v = 2.0 * abs(2.0 * (raw - math.floor(raw + 0.5)))
+                # The previous implementation multiplied by 2 twice,
+                # producing values in the range [0, 2] which were then
+                # clamped to [0, 1].  This resulted in a flattened top
+                # rather than a proper triangular waveform.  The correct
+                # formula scales the absolute deviation from the nearest
+                # half‑integer by two, yielding a smooth wave that ramps
+                # from 0 to 1 and back to 0 over each period.
+                v = 2.0 * abs(raw - math.floor(raw + 0.5))
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
 
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
-        return (mask_tensor, mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1))
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
+        return (mask_tensor, image_tensor)
 
 
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -350,53 +358,53 @@ class FractalNoiseAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
         # Convert mask to RGB image by repeating across three channels
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -449,52 +457,52 @@ class VoronoiAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -524,52 +532,52 @@ class SierpinskiAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -666,52 +674,52 @@ class HarmonographAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(latent_samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=latent_samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -825,52 +833,52 @@ class AttractorAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(latent_samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=latent_samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -915,52 +923,52 @@ class QuasicrystalAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
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
diff --git a/math_alpha_mask.py b/math_alpha_mask.py
index 014a90ea32ee740abe00504d0c8033ef569d0e4f..107c4f04d2fb4eed9c0dfc49cc5b28e8a8d6f571 100644
--- a/math_alpha_mask.py
+++ b/math_alpha_mask.py
@@ -1076,52 +1084,52 @@ class EquationAlphaMask:
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
-        mask_tensor = torch.stack(masks, dim=0).to(samples)
-        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
+        mask_tensor = torch.stack(masks, dim=0).to(device=device, dtype=torch.float32)
+        image_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(dtype=samples.dtype)
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
