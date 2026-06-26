import numpy as np
import scipy.ndimage


def recon_thread_is_cancelled(thread):
    """True when the UI has requested a cooperative stop for a recon QThread."""
    return bool(getattr(thread, "_cancel_requested", False))


def request_recon_thread_cancel(thread):
    if thread is not None:
        thread._cancel_requested = True


def add_3d_spherical_lesion(array, center, radius, value):
    """
    Adds a spherical lesion with a constant intensity value.
    """
    output_array = array.copy()
    x_dim, y_dim, z_dim = output_array.shape
    x_coords, y_coords, z_coords = np.ogrid[:x_dim, :y_dim, :z_dim]
    distance_sq = (x_coords - center[0])**2 + (y_coords - center[1])**2 + (z_coords - center[2])**2
    output_array[distance_sq <= radius**2] += value
    return output_array

def insert_real_lesion_array(patient_array, lesion_array, center_coords, intensity_scale=1.0):
    """
    Inserts a cropped 3D array (real lesion) into the patient array at specific coordinates.
    Non-zero lesion voxels are added (+=) into the target volume, matching spherical lesions.

    Args:
        patient_array (np.array): The main patient image (X, Y, Z).
        lesion_array (np.array): The small cropped lesion volume (L_x, L_y, L_z).
                                 Assumes 0 is background.
        center_coords (tuple): (x, y, z) target center in patient_array.
        intensity_scale (float): Multiplier for the lesion intensity.
        
    Returns:
        np.array: The modified patient array.
    """
    output_array = patient_array.copy()
    
    # 1. Get dimensions
    p_x, p_y, p_z = output_array.shape
    l_x, l_y, l_z = lesion_array.shape
    
    cx, cy, cz = center_coords
    
    # 2. Calculate start and end indices for the Patient Array
    # Center the lesion at the coordinates
    start_x = int(cx - l_x / 2)
    start_y = int(cy - l_y / 2)
    start_z = int(cz - l_z / 2)
    
    end_x = start_x + l_x
    end_y = start_y + l_y
    end_z = start_z + l_z
    
    # 3. Calculate cropping for the Lesion Array (Handling Boundary Conditions)
    # If the lesion goes out of the patient bounds (e.g. x < 0), we must crop the lesion array
    lesion_start_x, lesion_end_x = 0, l_x
    lesion_start_y, lesion_end_y = 0, l_y
    lesion_start_z, lesion_end_z = 0, l_z
    
    # Check Lower Bounds ( < 0 )
    if start_x < 0:
        lesion_start_x = -start_x
        start_x = 0
    if start_y < 0:
        lesion_start_y = -start_y
        start_y = 0
    if start_z < 0:
        lesion_start_z = -start_z
        start_z = 0
        
    # Check Upper Bounds ( > dimension )
    if end_x > p_x:
        diff = end_x - p_x
        end_x = p_x
        lesion_end_x -= diff
    if end_y > p_y:
        diff = end_y - p_y
        end_y = p_y
        lesion_end_y -= diff
    if end_z > p_z:
        diff = end_z - p_z
        end_z = p_z
        lesion_end_z -= diff
        
    # 4. Safety Check: If completely out of bounds, return original
    if (start_x >= end_x) or (start_y >= end_y) or (start_z >= end_z):
        return output_array

    # 5. Extract regions
    roi_patient = output_array[start_x:end_x, start_y:end_y, start_z:end_z]
    roi_lesion = lesion_array[lesion_start_x:lesion_end_x, lesion_start_y:lesion_end_y, lesion_start_z:lesion_end_z]
    
    # 6. Add the lesion into the patient ROI (same += rule as spherical lesions).
    mask = roi_lesion > 0
    roi_patient[mask] += roi_lesion[mask] * intensity_scale
    
    output_array[start_x:end_x, start_y:end_y, start_z:end_z] = roi_patient
    
    return output_array


def _align_bool_volume_to_shape(support, target_shape):
    """Crop or zero-pad a boolean volume to match target_shape."""
    support = np.asarray(support, dtype=bool)
    if support.shape == target_shape:
        return support
    out = np.zeros(target_shape, dtype=bool)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(support.shape, target_shape))
    out[slices] = support[slices]
    return out


def library_morphology_support_from_shape(custom_shape, size_scale=1.0):
    """
    Binary lesion support after size_scale resize.

    custom_shape must be the unscaled library template. size_scale is applied here
    (and in resized_library_template) — do not store a pre-scaled array in custom_shape.

    Morphology uses nearest-neighbor on the source mask (custom_shape > 0) so
    downscaling does not add cubic-interpolation fringe voxels. Intensity
    values are resized separately with cubic interpolation and clipped to this
    support.
    """
    orig = np.asarray(custom_shape, dtype=np.float32)
    binary = orig > 0
    if abs(float(size_scale) - 1.0) < 1e-6:
        return binary

    cubic = resize_volume_preserve_morphology(orig, size_scale)
    factors = tuple(t / s for t, s in zip(cubic.shape, binary.shape))
    support = scipy.ndimage.zoom(binary.astype(np.float32), factors, order=0) >= 0.5
    return _align_bool_volume_to_shape(support, cubic.shape)


def library_morphology_support(lesion):
    """Morphology support for a library lesion after size_scale."""
    return library_morphology_support_from_shape(
        lesion.custom_shape, getattr(lesion, "size_scale", 1.0)
    )


def resized_library_template(lesion):
    """Library lesion intensities after cubic size_scale resize."""
    scale = getattr(lesion, "size_scale", 1.0)
    return resize_volume_preserve_morphology(lesion.custom_shape, scale)


def library_intensity_patch(lesion):
    """Cubic-resized template intensities clipped to NN morphology support."""
    resized = resized_library_template(lesion)
    support = library_morphology_support(lesion)
    return np.where(support, resized, 0.0).astype(np.float32)


def uniform_library_patch_from_resized(resized, intensity, support=None):
    """
    After cubic resize, set every morphology-supported voxel to one fixed
    intensity. Pass support explicitly (NN-resized binary mask); do not use
    resized > 0, which includes cubic fringe voxels when downscaling.
    """
    resized = np.asarray(resized, dtype=np.float32)
    if support is None:
        support = resized > 0
    else:
        support = _align_bool_volume_to_shape(support, resized.shape)
    return np.where(support, float(intensity), 0.0).astype(np.float32)


def uniform_library_patch(lesion, intensity):
    """Uniform-intensity patch on NN morphology support."""
    support = library_morphology_support(lesion)
    return uniform_library_patch_from_resized(
        resized_library_template(lesion), intensity, support=support
    )


def apply_library_lesion_to_volume(volume, lesion, center_coords=None):
    """
    Place a library lesion on a volume.
    - Template mode: cubic resize for intensities, clipped to NN morphology.
    - Uniform mode: same morphology support, all supported voxels set to the
      fixed intensity (resize intensities first, then flatten within support).
    """
    intensity = float(getattr(lesion, "intensity", 0.0))
    center = lesion.center if center_coords is None else center_coords
    if getattr(lesion, "uniform_intensity", False):
        patch = uniform_library_patch(lesion, intensity)
    else:
        patch = library_intensity_patch(lesion) * intensity
    return insert_real_lesion_array(volume, patch, center, 1.0)


def build_lesion_mask_on_grid(shape, lesion):
    """
    Binary lesion mask on the analysis/reconstruction grid after size_scale
    resize and center placement (same geometry as insertion).
    """
    mask_base = np.zeros(shape, dtype=np.float32)
    if getattr(lesion, "custom_shape", None) is not None:
        support = library_morphology_support(lesion).astype(np.float32)
        mask_base = insert_real_lesion_array(mask_base, support, lesion.center, 1.0)
    else:
        scale = getattr(lesion, "size_scale", 1.0)
        mask_base = add_3d_spherical_lesion(mask_base, lesion.center, lesion.radius * scale, 1.0)
    return mask_base > 0.0


def build_lesion_truth_on_grid(shape, lesion):
    """
    Inserted-truth activity map on the analysis/reconstruction grid.
    """
    vol = np.zeros(shape, dtype=np.float32)
    if getattr(lesion, "custom_shape", None) is not None:
        return apply_library_lesion_to_volume(vol, lesion)
    scale = getattr(lesion, "size_scale", 1.0)
    intensity_scale = float(getattr(lesion, "intensity", 0.0))
    effective_radius = lesion.radius * scale
    vol = add_3d_spherical_lesion(vol, lesion.center, effective_radius, intensity_scale)
    return vol


def lesion_volume_ml_from_mask(mask, spacing_mm):
    """Physical lesion volume (mL) from a grid-aligned binary mask."""
    if mask is None or not np.any(mask):
        return 0.0
    spacing = tuple(float(s) for s in spacing_mm[:3])
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    return float(np.count_nonzero(mask) * voxel_vol_mm3 / 1000.0)


def sphere_radius_mm(lesion, spacing_mm):
    """Effective sphere radius in mm (uses x spacing, matching radius_px storage)."""
    scale = float(getattr(lesion, "size_scale", 1.0))
    return float(lesion.radius) * scale * float(spacing_mm[0])


def sphere_volume_ml_analytic(lesion, spacing_mm):
    """Continuous (4/3)πr³ sphere volume; for reference only — use lesion_volume_ml_on_grid."""
    radius_mm = sphere_radius_mm(lesion, spacing_mm)
    return float((4.0 / 3.0) * np.pi * (radius_mm ** 3) / 1000.0)


def preview_sphere_volume_ml_on_grid(radius_mm, size_scale, shape, spacing_mm, center=None):
    """
    Grid-aligned sphere volume for UI preview (voxel count × voxel size).
    Uses the same geometry as build_lesion_mask_on_grid / lesion insertion.
    """
    spacing = tuple(float(s) for s in spacing_mm[:3])
    if not spacing or spacing[0] <= 0:
        return None
    shape = tuple(int(s) for s in shape[:3])
    if len(shape) < 3 or min(shape) < 1:
        return None
    r_mm = float(radius_mm)
    scale = float(size_scale)
    if r_mm <= 0 or scale <= 0:
        return 0.0
    if center is None:
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    radius_px = r_mm / spacing[0]
    preview = type("_SpherePreview", (), {})()
    preview.custom_shape = None
    preview.center = (cx, cy, cz)
    preview.radius = radius_px
    preview.size_scale = scale
    preview.intensity = 1.0
    return lesion_volume_ml_on_grid(shape, preview, spacing)


def preview_library_volume_ml_on_grid(
    custom_shape, size_scale, shape, spacing_mm, center=None, uniform_intensity=False
):
    """Grid-aligned library lesion volume for UI preview (matches insertion)."""
    spacing = tuple(float(s) for s in spacing_mm[:3])
    if not spacing or spacing[0] <= 0:
        return None
    shape = tuple(int(s) for s in shape[:3])
    if len(shape) < 3 or min(shape) < 1:
        return None
    if custom_shape is None:
        return None
    if center is None:
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    preview = type("_LibraryPreview", (), {})()
    preview.custom_shape = np.asarray(custom_shape, dtype=np.float32)
    preview.center = (cx, cy, cz)
    preview.radius = 0
    preview.size_scale = float(size_scale)
    preview.intensity = 1.0
    preview.uniform_intensity = bool(uniform_intensity)
    return lesion_volume_ml_on_grid(shape, preview, spacing)


def reference_lesion_voxels_on_grid(truth_volume, lesion_mask):
    """
    Ground-truth voxel intensities under lesion_mask from the lesion definition
    on the analysis grid (build_lesion_truth_on_grid — mask + uniform or template
    intensity). Analysis does not read the cached multi-lesion phantom map.
    """
    mask = np.asarray(lesion_mask, dtype=bool)
    truth_volume = np.asarray(truth_volume, dtype=np.float32)
    return truth_volume[mask]


def expected_lesion_mean_on_grid(truth_volume, lesion_mask, lesion):
    """Mean inserted truth under the final grid mask from the lesion definition."""
    vals = reference_lesion_voxels_on_grid(truth_volume, lesion_mask)
    if vals.size > 0:
        return float(np.mean(vals))
    if getattr(lesion, "custom_shape", None) is None:
        return float(getattr(lesion, "intensity", 0.0))
    return 0.0


def lesion_truth_mean_on_grid(shape, lesion):
    """Mean inserted-truth SUV under the grid-aligned lesion mask."""
    mask = build_lesion_mask_on_grid(shape, lesion)
    if not np.any(mask):
        return float(getattr(lesion, "intensity", 0.0))
    truth = build_lesion_truth_on_grid(shape, lesion)
    return float(np.mean(truth[mask]))


def lesion_volume_spacing_mm(lesion, recon_spacing_mm):
    """
    Voxel spacing for grid-aligned mask volume and mm-based edge metrics.

    Masks from build_lesion_mask_on_grid live on the reconstruction object grid,
    so physical extent and signed-distance shells must use recon spacing — not the
    library import spacing (source_spacing_mm), which only describes the cropped PET.
    """
    spacing = tuple(float(s) for s in recon_spacing_mm[:3])
    if len(spacing) >= 3:
        return spacing[:3]
    if len(spacing) == 2:
        return (spacing[0], spacing[1], spacing[1])
    if len(spacing) == 1:
        return (spacing[0], spacing[0], spacing[0])
    return (1.0, 1.0, 1.0)


def lesion_volume_ml_on_grid(shape, lesion, recon_spacing_mm):
    """Grid-aligned lesion volume used for analysis and RC reporting."""
    mask = build_lesion_mask_on_grid(shape, lesion)
    spacing = lesion_volume_spacing_mm(lesion, recon_spacing_mm)
    if spacing is None:
        return 0.0
    return lesion_volume_ml_from_mask(mask, spacing)


def parse_iteration_from_scenario_name(scenario_name):
    """Parse trailing _iterN from a result/scenario name, e.g. Recon_1_iter2 -> 2."""
    low = str(scenario_name).lower()
    if "_iter" not in low:
        return None
    try:
        return int(low.split("_iter")[-1].split("_")[0])
    except (TypeError, ValueError):
        return None


def iter_number_for_recon_result(result_name, n_iters=None, save_all_iterations=False):
    """
    Iteration label for Tab 4 analysis.
    Per-iteration saves use _iterN in the name; a single final volume uses n_iters.
    """
    parsed = parse_iteration_from_scenario_name(result_name)
    if parsed is not None:
        return parsed
    if n_iters is not None:
        try:
            n = int(n_iters)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return 0


def resize_volume_preserve_morphology(array, size_scale_factor):
    """
    Resizes the 3D array using Cubic Spline Interpolation (order=3).
    Preserves morphology but stretches texture.
    """
    if size_scale_factor == 1.0:
        return array
        
    # Zoom factors for (x, y, z). Keep them equal to preserve shape.
    zoom_factors = (size_scale_factor, size_scale_factor, size_scale_factor)
    
    # Order 3 (cubic) is best for medical images to preserve smooth edges
    # prefilter=False is often faster and sharper for simple scaling
    resized_array = scipy.ndimage.zoom(array, zoom_factors, order=3, prefilter=False)
    
    # Clean up negative ringing artifacts from cubic interpolation
    resized_array[resized_array < 0] = 0
    
    return resized_array


DEFAULT_PSF_FWHM_MM = (3.4, 3.4, 3.8)


def normalize_psf_fwhm_mm(fwhm_mm, default=DEFAULT_PSF_FWHM_MM):
    """Return (fwhm_x, fwhm_y, fwhm_z) in mm; scalar is expanded isotropically."""
    if fwhm_mm is None:
        return tuple(float(x) for x in default[:3])
    if hasattr(fwhm_mm, "__len__") and not isinstance(fwhm_mm, (str, bytes)):
        parts = [float(x) for x in fwhm_mm[:3]]
        while len(parts) < 3:
            parts.append(parts[-1] if parts else float(default[0]))
        return tuple(max(parts[i], 1e-3) for i in range(3))
    val = max(float(fwhm_mm), 1e-3)
    return (val, val, val)


def apply_post_recon_gaussian_fwhm_mm(volume, voxel_spacing_mm, fwhm_mm):
    """
    Optional clinical-style Gaussian smoothing on a 3D recon volume.
    voxel_spacing_mm: (dx, dy, dz) in mm per voxel, matching volume axis order.
    fwhm_mm: scalar or (fx, fy, fz) Gaussian FWHM in mm per axis.
    """
    if volume is None or fwhm_mm is None:
        return volume
    fwhm_xyz = normalize_psf_fwhm_mm(fwhm_mm)
    if min(fwhm_xyz) <= 0:
        return volume
    sp_list = [float(x) for x in voxel_spacing_mm]
    while len(sp_list) < 3:
        sp_list.append(sp_list[-1] if sp_list else 1.0)
    spacing = np.array([max(sp_list[i], 1e-6) for i in range(3)], dtype=np.float64)
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sigmas = tuple(fwhm_xyz[i] / fwhm_to_sigma / spacing[i] for i in range(3))
    # Keep dtype at float32 to avoid ~2x transient memory spike from float64 upcast.
    in_f32 = np.ascontiguousarray(volume, dtype=np.float32)
    out = np.empty_like(in_f32)
    scipy.ndimage.gaussian_filter(in_f32, sigma=sigmas, mode="nearest", output=out)
    return out


def _pvc_spacing_and_fwhm(spacing_mm, fwhm_mm):
    spacing = tuple(float(s) for s in spacing_mm[:3]) if spacing_mm is not None else (1.0, 1.0, 1.0)
    while len(spacing) < 3:
        spacing = spacing + (spacing[-1],)
    fwhm_xyz = normalize_psf_fwhm_mm(fwhm_mm)
    return spacing, fwhm_xyz


def _pvc_psf_sigmas_vox(spacing_mm, fwhm_mm):
    spacing, fwhm_xyz = _pvc_spacing_and_fwhm(spacing_mm, fwhm_mm)
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return tuple(fwhm_xyz[i] / fwhm_to_sigma / max(spacing[i], 1e-6) for i in range(3))


def _pvc_background_mean(volume, mask, bg_ring_vox=3):
    ring = max(int(bg_ring_vox), 1)
    dilated = scipy.ndimage.binary_dilation(mask, iterations=ring)
    bg = np.logical_and(dilated, np.logical_not(mask))
    return float(np.mean(volume[bg])) if np.any(bg) else 0.0


def _mask_rsf_normalized(mask, sigmas_vox):
    """Regional spread function: blurred mask normalized to 1 at the core."""
    rsf = scipy.ndimage.gaussian_filter(
        np.asarray(mask, dtype=np.float32), sigma=sigmas_vox, mode="nearest"
    )
    mask = np.asarray(mask, dtype=bool)
    peak = float(np.max(rsf[mask])) if np.any(mask) else 0.0
    if peak < 1e-6:
        return rsf
    return rsf / peak


def _fft_convolve_same_shape(arr, psf):
    return np.real(np.fft.ifftn(np.fft.fftn(arr) * np.fft.fftn(psf)))


def precorrect_params_from_strength(strength_percent):
    """
    Map UI sharpening strength (5–95 %) to boundary-only edge correction.

    Only voxels in a thin surface shell (1 voxel, optionally 2 at high strength)
    are modified; all interior voxels stay at measured values.
    """
    s = float(np.clip(strength_percent, 5.0, 95.0)) / 100.0
    shell_layers = 1 if s < 0.88 else 2
    rim_blend_scale = float(0.40 + 0.60 * s)
    return shell_layers, rim_blend_scale


def regularization_from_sharpening_strength(strength_percent):
    """Backward-compatible stub (rim-only path ignores Wiener regularization)."""
    return precorrect_params_from_strength(strength_percent)[1]


def _richardson_lucy_masked(measured, mask, psf, n_iter, initial, fill_val):
    """Masked Richardson–Lucy refinement (helps rim voxels after Wiener init)."""
    estimate = np.maximum(np.asarray(initial, dtype=np.float64), 0.0)
    work = np.asarray(measured, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    estimate[~mask] = fill_val
    for _ in range(max(int(n_iter), 0)):
        blurred = _fft_convolve_same_shape(estimate, psf)
        ratio = work / np.maximum(blurred, 1e-8)
        correction = _fft_convolve_same_shape(ratio, psf)
        estimate[mask] = estimate[mask] * np.maximum(correction[mask], 0.0)
        estimate[~mask] = fill_val
        np.maximum(estimate, 0.0, out=estimate)
    return estimate


def _preserve_lesion_core(volume, out, mask, rsf, core_rsf_min=0.88):
    """Keep interior voxels at measured values (no peak amplification in the core)."""
    mask = np.asarray(mask, dtype=bool)
    core = mask & (rsf >= float(core_rsf_min))
    if np.any(core):
        vol = np.asarray(volume, dtype=np.float64)
        blended = np.asarray(out, dtype=np.float64).copy()
        blended[core] = vol[core]
        return blended
    return out


def _lesion_boundary_shell(mask, shell_layers=1):
    """Voxels in the outermost N layers of the binary lesion mask (surface shell only)."""
    mask = np.asarray(mask, dtype=bool)
    layers = max(int(shell_layers), 1)
    if layers == 1:
        return mask & ~scipy.ndimage.binary_erosion(mask, iterations=1)

    shell = np.zeros(mask.shape, dtype=bool)
    current = mask.copy()
    for _ in range(layers):
        eroded = scipy.ndimage.binary_erosion(current, iterations=1)
        shell |= current & ~eroded
        current = eroded
        if not np.any(current):
            break
    return shell


def _rbv_values(volume, rsf, suv_bg, idx):
    vol = np.asarray(volume, dtype=np.float64)
    rc = np.maximum(np.asarray(rsf, dtype=np.float64)[idx], 0.03)
    rbv = (vol[idx] - (1.0 - rc) * float(suv_bg)) / rc
    return np.maximum(rbv, 0.0)


def _blend_rim_rbv(
    volume,
    out,
    mask,
    rsf,
    suv_bg,
    boundary,
    rim_blend_scale=1.0,
):
    """
    RBV-style PVE correction on boundary voxels only.

    Every voxel outside ``boundary`` is reset to the measured crop unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    boundary = np.asarray(boundary, dtype=bool) & mask
    vol = np.asarray(volume, dtype=np.float64)
    blended = vol.copy()

    if not np.any(boundary):
        return blended

    rsf_f = np.asarray(rsf, dtype=np.float64)
    b_idx = boundary
    rbv = _rbv_values(vol, rsf_f, suv_bg, b_idx)
    w = np.clip(
        (0.92 - rsf_f[b_idx]) / max(0.92 - 0.55, 1e-3),
        0.0,
        1.0,
    )
    w = np.clip(w * float(rim_blend_scale), 0.0, 1.0)
    blended[b_idx] = (1.0 - w) * vol[b_idx] + w * rbv
    return blended


def _gaussian_psf_kernel_3d(shape, sigmas_vox):
    """Centered 3D Gaussian PSF kernel for FFT deconvolution (sums to 1)."""
    shape = tuple(int(s) for s in shape[:3])
    sigmas = tuple(max(float(s), 1e-3) for s in sigmas_vox[:3])
    while len(sigmas) < 3:
        sigmas = sigmas + (sigmas[-1],)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    kernel = np.exp(
        -0.5
        * (
            ((zz - cz) / sigmas[0]) ** 2
            + ((yy - cy) / sigmas[1]) ** 2
            + ((xx - cx) / sigmas[2]) ** 2
        )
    )
    kernel = kernel.astype(np.float64)
    kernel /= float(kernel.sum()) + 1e-12
    return np.fft.ifftshift(kernel)


def wiener_deconv_lesion_crop(
    volume,
    mask,
    spacing_mm,
    fwhm_mm,
    regularization=0.05,
    rl_iterations=None,
    shell_layers=1,
    rim_blend_scale=1.0,
    **_ignored,
):
    """
    Boundary-only edge sharpening on a lesion crop.

    Only the outermost 1–2 voxel layer of the mask is RBV-corrected; all other
    voxels remain at measured values.
    """
    vol = np.asarray(volume, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if vol.shape != mask.shape or not np.any(mask):
        return np.asarray(volume, dtype=np.float32)

    sigmas = _pvc_psf_sigmas_vox(spacing_mm, fwhm_mm)
    rsf = _mask_rsf_normalized(mask, sigmas)
    suv_bg = _pvc_background_mean(vol, mask, bg_ring_vox=3)
    boundary = _lesion_boundary_shell(mask, shell_layers=shell_layers)
    blended = _blend_rim_rbv(
        vol, vol, mask, rsf, suv_bg, boundary, rim_blend_scale=rim_blend_scale,
    )

    out = np.zeros(vol.shape, dtype=np.float32)
    out[mask] = np.maximum(blended[mask], 0.0).astype(np.float32)
    return out


def wiener_deconv_roundtrip_nmse(template, measured, mask, spacing_mm, fwhm_mm):
    """
    Normalized MSE between measured crop and PSF re-blur of the template (mask only).
    Lower is better; used to validate deconv strength.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    reb = apply_post_recon_gaussian_fwhm_mm(
        np.asarray(template, dtype=np.float32), spacing_mm, fwhm_mm
    )
    m = np.asarray(measured, dtype=np.float64)[mask]
    r = np.asarray(reb, dtype=np.float64)[mask]
    denom = float(np.mean(m ** 2)) + 1e-12
    return float(np.mean((r - m) ** 2) / denom)


def apply_lesion_precorrection(
    volume, mask, spacing_mm, fwhm_mm, sharpening_strength_percent=50.0
):
    """Rim-only edge sharpening when building a library lesion template."""
    shell_layers, rim_scale = precorrect_params_from_strength(sharpening_strength_percent)
    return wiener_deconv_lesion_crop(
        volume,
        mask,
        spacing_mm,
        fwhm_mm,
        shell_layers=shell_layers,
        rim_blend_scale=rim_scale,
    )