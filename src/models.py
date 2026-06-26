class Lesion:
    """A data class to store properties of a lesion."""
    def __init__(
        self,
        center,
        radius,
        intensity,
        custom_shape=None,
        size_scale=1.0,
        uniform_intensity=False,
        use_recon_voxel_spacing=False,
        source_spacing_mm=None,
    ):
        self.center = center
        self.radius = radius
        self.intensity = intensity
        self.custom_shape = custom_shape # Unscaled 3D library template; size_scale applied at insert/recon
        self.size_scale = size_scale     # Scaling factor applied once via utils library_* helpers
        self.custom_shape_is_source = True  # False only for legacy workflow lesions saved before this fix
        # Library-only: keep morphology, fill mask with fixed intensity (sphere-like truth).
        self.uniform_intensity = bool(uniform_intensity)
        # Library-only: when True, Tab 3/4 volume and edge metrics use Tab 2 recon spacing.
        self.use_recon_voxel_spacing = bool(use_recon_voxel_spacing)
        self.source_spacing_mm = (
            tuple(float(s) for s in source_spacing_mm[:3])
            if source_spacing_mm is not None
            else None
        )


class ReconstructionQueueItem:
    """A data class to store reconstruction parameters for queuing."""
    def __init__(self, name, modality, algo, n_iters, n_subsets, projection_method='ANALYTICAL', save_all_iterations=False, **kwargs):
        self.name = name
        self.modality = modality
        self.algo = algo
        self.n_iters = n_iters
        self.n_subsets = n_subsets
        self.projection_method = projection_method
        self.save_all_iterations = save_all_iterations
        self.params = kwargs
        self.result = None # Note: this will hold the final result for multi-iteration saves
        self.status = "Queued"
        # Set on saved recon outputs: 1..N per-iter snapshots, or N for a final-only volume.
        self.iter_number = None

    def __str__(self):
        extra = " (all iters)" if self.save_all_iterations else ""
        pf = self.params.get('post_recon_fwhm_mm')
        post = f", post {pf:g} mm" if pf else ""
        return f"{self.name}{extra} ({self.algo}, {self.n_iters}it, {self.n_subsets}sub{post})"