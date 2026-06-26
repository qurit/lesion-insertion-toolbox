import sys
import os
import numpy as np
import torch
import traceback
import pydicom
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

from src.utils import (
    add_3d_spherical_lesion,
    apply_library_lesion_to_volume,
    insert_real_lesion_array,
    resize_volume_preserve_morphology,
    recon_thread_is_cancelled,
)

def calculate_spect_bed_z_offsets(files_NM):
    """
    Per-bed z voxel offsets in the stitched multibed SPECT object grid.
    Returns {bed_index: z_offset_voxels}.
    """
    from pytomography.io.SPECT import dicom

    object_meta, _ = dicom.get_metadata(files_NM[0])
    dz_mm = float(object_meta.dr[2]) * 10.0
    if dz_mm <= 0:
        dz_mm = 1.0

    dss = np.array([pydicom.dcmread(f) for f in files_NM])
    zs_mm = np.array([ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss])
    sort_order = np.argsort(zs_mm)
    zs_mm_sorted = zs_mm[sort_order]
    zs_voxel_sorted = np.round((zs_mm_sorted - zs_mm_sorted[0]) / dz_mm).astype(int)
    z_offsets = {}
    original_indices_in_sorted_array = np.argsort(sort_order)
    for original_idx in range(len(files_NM)):
        sorted_position = original_indices_in_sorted_array[original_idx]
        z_offsets[original_idx] = int(zs_voxel_sorted[sorted_position])
    return z_offsets


class SPECTReconstructionThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object, object, str)

    def __init__(self, parent, projection_file, ct_folder, algo, n_iters, n_subsets,
                 beta=None, gamma=None, n_neighbours=8,
                 support_kernel_param=0.005, distance_kernel_param=0.4, top_n=40, kernel_on_gpu=True,
                 collimator_name='SY-ME', energy_kev=208, intrinsic_resolution=0.38,
                 index_peak=0, index_lower=1, index_upper=2):
        super().__init__(parent)
        self.parent = parent
        self.projection_file = projection_file
        self.ct_folder = ct_folder
        self.algo = algo
        self.n_iters = n_iters
        self.n_subsets = n_subsets
        self.beta = beta
        self.gamma = gamma
        self.n_neighbours = n_neighbours
        self.support_kernel_param = support_kernel_param
        self.distance_kernel_param = distance_kernel_param
        self.top_n = top_n
        self.kernel_on_gpu = kernel_on_gpu
        self.collimator_name = collimator_name
        self.energy_kev = energy_kev
        self.intrinsic_resolution = intrinsic_resolution
        self.index_peak = index_peak
        self.index_lower = index_lower
        self.index_upper = index_upper
        self._cancel_requested = False

    def _calculate_bed_z_offsets(self, files_NM):
        return calculate_spect_bed_z_offsets(files_NM)

    def run(self):
        try:
            from pytomography.io.SPECT import dicom
            from pytomography.projectors.SPECT import SPECTSystemMatrix
            from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
            from pytomography.likelihoods import PoissonLogLikelihood

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            is_multibed = os.path.isdir(self.projection_file)
            files_CT = [os.path.join(self.ct_folder, file) for file in os.listdir(self.ct_folder)]

            if is_multibed:
                files_NM = sorted([os.path.join(self.projection_file, f) for f in os.listdir(self.projection_file) if f.lower().endswith(('.dcm', '.IMA'))])
                if not files_NM:
                    raise ValueError("No DICOM files found in the specified multi-bed projection folder.")

                all_projections = dicom.load_multibed_projections(files_NM)
                recons = []
                num_beds = len(files_NM)

                for i in range(num_beds):
                    if recon_thread_is_cancelled(self):
                        self.finished_signal.emit(None, None, "Cancelled")
                        return
                    file_NM_i = files_NM[i]
                    projections_i = all_projections[i].to(device)
                    object_meta, proj_meta = dicom.get_metadata(file_NM_i, index_peak=self.index_peak)
                    photopeak = projections_i[self.index_peak]
                    scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM_i, projections_i, index_peak=self.index_peak, index_lower=self.index_lower, index_upper=self.index_upper)
                    attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM_i, E_SPECT=self.energy_kev)
                    att_transform = SPECTAttenuationTransform(attenuation_map)
                    psf_meta = dicom.get_psfmeta_from_scanner_params(self.collimator_name, self.energy_kev, intrinsic_resolution=self.intrinsic_resolution)
                    psf_transform = SPECTPSFTransform(psf_meta)
                    system_matrix = SPECTSystemMatrix(obj2obj_transforms=[att_transform, psf_transform], proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
                    likelihood = PoissonLogLikelihood(system_matrix, projections=photopeak, additive_term=scatter)
                    recon_algorithm = self._get_recon_algorithm(likelihood, system_matrix, att_transform, device)
                    recon_i = None
                    for it in range(self.n_iters):
                        if recon_thread_is_cancelled(self):
                            self.finished_signal.emit(None, None, "Cancelled")
                            return
                        recon_i = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                        current_step = i * self.n_iters + it + 1
                        progress_label = f"Processing Bed {i+1}/{num_beds}, Iteration {it+1}/{self.n_iters}"
                        self.progress_signal.emit(current_step, progress_label)
                    recons.append(recon_i)

                final_recon_tensor = dicom.stitch_multibed(recons=torch.stack(recons), files_NM=files_NM)
                z_offsets = self._calculate_bed_z_offsets(files_NM)
                final_recon = final_recon_tensor.cpu().numpy()
                self.finished_signal.emit(final_recon, z_offsets, "Success")
            else:
                object_meta, proj_meta = dicom.get_metadata(self.projection_file, index_peak=self.index_peak)
                all_projections = dicom.get_projections(self.projection_file).to(device)
                photopeak = all_projections[self.index_peak]
                scatter = dicom.get_energy_window_scatter_estimate(self.projection_file, index_peak=self.index_peak, index_lower=self.index_lower, index_upper=self.index_upper)
                attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, self.projection_file, E_SPECT=self.energy_kev)
                att_transform = SPECTAttenuationTransform(attenuation_map)
                psf_meta = dicom.get_psfmeta_from_scanner_params(self.collimator_name, self.energy_kev, intrinsic_resolution=self.intrinsic_resolution)
                psf_transform = SPECTPSFTransform(psf_meta)
                system_matrix = SPECTSystemMatrix(obj2obj_transforms=[att_transform, psf_transform], proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
                likelihood = PoissonLogLikelihood(system_matrix, projections=photopeak, additive_term=scatter)
                recon_algorithm = self._get_recon_algorithm(likelihood, system_matrix, att_transform, device)
                reconstruction = None
                for i in range(self.n_iters):
                    if recon_thread_is_cancelled(self):
                        self.finished_signal.emit(None, None, "Cancelled")
                        return
                    reconstruction = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                    progress_label = f"Running iteration {i+1}/{self.n_iters}"
                    self.progress_signal.emit(i + 1, progress_label)
                final_recon = reconstruction.cpu().numpy()
                self.finished_signal.emit(final_recon, None, "Success")
        except Exception as e:
            self.finished_signal.emit(None, None, f"SPECT Initial Recon Error: {traceback.format_exc()}")

    def _get_recon_algorithm(self, likelihood, system_matrix, att_transform, device):
        from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
        from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight
        from pytomography.transforms.shared import KEMTransform
        from pytomography.projectors.shared import KEMSystemMatrix
        from pytomography.likelihoods import PoissonLogLikelihood
        amap = att_transform.attenuation_map.to(device)
        if self.algo == "OSEM":
            return OSEM(likelihood)
        elif self.algo == "OSMAPOSL":
            weight = TopNAnatomyNeighbourWeight(amap, N_neighbours=self.n_neighbours)
            prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma, weight=weight)
            return OSMAPOSL(likelihood=likelihood, prior=prior)
        elif self.algo == "BSREM":
            weight = TopNAnatomyNeighbourWeight(amap, N_neighbours=self.n_neighbours)
            prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma, weight=weight)
            return BSREM(likelihood=likelihood, prior=prior, relaxation_sequence=lambda n: 1/(n/50+1))
        elif self.algo == "KEM":
            kem_transform = KEMTransform(support_objects=[amap], support_kernels_params=[[self.support_kernel_param]], distance_kernel_params=[self.distance_kernel_param], top_N=self.top_n, kernel_on_gpu=self.kernel_on_gpu)
            system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)
            likelihood_kem = PoissonLogLikelihood(system_matrix_kem, projections=likelihood.projections, additive_term=likelihood.additive_term)
            return KEM(likelihood_kem)
        else:
            raise ValueError(f"Unknown reconstruction algorithm: {self.algo}")


class SPECTLesionFinalizationThread(QThread):
    item_finished_signal = pyqtSignal(str)
    result_ready_signal = pyqtSignal(object, str, str)
    progress_signal = pyqtSignal(int, int, str)

    def __init__(self, parent, lesion_objects, projection_file_path, ct_folder_path, n_iters, n_subsets, recon_method, beta, gamma, n_neighbours, support_kernel, distance_kernel, top_n, kernel_gpu, collimator, energy, intrinsic_res, idx_peak, idx_lower, idx_upper, bed_z_offsets, projection_method,
                 save_all_iterations=False, base_name="", initial_image=None,
                 mc_n_events=1e6, mc_n_parallel=8, mc_isotope_names=['lu177'], mc_isotope_ratios=[1.0], mc_crystal_thickness=0.9525, mc_cover_thickness=0.1, mc_backscatter_thickness=6.6, mc_energy_res_140=10, mc_adv_collimator=True,
                 post_recon_fwhm_mm=None):
        super().__init__(parent)
        self.parent = parent
        self.lesion_objects = lesion_objects
        self.projection_file_path = projection_file_path
        self.ct_folder_path = ct_folder_path
        self.n_iters = n_iters
        self.n_subsets = n_subsets
        self.recon_method = recon_method
        self.beta = beta
        self.gamma = gamma
        self.n_neighbours = n_neighbours
        self.support_kernel = support_kernel
        self.distance_kernel = distance_kernel
        self.top_n = top_n
        self.kernel_gpu = kernel_gpu
        self.collimator = collimator
        self.energy = energy
        self.intrinsic_res = intrinsic_res
        self.idx_peak = idx_peak
        self.idx_lower = idx_lower
        self.idx_upper = idx_upper
        self.bed_z_offsets = bed_z_offsets
        self.projection_method = projection_method
        self.save_all_iterations = save_all_iterations
        self.base_name = base_name
        self.initial_image = initial_image
        self.mc_n_events = mc_n_events
        self.mc_n_parallel = mc_n_parallel
        self.mc_isotope_names = mc_isotope_names
        self.mc_isotope_ratios = mc_isotope_ratios
        self.mc_crystal_thickness = mc_crystal_thickness
        self.mc_cover_thickness = mc_cover_thickness
        self.mc_backscatter_thickness = mc_backscatter_thickness
        self.mc_energy_res_140 = mc_energy_res_140
        self.mc_adv_collimator = mc_adv_collimator
        self.post_recon_fwhm_mm = post_recon_fwhm_mm
        self._cancel_requested = False

    def run(self):
        try:
            from pytomography.utils import simind_mc
            from pytomography.projectors.SPECT import MonteCarloHybridSPECTSystemMatrix
            from pytomography.io.SPECT import dicom
            from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
            from pytomography.projectors.SPECT import SPECTSystemMatrix
            from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
            from pytomography.likelihoods import PoissonLogLikelihood
            import torch
            
            from src.utils import add_3d_spherical_lesion, apply_library_lesion_to_volume, insert_real_lesion_array, resize_volume_preserve_morphology, apply_post_recon_gaussian_fwhm_mm

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            spect_dr_mm = None
            final_recon = None
            is_multibed = os.path.isdir(self.projection_file_path)
            files_CT = [os.path.join(self.ct_folder_path, file) for file in os.listdir(self.ct_folder_path)]

            if is_multibed:
                files_NM = sorted([os.path.join(self.projection_file_path, f) for f in os.listdir(self.projection_file_path) if f.lower().endswith(('.dcm', '.IMA'))])
                if not files_NM: raise ValueError("No DICOM files found for multi-bed lesion finalization.")

                real_projectionss = dicom.load_multibed_projections(files_NM)
                recons_with_lesion = []
                num_beds = len(files_NM)
                
                recon_algorithms = []

                for i in range(num_beds):
                    if recon_thread_is_cancelled(self):
                        self.item_finished_signal.emit("Cancelled")
                        return
                    file_NM_i = files_NM[i]
                    object_meta_i, proj_meta_i = dicom.get_metadata(file_NM_i, index_peak=self.idx_peak)
                    if spect_dr_mm is None:
                        spect_dr_mm = tuple(float(x) for x in object_meta_i.dr)

                    # Work in raw counts (no ActualFrameDuration scaling).
                    poisson_scaling = 1.0

                    real_proj_i_all_windows = real_projectionss[i].to(device)
                    photopeak_projections = real_proj_i_all_windows[self.idx_peak].clone().float()

                    attenuation_map_recon = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM_i, E_SPECT=self.energy)
                    att_transform_recon = SPECTAttenuationTransform(attenuation_map_recon)
                    psf_meta = dicom.get_psfmeta_from_scanner_params(self.collimator, self.energy, self.intrinsic_res)
                    psf_transform = SPECTPSFTransform(psf_meta)

                    system_matrix_basic = SPECTSystemMatrix(
                        obj2obj_transforms=[att_transform_recon, psf_transform],
                        proj2proj_transforms=[],
                        object_meta=object_meta_i,
                        proj_meta=proj_meta_i
                    )

                    # Build the lesion mask (in patient/local coordinates for this bed).
                    local_lesion_mask_np = np.zeros(object_meta_i.shape, dtype=np.float32)
                    z_offset = self.bed_z_offsets[i]
                    for lesion in self.lesion_objects:
                        current_scale = getattr(lesion, 'size_scale', 1.0)
                        
                        global_center = lesion.center
                        local_center = (global_center[0], global_center[1], global_center[2] - z_offset)
                        
                        if lesion.custom_shape is not None:
                             local_lesion_mask_np = apply_library_lesion_to_volume(
                                 local_lesion_mask_np, lesion, center_coords=local_center
                             )
                        else:
                             effective_radius = lesion.radius * current_scale
                             local_lesion_mask_np = add_3d_spherical_lesion(local_lesion_mask_np, local_center, effective_radius, lesion.intensity)
                    
                    local_lesion_mask_np = np.flip(local_lesion_mask_np, axis=1).copy()
                    local_lesion_mask_tensor = torch.tensor(local_lesion_mask_np, device=device)

                    # Background ablation: probabilistically remove counts that would have
                    # originated from the lesion ROI, so the inserted lesion replaces (rather
                    # than adds to) the original activity in that region.
                    if self.initial_image is not None and local_lesion_mask_tensor.sum() > 0:
                        local_initial_np = self.initial_image[:, :, max(0, z_offset):min(self.initial_image.shape[2], z_offset + object_meta_i.shape[2])]
                        if local_initial_np.shape[2] < object_meta_i.shape[2]:
                            diff = object_meta_i.shape[2] - local_initial_np.shape[2]
                            local_initial_np = np.pad(local_initial_np, ((0,0),(0,0),(0, diff)), 'constant')

                        local_initial_tensor = torch.tensor(np.flip(local_initial_np, axis=1).copy(), dtype=torch.float32).to(device)
                        roi_mask = (local_lesion_mask_tensor > 0.001).float()
                        img_roi = local_initial_tensor * roi_mask

                        proj_roi_mean = system_matrix_basic.forward(img_roi)
                        proj_total = system_matrix_basic.forward(local_initial_tensor)

                        prob_map = proj_roi_mean / (proj_total + 1e-9)
                        prob_map = torch.clamp(prob_map, 0.0, 1.0)

                        counts_to_remove = torch.binomial(photopeak_projections, prob_map)
                        photopeak_projections = torch.clamp(photopeak_projections - counts_to_remove, min=0)

                    # Forward-project the lesion and draw a Poisson realisation (raw counts).
                    lesion_proj_rate = system_matrix_basic.forward(local_lesion_mask_tensor)
                    lesion_proj_poisson = torch.poisson(lesion_proj_rate)
                    projections_with_lesion = photopeak_projections + lesion_proj_poisson

                    if self.projection_method == 'Monte Carlo':
                        self.parent.status_bar.showMessage(f"Running MC Simulation for bed {i+1}...")
                        QApplication.processEvents()
                        energy_window_params = simind_mc.get_energy_window_params_dicom(file_NM_i)
                        attenuation_map_140kev = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM_i, E_SPECT=140.5)
                        system_matrix = MonteCarloHybridSPECTSystemMatrix(
                            obj2obj_transforms=[att_transform_recon, psf_transform],
                            proj2proj_transforms=[],
                            object_meta=object_meta_i,
                            proj_meta=proj_meta_i,
                            attenuation_map_140keV=attenuation_map_140kev,
                            n_events=self.mc_n_events,
                            n_parallel=self.mc_n_parallel,
                            energy_window_params=energy_window_params,
                            primary_window_idx=self.idx_peak,
                            isotope_names=self.mc_isotope_names,
                            isotope_ratios=self.mc_isotope_ratios,
                            collimator_type=self.collimator,
                            crystal_thickness=self.mc_crystal_thickness,
                            cover_thickness=self.mc_cover_thickness,
                            backscatter_thickness=self.mc_backscatter_thickness,
                            energy_resolution_140keV=self.mc_energy_res_140,
                            advanced_collimator_modeling=self.mc_adv_collimator,
                            advanced_energy_resolution_model='siemens'
                        )
                    else: 
                         system_matrix = system_matrix_basic

                    scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM_i, real_proj_i_all_windows, index_peak=self.idx_peak, index_lower=self.idx_lower, index_upper=self.idx_upper)

                    recon_system_matrix = system_matrix
                    likelihood = PoissonLogLikelihood(recon_system_matrix, projections=projections_with_lesion, additive_term=scatter)
                    recon_algorithms.append(self._get_recon_algorithm(likelihood, recon_system_matrix, att_transform_recon, device))

                for it in range(self.n_iters):
                    if recon_thread_is_cancelled(self):
                        self.item_finished_signal.emit("Cancelled")
                        return
                    current_iter_recons_tensors = []
                    for i in range(num_beds):
                        recon_i_tensor = recon_algorithms[i](n_iters=1, n_subsets=self.n_subsets)
                        current_iter_recons_tensors.append(recon_i_tensor)
                    
                    stitched_tensor = dicom.stitch_multibed(recons=torch.stack(current_iter_recons_tensors), files_NM=files_NM)
                    final_recon = stitched_tensor.cpu().numpy()
                    final_recon = np.flip(final_recon, axis=1).copy()
                    final_recon = apply_post_recon_gaussian_fwhm_mm(final_recon, spect_dr_mm, self.post_recon_fwhm_mm)
                    
                    if self.save_all_iterations:
                         self.result_ready_signal.emit(final_recon, f"{self.base_name}_iter{it+1}", "Success")
                    self.progress_signal.emit(it + 1, self.n_iters, f"{self.base_name} (Multi-Bed)")

            else: # Single-bed
                object_meta, proj_meta = dicom.get_metadata(self.projection_file_path, index_peak=self.idx_peak)
                spect_dr_mm = tuple(float(x) for x in object_meta.dr)
                real_proj_all_windows = dicom.get_projections(self.projection_file_path).to(device)
                real_proj_photopeak = real_proj_all_windows[self.idx_peak].clone().float()

                # Work in raw counts (no ActualFrameDuration scaling).
                poisson_scaling = 1.0

                attenuation_map_recon = dicom.get_attenuation_map_from_CT_slices(files_CT, self.projection_file_path, E_SPECT=self.energy)
                att_transform_recon = SPECTAttenuationTransform(attenuation_map_recon)
                psf_meta = dicom.get_psfmeta_from_scanner_params(self.collimator, self.energy, self.intrinsic_res)
                psf_transform = SPECTPSFTransform(psf_meta)

                system_matrix_basic = SPECTSystemMatrix(
                    obj2obj_transforms=[att_transform_recon, psf_transform],
                    proj2proj_transforms=[],
                    object_meta=object_meta,
                    proj_meta=proj_meta
                )

                # Build the lesion mask in patient coordinates.
                lesion_array = np.zeros(object_meta.shape, dtype=np.float32)
                for l in self.lesion_objects:
                    current_scale = getattr(l, 'size_scale', 1.0)
                    if l.custom_shape is not None:
                        lesion_array = apply_library_lesion_to_volume(lesion_array, l)
                    else:
                        effective_radius = l.radius * current_scale
                        lesion_array = add_3d_spherical_lesion(lesion_array, l.center, effective_radius, l.intensity)
                
                lesion_array = np.flip(lesion_array, axis=1).copy()
                lesion_mask = torch.tensor(lesion_array, dtype=torch.float32).to(device)

                # Background ablation (see multibed branch for the rationale).
                if self.initial_image is not None and lesion_mask.sum() > 0:
                    img_tensor = torch.tensor(np.flip(self.initial_image, axis=1).copy(), dtype=torch.float32).to(device)
                    roi_mask = (lesion_mask > 0.001).float()
                    img_roi = img_tensor * roi_mask

                    proj_roi_mean = system_matrix_basic.forward(img_roi)
                    proj_total = system_matrix_basic.forward(img_tensor)

                    prob_map = proj_roi_mean / (proj_total + 1e-9)
                    prob_map = torch.clamp(prob_map, 0.0, 1.0)

                    counts_to_remove = torch.binomial(real_proj_photopeak, prob_map)
                    real_proj_photopeak = torch.clamp(real_proj_photopeak - counts_to_remove, min=0)

                # Forward-project the lesion and draw a Poisson realisation (raw counts).
                lesion_proj_rate = system_matrix_basic.forward(lesion_mask)
                lesion_proj_poisson = torch.poisson(lesion_proj_rate)
                projections_with_lesion = real_proj_photopeak + lesion_proj_poisson

                if self.projection_method == 'Monte Carlo':
                    self.parent.status_bar.showMessage("Running MC Simulation...")
                    QApplication.processEvents()
                    energy_window_params = simind_mc.get_energy_window_params_dicom(self.projection_file_path)
                    attenuation_map_140kev = dicom.get_attenuation_map_from_CT_slices(files_CT, self.projection_file_path, E_SPECT=140.5)
                    system_matrix = MonteCarloHybridSPECTSystemMatrix(
                        obj2obj_transforms=[att_transform_recon, psf_transform],
                        proj2proj_transforms=[],
                        object_meta=object_meta,
                        proj_meta=proj_meta,
                        attenuation_map_140keV=attenuation_map_140kev,
                        n_events=self.mc_n_events,
                        n_parallel=self.mc_n_parallel,
                        energy_window_params=energy_window_params,
                        primary_window_idx=self.idx_peak,
                        isotope_names=self.mc_isotope_names,
                        isotope_ratios=self.mc_isotope_ratios,
                        collimator_type=self.collimator,
                        crystal_thickness=self.mc_crystal_thickness,
                        cover_thickness=self.mc_cover_thickness,
                        backscatter_thickness=self.mc_backscatter_thickness,
                        energy_resolution_140keV=self.mc_energy_res_140,
                        advanced_collimator_modeling=self.mc_adv_collimator,
                        advanced_energy_resolution_model='siemens'
                    )
                else: 
                    system_matrix = system_matrix_basic
                
                scatter = dicom.get_energy_window_scatter_estimate(self.projection_file_path, index_peak=self.idx_peak, index_lower=self.idx_lower, index_upper=self.idx_upper)

                recon_system_matrix = system_matrix
                likelihood = PoissonLogLikelihood(recon_system_matrix, projections=projections_with_lesion, additive_term=scatter)
                recon_algorithm = self._get_recon_algorithm(likelihood, recon_system_matrix, att_transform_recon, device)

                recon_result = None
                for i in range(self.n_iters):
                    if recon_thread_is_cancelled(self):
                        self.item_finished_signal.emit("Cancelled")
                        return
                    recon_result = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                    self.progress_signal.emit(i + 1, self.n_iters, self.parent.current_lesion_recon_name)
                    if self.save_all_iterations:
                        result_name = f"{self.base_name}_iter{i+1}"
                        result_np = recon_result.cpu().numpy()
                        result_np = np.flip(result_np, axis=1).copy()
                        result_np = apply_post_recon_gaussian_fwhm_mm(result_np, spect_dr_mm, self.post_recon_fwhm_mm)
                        self.result_ready_signal.emit(result_np, result_name, "Success")
                final_recon = recon_result.cpu().numpy()
                final_recon = np.flip(final_recon, axis=1).copy()
                final_recon = apply_post_recon_gaussian_fwhm_mm(final_recon, spect_dr_mm, self.post_recon_fwhm_mm)

            if not self.save_all_iterations and final_recon is not None:
                self.result_ready_signal.emit(final_recon, self.base_name, "Success")

            self.item_finished_signal.emit("Success")

        except Exception as e:
            self.item_finished_signal.emit(f"SPECT Finalization Error: {traceback.format_exc()}")

    def _get_recon_algorithm(self, likelihood, system_matrix, att_transform, device):
        from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
        from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight
        from pytomography.transforms.shared import KEMTransform
        from pytomography.projectors.shared import KEMSystemMatrix
        from pytomography.likelihoods import PoissonLogLikelihood

        amap = att_transform.attenuation_map.to(device)

        if self.recon_method == "OSEM":
            return OSEM(likelihood)
        elif self.recon_method in ["OSMAPOSL", "BSREM"]:
            weight = TopNAnatomyNeighbourWeight(amap, N_neighbours=self.n_neighbours)
            prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma, weight=weight)
            if self.recon_method == "OSMAPOSL":
                return OSMAPOSL(likelihood, prior=prior)
            else:
                return BSREM(likelihood, prior=prior, relaxation_sequence=lambda n: 1 / (n / 50 + 1))
        elif self.recon_method == "KEM":
            kem_transform = KEMTransform(support_objects=[amap], support_kernels_params=[[self.support_kernel]], distance_kernel_params=[self.distance_kernel], top_N=self.top_n, kernel_on_gpu=self.kernel_gpu)
            system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)
            likelihood_kem = PoissonLogLikelihood(system_matrix_kem, projections=likelihood.projections, additive_term=likelihood.additive_term)
            return KEM(likelihood_kem)
        else:
            raise ValueError(f"Unknown reconstruction algorithm: {self.recon_method}")