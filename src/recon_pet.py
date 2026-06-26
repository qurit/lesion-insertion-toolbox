import numpy as np
import torch
import h5py
import traceback
import gc
import os
import glob
import re
from PyQt5.QtCore import QThread, pyqtSignal

from src.utils import (
    add_3d_spherical_lesion,
    apply_library_lesion_to_volume,
    insert_real_lesion_array,
    resize_volume_preserve_morphology,
    recon_thread_is_cancelled,
)

# --- Natural Sorting Helper ---
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def _resolve_corrections_h5(h5_files, corr_matches, bed_index):
    if corr_matches:
        return corr_matches[0]
    if not h5_files:
        raise FileNotFoundError("No corrections (.h5) files found for PET reconstruction.")
    return h5_files[min(bed_index, len(h5_files) - 1)]

# Permanent memory fix with rollback path:
# - True: lower-peak-memory tensor packing (preallocated merges; intended to match torch.cat/concatenate outputs)
# - False: exact legacy in-memory implementation kept below
USE_LOW_MEMORY_BACKEND = True

# If True, keep the legacy numpy masking path for ablation thinning (higher peak RAM).
# If False, use torch boolean indexing on CPU with the same keep-mask (same selected events, lower peak RAM).
USE_LEGACY_ABLATION_MASK_PATH = False

def _apply_keep_mask_cpu(detector_ids, weights, additive_term, keep_mask_cpu, use_legacy_numpy=False):
    """
    Apply ablation keep-mask on CPU.
    Legacy numpy route is preserved for instant rollback.
    """
    if use_legacy_numpy:
        keep_mask_np = keep_mask_cpu.numpy()
        if isinstance(detector_ids, torch.Tensor): detector_ids = detector_ids.numpy()
        if isinstance(weights, torch.Tensor): weights = weights.numpy()
        if isinstance(additive_term, torch.Tensor): additive_term = additive_term.numpy()
        detector_ids = detector_ids[keep_mask_np]
        weights = weights[keep_mask_np]
        additive_term = additive_term[keep_mask_np]
        detector_ids = torch.from_numpy(detector_ids)
        weights = torch.from_numpy(weights)
        additive_term = torch.from_numpy(additive_term)
        return detector_ids, weights, additive_term

    # Memory-lean path (no round-trip through numpy).
    if not isinstance(detector_ids, torch.Tensor):
        detector_ids = torch.as_tensor(detector_ids)
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights)
    if not isinstance(additive_term, torch.Tensor):
        additive_term = torch.as_tensor(additive_term)
    detector_ids = detector_ids[keep_mask_cpu]
    weights = weights[keep_mask_cpu]
    additive_term = additive_term[keep_mask_cpu]
    return detector_ids, weights, additive_term


def _merge_torch_chunks_no_peak_double_cat(chunks, out_tensor, start_row=0):
    """
    Copy list of 2D/1D torch CPU tensors into out_tensor[start_row:...] without torch.cat(chunks),
    which would temporarily need ~2x peak RAM for the full concatenated size.
    """
    row = int(start_row)
    for ch in chunks:
        if ch is None or ch.numel() == 0:
            continue
        n = int(ch.shape[0])
        out_tensor[row : row + n] = ch
        row += n
    return row


def stitch_pet_multibed_volumes(bed_tensors, shape_xy, z_slices_per_bed, overlap, num_beds, total_z):
    """Stitch per-bed 3D recon tensors; bed_tensors[k] is the volume for bed index k (0..num_beds-1)."""
    multibed_recon = torch.zeros((shape_xy[0], shape_xy[1], total_z), dtype=bed_tensors[0].dtype)
    weights_stitch = (torch.arange(overlap, dtype=torch.float32) + 1) / (overlap + 1)
    weights_stitch = weights_stitch.reshape(1, 1, -1)
    for wrap_i in range(num_beds):
        bed_idx_in_list = (num_beds - 1) - wrap_i
        current_bed = bed_tensors[bed_idx_in_list]
        if not isinstance(current_bed, torch.Tensor):
            current_bed = torch.as_tensor(current_bed)
        mask = torch.ones_like(current_bed)
        if wrap_i != 0:
            mask[:, :, :overlap] *= weights_stitch
        if wrap_i != (num_beds - 1):
            mask[:, :, -overlap:] *= (1 - weights_stitch)
        avg_recon = current_bed * mask
        z_start = z_slices_per_bed * wrap_i - overlap * wrap_i
        z_end = z_slices_per_bed * (wrap_i + 1) - overlap * wrap_i
        d_z = z_end - z_start
        if avg_recon.shape[2] != d_z:
            avg_recon = avg_recon[:, :, :d_z]
        multibed_recon[:, :, z_start:z_end] += avg_recon
    return multibed_recon


# --- Custom Memory-Efficient Loaders ---
def modify_tof_events_custom(TOF_ids, scanner_name):
    if scanner_name == 'discovery_MI':
        return (-(TOF_ids // 13) + 14).to(torch.int32)
    return TOF_ids

def get_detector_ids_custom(listmode_file, info, scanner_name, chunk_size=5000000):
    with h5py.File(listmode_file, 'r') as f:
        key = 'MiceList/TofCoinc'
        if key not in f:
             keys = list(f.keys())
             if 'MiceList' in keys:
                 if 'TofCoinc' in f['MiceList']:
                     key = 'MiceList/TofCoinc'
                 elif 'Coinc' in f['MiceList']:
                     key = 'MiceList/Coinc'
        
        ds = f[key]
        N_events = ds.shape[0]
        has_tof = bool(info['TOF'])
        cols = 3 if has_tof else 2
        
        all_ids = torch.empty((N_events, cols), dtype=torch.int32)
        crystals_per_ring = info['NrCrystalsPerRing']
        
        for i in range(0, N_events, chunk_size):
            end = min(i + chunk_size, N_events)
            events = np.array(ds[i:end]).astype(np.int32)
            events = torch.from_numpy(events)
            det_ids0 = crystals_per_ring * events[:, 0] + events[:, 1]
            det_ids1 = crystals_per_ring * events[:, 2] + events[:, 3]
            
            # Clamp IDs to prevent CUDA errors
            max_id = info['NrCrystalsPerRing'] * info['NrRings'] - 1
            det_ids0 = torch.clamp(det_ids0, 0, max_id)
            det_ids1 = torch.clamp(det_ids1, 0, max_id)

            if has_tof:
                tof_ids = events[:, 4]
                tof_ids = modify_tof_events_custom(tof_ids, scanner_name)
                chunk_result = torch.stack([det_ids0, det_ids1, tof_ids], dim=1)
            else:
                chunk_result = torch.stack([det_ids0, det_ids1], dim=1)
            all_ids[i:end] = chunk_result.to(torch.int32)
            del events, det_ids0, det_ids1, chunk_result
    gc.collect() 
    return all_ids

def get_sensitivity_ids_and_weights_custom(corrections_file, info, chunk_size=5000000):
    with h5py.File(corrections_file, 'r') as data:
        ds_ids = data['all_xtals/xtal_ids']
        ds_atten = data['all_xtals/atten']
        ds_sens = data['all_xtals/sens']
        N_all = ds_ids.shape[0]
        detector_ids_all = torch.empty((N_all, 2), dtype=torch.int32)
        weights_sensitivity = torch.empty((N_all,), dtype=torch.float32)
        crystals_per_ring = info['NrCrystalsPerRing']
        for i in range(0, N_all, chunk_size):
            end = min(i + chunk_size, N_all)
            chunk_ids = ds_ids[i:end].astype(np.int32)
            det1 = crystals_per_ring * chunk_ids[:, 1] + chunk_ids[:, 0]
            det2 = crystals_per_ring * chunk_ids[:, 3] + chunk_ids[:, 2]
            detector_ids_all[i:end, 0] = torch.from_numpy(det1).to(torch.int32)
            detector_ids_all[i:end, 1] = torch.from_numpy(det2).to(torch.int32)
            chunk_weight = ds_atten[i:end] * ds_sens[i:end]
            weights_sensitivity[i:end] = torch.from_numpy(chunk_weight)
            del chunk_ids, det1, det2, chunk_weight
    gc.collect()
    return detector_ids_all, weights_sensitivity

def get_event_weights_custom(corrections_file, n_events):
    try:
        with h5py.File(corrections_file, 'r') as f:
            g = None
            if 'correction_lists' in f: g = f['correction_lists']
            elif 'MiceList' in f: g = f['MiceList']
            
            if g:
                if 'atten' in g:
                    atten = np.array(g['atten'][:n_events], dtype=np.float32)
                    weights = torch.from_numpy(atten)
                    del atten
                    gc.collect()
                    if 'sens' in g:
                        sens = np.array(g['sens'][:n_events], dtype=np.float32)
                        weights *= torch.from_numpy(sens)
                        del sens
                        gc.collect()
                    return weights
            return torch.ones(n_events, dtype=torch.float32)
    except: return torch.ones(n_events, dtype=torch.float32)

def get_additive_term_custom(corrections_file, n_events):
    try:
        with h5py.File(corrections_file, 'r') as f:
            g = None
            if 'correction_lists' in f: g = f['correction_lists']
            elif 'MiceList' in f: g = f['MiceList']
            
            if g:
                if 'randoms' in g:
                    randoms = np.array(g['randoms'][:n_events], dtype=np.float32)
                    additive = torch.from_numpy(randoms)
                    del randoms
                    gc.collect()
                    if 'scatter' in g:
                        scatter = np.array(g['scatter'][:n_events], dtype=np.float32)
                        additive += torch.from_numpy(scatter)
                        del scatter
                        gc.collect()
                    return additive
            return torch.zeros(n_events, dtype=torch.float32)
    except: return torch.zeros(n_events, dtype=torch.float32)

# --- INITIAL RECONSTRUCTION THREAD ---
class PETReconstructionThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object, object, str)

    def __init__(self, parent, folder_path, scanner_name, dr, shape, psf_value, n_iters, n_subsets, algo, beta=None, gamma=None, 
                 overlap=25, z_slices_per_bed=89):
        super().__init__(parent)
        self.parent = parent
        self.folder_path = folder_path
        self.scanner_name = scanner_name
        self.dr = dr
        self.shape = shape 
        self.psf_value = psf_value
        self.n_iters = n_iters
        self.n_subsets = n_subsets
        self.algo = algo
        self.beta = beta
        self.gamma = gamma
        self.overlap = overlap
        self.z_slices_per_bed = z_slices_per_bed
        self._cancel_requested = False

    def run(self):
        try:
            from pytomography.metadata import ObjectMeta
            from pytomography.metadata.PET import PETLMProjMeta
            from pytomography.projectors.PET import PETLMSystemMatrix
            from pytomography.algorithms import OSEM, BSREM
            from pytomography.priors import RelativeDifferencePrior
            from pytomography.likelihoods import PoissonLogLikelihood
            from pytomography.transforms.shared import GaussianFilter
            from pytomography.io.PET import clinical

            all_files = sorted(os.listdir(self.folder_path), key=natural_keys)
            list_files = [os.path.join(self.folder_path, f) for f in all_files if f.lower().endswith('.blf')]
            if len(list_files) == 0: raise FileNotFoundError("No .BLF files found.")
            num_beds = len(list_files)
            is_multibed = num_beds > 1
            info = clinical.get_detector_info(self.scanner_name)

            if not is_multibed:
                self.progress_signal.emit(0, "Running Single Bed Reconstruction...")
                listmode_file = list_files[0]
                h5_files = [os.path.join(self.folder_path, f) for f in all_files if f.lower().endswith('.h5')]
                if len(h5_files) == 0:
                    raise FileNotFoundError("No .h5 corrections file found for single-bed PET reconstruction.")
                corrections_file = h5_files[0]

                detector_ids = get_detector_ids_custom(listmode_file, info, self.scanner_name)
                n_events = detector_ids.shape[0]
                weights = get_event_weights_custom(corrections_file, n_events)
                additive_term = get_additive_term_custom(corrections_file, n_events)
                
                detector_ids_sensitivity, weights_sensitivity = get_sensitivity_ids_and_weights_custom(corrections_file, info)

                z_dim = self.shape[2] if len(self.shape) > 2 else self.z_slices_per_bed
                object_meta = ObjectMeta(dr=self.dr, shape=(self.shape[0], self.shape[1], z_dim))
                tof_meta = clinical.get_tof_meta(self.scanner_name)
                proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, detector_ids_sensitivity=detector_ids_sensitivity, weights_sensitivity=weights_sensitivity)
                
                psf_transform = GaussianFilter(self.psf_value)
                system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform], N_splits=8)
                likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term / weights)

                if self.algo == "OSEM": recon_algorithm = OSEM(likelihood)
                else:
                    prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma)
                    recon_algorithm = BSREM(likelihood, prior=prior)

                reconstruction = None
                for iter_num in range(self.n_iters):
                    if recon_thread_is_cancelled(self):
                        self.finished_signal.emit(None, None, "Cancelled")
                        return
                    reconstruction = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                    self.progress_signal.emit(iter_num + 1, f"Iteration {iter_num+1}/{self.n_iters}")

                final_img = reconstruction.cpu().numpy()
                final_img = np.flip(final_img, axis=2) 
                self.finished_signal.emit(final_img, None, "Success")
            else:
                info['moduleAxialNr'] = 5 
                info['NrRings'] = info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr'] * info['rsectorAxialNr']
                reconstructed_beds = []
                h5_files = [os.path.join(self.folder_path, f) for f in all_files if f.lower().endswith('.h5')]
                for i in range(num_beds):
                    if recon_thread_is_cancelled(self):
                        self.finished_signal.emit(None, None, "Cancelled")
                        return
                    listmode_path = list_files[i]
                    bed_identifier = os.path.splitext(os.path.basename(listmode_path))[0]
                    corr_matches = [f for f in h5_files if bed_identifier in f or f"bed_{i+1}" in f.lower()]
                    corrections_path = _resolve_corrections_h5(h5_files, corr_matches, i)
                    
                    detector_ids = get_detector_ids_custom(listmode_path, info, self.scanner_name)
                    n_events = detector_ids.shape[0]
                    weights = get_event_weights_custom(corrections_path, n_events)
                    additive_term = get_additive_term_custom(corrections_path, n_events)
                    
                    detector_ids_sensitivity, weights_sensitivity = get_sensitivity_ids_and_weights_custom(corrections_path, info)
                    object_meta = ObjectMeta(dr=self.dr, shape=(self.shape[0], self.shape[1], self.z_slices_per_bed))
                    tof_meta = clinical.get_tof_meta(self.scanner_name)
                    proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, detector_ids_sensitivity=detector_ids_sensitivity, weights_sensitivity=weights_sensitivity)
                    psf_transform = GaussianFilter(self.psf_value)
                    system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform], N_splits=8)
                    likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term/weights)
                    if self.algo == "OSEM": recon_algorithm = OSEM(likelihood)
                    else:
                        prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma)
                        recon_algorithm = BSREM(likelihood, prior=prior)
                    recon_bed = None
                    for it in range(self.n_iters):
                        if recon_thread_is_cancelled(self):
                            self.finished_signal.emit(None, None, "Cancelled")
                            return
                        recon_bed = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                        current_global_step = (i * self.n_iters) + (it + 1)
                        self.progress_signal.emit(current_global_step, f"Processing Bed {i+1}/{num_beds} - Iteration {it+1}/{self.n_iters}")
                    reconstructed_beds.append(recon_bed.cpu())
                    del system_matrix, likelihood, recon_algorithm, detector_ids, weights, additive_term
                    gc.collect()
                    torch.cuda.empty_cache()

                self.progress_signal.emit(num_beds * self.n_iters, "Stitching beds...")
                total_z = self.z_slices_per_bed * num_beds - self.overlap * (num_beds - 1)
                multibed_recon = torch.zeros((self.shape[0], self.shape[1], total_z))
                weights_stitch = (torch.arange(self.overlap, dtype=torch.float32) + 1) / (self.overlap + 1)
                weights_stitch = weights_stitch.reshape(1, 1, -1)
                for i in range(num_beds):
                    bed_idx_in_list = (num_beds - 1) - i
                    current_bed = reconstructed_beds[bed_idx_in_list]
                    mask = torch.ones_like(current_bed)
                    if i != 0: mask[:, :, 0:self.overlap] *= weights_stitch
                    if i != (num_beds - 1): mask[:, :, -self.overlap:] *= (1 - weights_stitch)
                    avg_recon = current_bed * mask
                    z_start = self.z_slices_per_bed * i - self.overlap * i
                    
                    z_end   = self.z_slices_per_bed * (i + 1) - self.overlap * i
                    
                    d_z = z_end - z_start
                    if avg_recon.shape[2] != d_z: avg_recon = avg_recon[:, :, :d_z]
                    multibed_recon[:, :, z_start:z_end] += avg_recon
                final_img = multibed_recon.numpy()
                final_img = np.flip(final_img, axis=2)
                self.finished_signal.emit(final_img, None, "Success")
        except Exception as e:
            self.finished_signal.emit(None, None, f"PET Reconstruction Error: {traceback.format_exc()}")

# --- FINAL RECONSTRUCTION THREAD ---
class LesionFinalizationThread(QThread):
    item_finished_signal = pyqtSignal(str)
    result_ready_signal = pyqtSignal(object, str, str)
    progress_signal = pyqtSignal(int, int, str)

    def __init__(self, parent, lesion_objects, listmode_file, corrections_file, psf_value, n_iters, n_subsets, 
                 recon_method, beta, gamma, save_all_iterations=False, base_name="", initial_image=None,
                 post_recon_fwhm_mm=None):
        super().__init__(parent)
        self.parent = parent
        self.lesion_objects = lesion_objects
        self.listmode_file = listmode_file
        self.corrections_file = corrections_file
        self.psf_value = psf_value
        self.n_iters = n_iters
        self.n_subsets = n_subsets
        self.recon_method = recon_method
        self.beta = beta
        self.gamma = gamma
        self.save_all_iterations = save_all_iterations
        self.base_name = base_name
        self.initial_image = initial_image
        self.post_recon_fwhm_mm = post_recon_fwhm_mm
        self._cancel_requested = False

    def run(self):
            try:
                from pytomography.metadata import ObjectMeta
                from pytomography.metadata.PET import PETLMProjMeta
                from pytomography.projectors.PET import PETLMSystemMatrix
                from pytomography.algorithms import OSEM, BSREM
                from pytomography.priors import RelativeDifferencePrior
                from pytomography.likelihoods import PoissonLogLikelihood
                from pytomography.transforms.shared import GaussianFilter
                from pytomography.io.PET import clinical
                from src.utils import add_3d_spherical_lesion, apply_library_lesion_to_volume, insert_real_lesion_array, resize_volume_preserve_morphology, apply_post_recon_gaussian_fwhm_mm
                import h5py

                # Start every queue item from a clean allocator state. Without this, the
                # CPU + CUDA caching allocators keep the previous item's reserved/fragmented
                # blocks, so a later (larger) bed can push total memory past the limit and
                # the OS hard-kills the process (no Python traceback, exit code 1).
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cuda":
                    try:
                        _dev_name = torch.cuda.get_device_name(0)
                    except Exception:
                        _dev_name = "unknown CUDA device"
                    print(f"[PET recon] Using GPU: {_dev_name} (CUDA build={torch.version.cuda})", flush=True)
                else:
                    print("[PET recon] Using CPU (torch.cuda.is_available() returned False)", flush=True)
                scanner_name = self.parent.scanner_name_input.text()
                info = clinical.get_detector_info(scanner_name)
                
                is_multibed = False
                list_files = []
                h5_files = []
                if os.path.isdir(self.listmode_file):
                    all_files = sorted(os.listdir(self.listmode_file), key=natural_keys)
                    list_files = [os.path.join(self.listmode_file, f) for f in all_files if f.lower().endswith('.blf')]
                    h5_files = [os.path.join(self.corrections_file, f) for f in all_files if f.lower().endswith('.h5')]
                    if len(list_files) > 1: is_multibed = True
                    else:
                        if list_files: list_files = [list_files[0]]
                        if h5_files: h5_files = [h5_files[0]]
                else:
                    list_files = [self.listmode_file]
                    h5_files = [self.corrections_file]
                if not list_files: raise FileNotFoundError("No .BLF files found.")
                num_beds = len(list_files)

                try:
                    dr = tuple(map(float, self.parent.object_dr_input.text().split(',')))
                    shape_xy = tuple(map(int, self.parent.object_shape_input.text().split(',')))
                    if len(shape_xy) == 3: shape_xy = shape_xy[0:2]
                    overlap = int(self.parent.pet_overlap_input.text())
                    if self.initial_image is not None and not is_multibed:
                        z_slices_per_bed = self.initial_image.shape[2]
                    else:
                        z_slices_per_bed = int(self.parent.pet_z_slices_input.text())
                except:
                    dr = (2.78, 2.78, 2.78)
                    shape_xy = (192, 192)
                    overlap = 25
                    z_slices_per_bed = 89
                
                total_z = z_slices_per_bed if not is_multibed else (z_slices_per_bed * num_beds - overlap * (num_beds - 1))
                if is_multibed:
                    info['moduleAxialNr'] = 5 
                    info['NrRings'] = info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr'] * info['rsectorAxialNr']

                display_mask_full = np.zeros((shape_xy[0], shape_xy[1], total_z), dtype=np.float32)

                for l in self.lesion_objects:
                    current_scale = getattr(l, 'size_scale', 1.0)
                    if l.custom_shape is not None:
                        display_mask_full = apply_library_lesion_to_volume(display_mask_full, l)
                    else:
                        effective_radius = l.radius * current_scale
                        display_mask_full = add_3d_spherical_lesion(display_mask_full, l.center, effective_radius, l.intensity)

                raw_mask_full = np.flip(display_mask_full, axis=2).copy()
                raw_image_full = None
                if self.initial_image is not None:
                    raw_image_full = np.flip(self.initial_image, axis=2).copy()
                else:
                    print("No initial image found. Background ablation SKIPPED.")

                reconstructed_beds = []
                # Memory-safe save_all_iterations:
                # - single-bed: emit each iteration immediately (no accumulation)
                # - multibed : maintain N_iters stitched output volumes and fold each
                #              bed/iteration into them as they finish, so peak memory is
                #              O(N_iters) full volumes instead of O(N_iters * N_beds).
                multibed_outputs_iter = None
                weights_stitch_shared = None
                if self.save_all_iterations and is_multibed:
                    multibed_outputs_iter = [
                        torch.zeros((shape_xy[0], shape_xy[1], total_z), dtype=torch.float32)
                        for _ in range(self.n_iters)
                    ]
                    weights_stitch_shared = (torch.arange(overlap, dtype=torch.float32) + 1) / (overlap + 1)
                    weights_stitch_shared = weights_stitch_shared.reshape(1, 1, -1)

                for i in range(num_beds):
                    if recon_thread_is_cancelled(self):
                        self.item_finished_signal.emit("Cancelled")
                        return
                    use_list = list_files[i]
                    if is_multibed:
                        bed_identifier = os.path.splitext(os.path.basename(use_list))[0]
                        corr_matches = [f for f in h5_files if bed_identifier in f or f"bed_{i+1}" in f.lower()]
                        use_corr = _resolve_corrections_h5(h5_files, corr_matches, i)
                        stitch_index = (num_beds - 1) - i
                        z_start_abs = stitch_index * (z_slices_per_bed - overlap)
                    else:
                        use_corr = h5_files[0]
                        z_start_abs = 0
                    
                    z_end_abs = z_start_abs + z_slices_per_bed
                    z_start_crop = max(0, z_start_abs)
                    z_end_crop = min(total_z, z_end_abs)
                    
                    local_mask_np = raw_mask_full[:, :, z_start_crop:z_end_crop]
                    local_image_np = None
                    if raw_image_full is not None:
                        local_image_np = raw_image_full[:, :, z_start_crop:z_end_crop]
                    
                    required_z = z_slices_per_bed
                    current_z = local_mask_np.shape[2]
                    if current_z < required_z:
                        diff = required_z - current_z
                        local_mask_np = np.pad(local_mask_np, ((0,0),(0,0),(0, diff)), 'constant')
                        if local_image_np is not None:
                            local_image_np = np.pad(local_image_np, ((0,0),(0,0),(0, diff)), 'constant')
                    
                    local_mask_tensor = torch.tensor(local_mask_np.copy(), dtype=torch.float32).to(device)

                    detector_ids = get_detector_ids_custom(use_list, info, scanner_name)
                    n_events = detector_ids.shape[0]
                    detector_ids_sensitivity, weights_sensitivity = get_sensitivity_ids_and_weights_custom(use_corr, info)
                    
                    additive_term = get_additive_term_custom(use_corr, n_events)
                    weights = get_event_weights_custom(use_corr, n_events)
                    
                    object_meta = ObjectMeta(dr=dr, shape=(shape_xy[0], shape_xy[1], z_slices_per_bed))
                    tof_meta = clinical.get_tof_meta(scanner_name)
                    proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, detector_ids_sensitivity=detector_ids_sensitivity, weights_sensitivity=weights_sensitivity)
                    psf_transform = GaussianFilter(self.psf_value)
                    system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform], N_splits=8)

                    # --- STOCHASTIC THINNING (Background Ablation) ---
                    if local_image_np is not None:
                        img_total_tensor = torch.tensor(local_image_np.copy(), dtype=torch.float32).to(device)
                        binary_mask = (local_mask_tensor > 0.0001).float() 
                        img_background_roi_tensor = img_total_tensor * binary_mask
                        
                        roi_signal_sum = img_background_roi_tensor.sum().item()
                        print(f"Bed {i+1}: Mask voxels={binary_mask.sum().item()}, ROI signal sum={roi_signal_sum:.2f}", flush=True)

                        if roi_signal_sum > 0:
                            expected_total = system_matrix.forward(img_total_tensor)
                            expected_roi = system_matrix.forward(img_background_roi_tensor)
                            prob_delete = expected_roi / (expected_total + 1e-9)
                            prob_delete = torch.clamp(prob_delete, 0.0, 1.0)
                            
                            print(f"Bed {i+1} Stats: Max P_delete={prob_delete.max().item():.4f}, Mean P_delete={prob_delete.mean().item():.4f}")

                            rand_vals = torch.rand_like(prob_delete)
                            keep_mask_gpu = rand_vals >= prob_delete
                            
                            keep_mask_cpu = keep_mask_gpu.cpu()
                            del rand_vals, keep_mask_gpu, prob_delete, expected_total, expected_roi
                            torch.cuda.empty_cache()

                            detector_ids, weights, additive_term = _apply_keep_mask_cpu(
                                detector_ids,
                                weights,
                                additive_term,
                                keep_mask_cpu,
                                use_legacy_numpy=(USE_LEGACY_ABLATION_MASK_PATH or not USE_LOW_MEMORY_BACKEND),
                            )
                            del keep_mask_cpu

                            n_after = detector_ids.shape[0]
                            print(f"Bed {i+1}: Ablated events. New count: {n_after}", flush=True)

                            del img_total_tensor, img_background_roi_tensor
                            # Release the pre-ablation system matrix and projector metadata before
                            # building the post-ablation ones, to avoid holding two full sets of
                            # detector-id LUTs (CPU+GPU) in memory simultaneously.
                            del system_matrix, proj_meta
                            gc.collect()
                            torch.cuda.empty_cache()

                            proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, detector_ids_sensitivity=detector_ids_sensitivity, weights_sensitivity=weights_sensitivity)
                            system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform], N_splits=8)
                        else:
                            print(f"Bed {i+1}: Lesion ROI has 0 signal. Skipping ablation.")

                    # --- PROJECTION STEP (Add Lesion) ---
                    # Always initialise chunk lists so the final concat section below
                    # can uniformly check them regardless of which branch ran.
                    if USE_LOW_MEMORY_BACKEND:
                        lesion_evt_id_chunks = []
                        lesion_evt_w_chunks = []
                        lesion_evt_add_chunks = []
                    if local_mask_tensor.sum() == 0:
                        print(f"Bed {i+1}: Lesion mask is empty. SKIPPING projection step.", flush=True)
                        listmode_lesion_events = torch.zeros((0, 3), dtype=torch.int32)
                        weights_lesion_events = torch.zeros((0), dtype=torch.float32)
                        additive_term_lesion_events = torch.zeros((0), dtype=torch.float32)
                    else:
                        try:
                            listmode_lesion_events = torch.zeros((0, 3), dtype=torch.int32)
                            weights_lesion_events = torch.zeros((0), dtype=torch.float32)
                            additive_term_lesion_events = torch.zeros((0), dtype=torch.float32)

                            # Detector IDs must be on CPU for PyTomography LUTs;
                            # weights must be on GPU for the forward projection calculation.
                            if isinstance(detector_ids_sensitivity, torch.Tensor): 
                                detector_ids_sensitivity = detector_ids_sensitivity.cpu()
                            
                            weights_sensitivity_gpu = weights_sensitivity
                            if isinstance(weights_sensitivity, torch.Tensor): 
                                weights_sensitivity_gpu = weights_sensitivity.to(device)

                            # Preallocate (n_sens, 3) int32 buffer ONCE and mutate column 2 per
                            # TOF bin, matching the behaviour of the previous working version.
                            # Re-allocating this ~2.4 GB tensor per iteration (via torch.cat)
                            # caused peak CPU RAM to exceed the physical limit and the process
                            # was OOM-killed. In-place mutation keeps peak flat across the loop.
                            n_sens = int(detector_ids_sensitivity.shape[0])
                            all_detector_ids_tof_bin_idx = torch.empty((n_sens, 3), dtype=torch.int32)
                            all_detector_ids_tof_bin_idx[:, :2] = detector_ids_sensitivity.to(torch.int32)
                            all_weights_tof_bin_idx_gpu = weights_sensitivity_gpu

                            for tof_bin_idx in range(tof_meta.num_bins):
                                # In-place fill of the TOF-bin column — zero new allocation.
                                all_detector_ids_tof_bin_idx[:, 2].fill_(int(tof_bin_idx))

                                system_matrix.proj_meta.detector_ids = all_detector_ids_tof_bin_idx
                                system_matrix.proj_meta.weights = all_weights_tof_bin_idx_gpu
                                system_matrix.scale_projection_by_sensitivity = True

                                lesion_proj_data_mean = system_matrix.forward(local_mask_tensor)
                                # Clamp small negative projector artifacts so Poisson sampling is well-defined.
                                lesion_proj_data_mean = torch.clamp(lesion_proj_data_mean, min=0.0)

                                if lesion_proj_data_mean.max() > 0:
                                    print(f"  TOF Bin {tof_bin_idx}: Max Projected Counts = {lesion_proj_data_mean.max().item():.4f}")

                                lesion_proj_data_realization = torch.poisson(lesion_proj_data_mean)
                                counts_int = lesion_proj_data_realization.cpu().to(torch.int)

                                if counts_int.sum() > 0:
                                    evt_ids = torch.repeat_interleave(all_detector_ids_tof_bin_idx, counts_int, dim=0)
                                    evt_weights = torch.repeat_interleave(all_weights_tof_bin_idx_gpu.cpu(), counts_int, dim=0)
                                    # Synthetic events carry no additive (randoms/scatter) contribution.
                                    evt_add = torch.zeros(evt_ids.shape[0], dtype=torch.float32)
                                    
                                    if USE_LOW_MEMORY_BACKEND:
                                        lesion_evt_id_chunks.append(evt_ids)
                                        lesion_evt_w_chunks.append(evt_weights)
                                        lesion_evt_add_chunks.append(evt_add)
                                    else:
                                        listmode_lesion_events = torch.concatenate([listmode_lesion_events, evt_ids], dim=0)
                                        weights_lesion_events = torch.concatenate([weights_lesion_events, evt_weights], dim=0)
                                        additive_term_lesion_events = torch.concatenate([additive_term_lesion_events, evt_add], dim=0)

                                # NOTE: do NOT delete `all_detector_ids_tof_bin_idx` — it is the
                                # preallocated shared buffer reused by the next TOF iteration.
                                del lesion_proj_data_mean, lesion_proj_data_realization, counts_int
                                gc.collect()
                                torch.cuda.empty_cache()
                            # Release the shared (n_sens, 3) buffer now that the TOF loop finished.
                            system_matrix.proj_meta.detector_ids = None
                            system_matrix.proj_meta.weights = None
                            del all_detector_ids_tof_bin_idx
                            gc.collect()
                            torch.cuda.empty_cache()
                            if USE_LOW_MEMORY_BACKEND:
                                # Defer the chunk-merge to the final concat section below.
                                # Keeping chunks in a list avoids materialising an intermediate
                                # (~1 GB for large lesions) listmode_lesion_events tensor that
                                # would otherwise coexist with both the originals and the final
                                # concatenated buffers.
                                n_les_for_print = int(sum(int(c.shape[0]) for c in lesion_evt_id_chunks)) if lesion_evt_id_chunks else 0
                                print(f"Bed {i+1}: Generated {n_les_for_print} synthetic lesion events.", flush=True)
                            else:
                                print(f"Bed {i+1}: Generated {listmode_lesion_events.shape[0]} synthetic lesion events.", flush=True)

                        except Exception as proj_e:
                            print(f"ERROR in Projection Step: {proj_e}")
                            traceback.print_exc()
                            listmode_lesion_events = torch.zeros((0, 3), dtype=torch.int32)
                            weights_lesion_events = torch.zeros((0), dtype=torch.float32)
                            additive_term_lesion_events = torch.zeros((0), dtype=torch.float32)
                            if USE_LOW_MEMORY_BACKEND:
                                # If the try-block raised before it could populate the chunks,
                                # drop anything partial so the concat path below falls through
                                # to the empty-tensor branch cleanly.
                                lesion_evt_id_chunks = []
                                lesion_evt_w_chunks = []
                                lesion_evt_add_chunks = []

                    system_matrix.proj_meta.detector_ids = detector_ids
                    system_matrix.scale_projection_by_sensitivity = False
                    system_matrix.proj_meta.weights = None

                    if USE_LOW_MEMORY_BACKEND:
                        # Sequential allocate/fill/free to avoid peaking with all six
                        # event buffers (originals + new finals) alive simultaneously.
                        # Byte-identical output to the previous single-block version: the
                        # same events end up at the same positions; only the allocation
                        # and release order differs so peak CPU RAM is lower.
                        n0 = int(detector_ids.shape[0])
                        # Prefer the (direct) chunk count; fall back to the empty-tensor
                        # placeholder (n=0) if chunks were never populated.
                        if lesion_evt_id_chunks:
                            n_les = int(sum(int(c.shape[0]) for c in lesion_evt_id_chunks))
                        else:
                            n_les = int(listmode_lesion_events.shape[0])
                        n_tot = n0 + n_les
                        det_cols = int(detector_ids.shape[1])
                        det_dtype = detector_ids.dtype
                        w_dtype = weights.dtype
                        add_dtype = additive_term.dtype

                        # --- detector_ids ---
                        detector_ids_with_lesion_events = torch.empty((n_tot, det_cols), dtype=det_dtype)
                        detector_ids_with_lesion_events[:n0] = detector_ids
                        del detector_ids
                        gc.collect()
                        if lesion_evt_id_chunks:
                            _merge_torch_chunks_no_peak_double_cat(
                                lesion_evt_id_chunks, detector_ids_with_lesion_events, n0
                            )
                            lesion_evt_id_chunks.clear()
                        elif n_les > 0:
                            detector_ids_with_lesion_events[n0:] = listmode_lesion_events
                        del listmode_lesion_events
                        gc.collect()

                        # --- weights ---
                        weights_with_lesion_events = torch.empty((n_tot,), dtype=w_dtype)
                        weights_with_lesion_events[:n0] = weights
                        del weights
                        gc.collect()
                        if lesion_evt_w_chunks:
                            _merge_torch_chunks_no_peak_double_cat(
                                lesion_evt_w_chunks, weights_with_lesion_events, n0
                            )
                            lesion_evt_w_chunks.clear()
                        elif n_les > 0:
                            weights_with_lesion_events[n0:] = weights_lesion_events
                        del weights_lesion_events
                        gc.collect()

                        # --- additive_term ---
                        additive_term_with_lesion_events = torch.empty((n_tot,), dtype=add_dtype)
                        additive_term_with_lesion_events[:n0] = additive_term
                        del additive_term
                        gc.collect()
                        if lesion_evt_add_chunks:
                            _merge_torch_chunks_no_peak_double_cat(
                                lesion_evt_add_chunks, additive_term_with_lesion_events, n0
                            )
                            lesion_evt_add_chunks.clear()
                        elif n_les > 0:
                            additive_term_with_lesion_events[n0:] = additive_term_lesion_events
                        del additive_term_lesion_events
                        gc.collect()
                    else:
                        detector_ids_with_lesion_events = torch.concatenate([detector_ids, listmode_lesion_events], dim=0)
                        weights_with_lesion_events = torch.concatenate([weights, weights_lesion_events], dim=0)
                        additive_term_with_lesion_events = torch.concatenate([additive_term, additive_term_lesion_events], dim=0)

                    # Keep legacy shuffle semantics exactly (subset composition depends on this ordering).
                    # Note: we downcast the permutation indices to int32 *after* torch.randperm has
                    # produced them, so the RNG state and the permutation values themselves are
                    # unchanged. For n < 2^31 every randperm value is exactly representable in
                    # int32, so indexing is bit-identical to the int64 version, while the index
                    # buffer uses half the RAM (saves ~0.76 GB at ~190M events).
                    idx_shuffle = torch.randperm(detector_ids_with_lesion_events.shape[0]).to(torch.int32)
                    detector_ids_with_lesion_events = detector_ids_with_lesion_events[idx_shuffle]
                    weights_with_lesion_events = weights_with_lesion_events[idx_shuffle]
                    additive_term_with_lesion_events = additive_term_with_lesion_events[idx_shuffle]
                    del idx_shuffle
                    
                    if isinstance(detector_ids_sensitivity, torch.Tensor): detector_ids_sensitivity = detector_ids_sensitivity.cpu()
                    if isinstance(weights_sensitivity, torch.Tensor): weights_sensitivity = weights_sensitivity.cpu()

                    # Release the projection-step system matrix and projector metadata before
                    # building the recon-stage ones, to avoid holding two full LUT sets at once.
                    del system_matrix, proj_meta
                    gc.collect()
                    torch.cuda.empty_cache()

                    proj_meta_recon = PETLMProjMeta(detector_ids_with_lesion_events, info, tof_meta=tof_meta, detector_ids_sensitivity=detector_ids_sensitivity, weights_sensitivity=weights_sensitivity)
                    system_matrix_recon = PETLMSystemMatrix(object_meta, proj_meta_recon, obj2obj_transforms=[psf_transform], N_splits=8)
                    likelihood = PoissonLogLikelihood(system_matrix_recon, additive_term=additive_term_with_lesion_events / weights_with_lesion_events)
                    
                    if self.recon_method == "OSEM": recon_algorithm = OSEM(likelihood)
                    else:
                        prior = RelativeDifferencePrior(beta=self.beta, gamma=self.gamma)
                        recon_algorithm = BSREM(likelihood, prior=prior)

                    recon_bed = None
                    for it in range(self.n_iters):
                        if recon_thread_is_cancelled(self):
                            self.item_finished_signal.emit("Cancelled")
                            return
                        recon_bed = recon_algorithm(n_iters=1, n_subsets=self.n_subsets)
                        global_step = (i * self.n_iters) + (it + 1)
                        total_steps = num_beds * self.n_iters
                        self.progress_signal.emit(global_step, total_steps, f"{self.base_name} (Bed {i+1})")
                        if self.save_all_iterations and not is_multibed:
                            # Stream single-bed iteration outputs immediately instead of caching all volumes.
                            iter_np = recon_bed.detach().cpu().numpy()
                            iter_np = np.flip(iter_np, axis=2)
                            iter_np = apply_post_recon_gaussian_fwhm_mm(iter_np, dr, self.post_recon_fwhm_mm)
                            self.result_ready_signal.emit(iter_np, f"{self.base_name}_iter{it+1}", "Success")
                            del iter_np
                        elif self.save_all_iterations and is_multibed:
                            # Incrementally fold this bed's iteration recon into the multibed iteration
                            # output. 'i' is the processing order (0..num_beds-1); the stitching wraps
                            # the last processed bed to the lowest z (wrap_i = num_beds - 1 - i).
                            bed_cpu = recon_bed.detach().cpu()
                            wrap_i = (num_beds - 1) - i
                            mask = torch.ones_like(bed_cpu)
                            if wrap_i != 0:
                                mask[:, :, 0:overlap] *= weights_stitch_shared
                            if wrap_i != (num_beds - 1):
                                mask[:, :, -overlap:] *= (1 - weights_stitch_shared)
                            avg_recon_it = bed_cpu * mask
                            z_start_it = z_slices_per_bed * wrap_i - overlap * wrap_i
                            z_end_it = z_slices_per_bed * (wrap_i + 1) - overlap * wrap_i
                            d_z_it = z_end_it - z_start_it
                            if avg_recon_it.shape[2] != d_z_it:
                                avg_recon_it = avg_recon_it[:, :, :d_z_it]
                            multibed_outputs_iter[it][:, :, z_start_it:z_end_it] += avg_recon_it
                            del bed_cpu, mask, avg_recon_it
                            if i == (num_beds - 1):
                                # Once the last bed contributes to this iteration, the stitched
                                # multibed volume is complete and can be streamed immediately.
                                iter_tensor = multibed_outputs_iter[it]
                                if iter_tensor is not None:
                                    iter_np = iter_tensor.numpy()
                                    multibed_outputs_iter[it] = None
                                    iter_np = np.flip(iter_np, axis=2)
                                    iter_np = apply_post_recon_gaussian_fwhm_mm(iter_np, dr, self.post_recon_fwhm_mm)
                                    self.result_ready_signal.emit(iter_np, f"{self.base_name}_iter{it+1}", "Success")
                                    del iter_np
                    # In multibed save_all_iterations we already fold recon_bed into the stitched
                    # outputs per iteration, so we don't need to keep all per-bed volumes alive.
                    if self.save_all_iterations and is_multibed:
                        reconstructed_beds.append(None)
                    else:
                        reconstructed_beds.append(recon_bed.cpu())
                    # NOTE: `system_matrix` was already deleted above before building
                    # `system_matrix_recon`; only clean up what still exists here.
                    del system_matrix_recon, likelihood, recon_algorithm, recon_bed
                    if not USE_LOW_MEMORY_BACKEND:
                        del detector_ids
                    # Explicitly drop every large working tensor so that the next bed starts
                    # with a clean RSS. Without this, Python keeps the previous bed's buffers
                    # alive until they are overwritten by the next bed's assignments, which is
                    # too late and causes RSS to creep up across beds.
                    for _name in (
                        'proj_meta_recon',
                        'detector_ids_with_lesion_events',
                        'weights_with_lesion_events',
                        'additive_term_with_lesion_events',
                        'listmode_lesion_events',
                        'weights_lesion_events',
                        'additive_term_lesion_events',
                        'detector_ids_sensitivity',
                        'weights_sensitivity',
                        'weights_sensitivity_gpu',
                        'all_detector_ids_tof_bin_idx',
                        'local_mask_np',
                        'local_image_np',
                        'local_mask_tensor',
                        'detector_ids',
                        'weights',
                        'additive_term',
                    ):
                        if _name in locals():
                            try:
                                del locals()[_name]
                            except Exception:
                                pass
                    # The `del locals()[...]` trick does not always succeed inside functions
                    # (CPython may ignore it). Explicit rebinding guarantees the reference drops.
                    proj_meta_recon = None
                    detector_ids_with_lesion_events = None
                    weights_with_lesion_events = None
                    additive_term_with_lesion_events = None
                    listmode_lesion_events = None
                    weights_lesion_events = None
                    additive_term_lesion_events = None
                    detector_ids_sensitivity = None
                    weights_sensitivity = None
                    weights_sensitivity_gpu = None
                    all_detector_ids_tof_bin_idx = None
                    local_mask_np = None
                    local_image_np = None
                    local_mask_tensor = None
                    detector_ids = None
                    weights = None
                    additive_term = None
                    gc.collect()
                    torch.cuda.empty_cache()

                # Release large numpy buffers before final stitching/filtering to minimise peak RAM.
                del raw_mask_full
                if raw_image_full is not None:
                    del raw_image_full
                gc.collect()

                if self.save_all_iterations and is_multibed:
                    for it in range(self.n_iters):
                        if multibed_outputs_iter[it] is None:
                            continue
                        iter_np = multibed_outputs_iter[it].numpy()
                        # Release the stitched tensor reference before creating the flipped/filtered copy.
                        multibed_outputs_iter[it] = None
                        iter_np = np.flip(iter_np, axis=2)
                        iter_np = apply_post_recon_gaussian_fwhm_mm(iter_np, dr, self.post_recon_fwhm_mm)
                        self.result_ready_signal.emit(iter_np, f"{self.base_name}_iter{it+1}", "Success")
                        del iter_np
                        gc.collect()
                    multibed_outputs_iter = None
                else:
                    if is_multibed:
                        multibed_recon = torch.zeros((shape_xy[0], shape_xy[1], total_z))
                        weights_stitch = (torch.arange(overlap, dtype=torch.float32) + 1) / (overlap + 1)
                        weights_stitch = weights_stitch.reshape(1, 1, -1)
                        for i in range(num_beds):
                            bed_idx_in_list = (num_beds - 1) - i
                            current_bed = reconstructed_beds[bed_idx_in_list]
                            mask = torch.ones_like(current_bed)
                            if i != 0:
                                mask[:, :, 0:overlap] *= weights_stitch
                            if i != (num_beds - 1):
                                mask[:, :, -overlap:] *= (1 - weights_stitch)
                            avg_recon = current_bed * mask
                            z_start = z_slices_per_bed * i - overlap * i
                            z_end = z_slices_per_bed * (i + 1) - overlap * i
                            d_z = z_end - z_start
                            if avg_recon.shape[2] != d_z:
                                avg_recon = avg_recon[:, :, :d_z]
                            multibed_recon[:, :, z_start:z_end] += avg_recon
                        final_np = multibed_recon.numpy()
                    else:
                        final_np = reconstructed_beds[0].numpy()
                    final_np = np.flip(final_np, axis=2)
                    final_np = apply_post_recon_gaussian_fwhm_mm(final_np, dr, self.post_recon_fwhm_mm)
                    self.result_ready_signal.emit(final_np, self.base_name, "Success")
                self.item_finished_signal.emit("Success")
            except Exception as e:
                self.item_finished_signal.emit(f"PET Finalization Error: {traceback.format_exc()}")
            finally:
                # Aggressively release this item's GPU/CPU buffers so the next queue
                # item starts with a defragmented allocator. synchronize() ensures all
                # in-flight CUDA work finishes before we drop the cached blocks.
                try:
                    reconstructed_beds = None
                except Exception:
                    pass
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def get_all_additive_terms_tof_bin_idx(self, corrections_file, tof_bin_idx, proj_meta):
        data = h5py.File(corrections_file, 'r')
        tof_bin_in_file = tof_bin_idx - proj_meta.tof_meta.num_bins // 2
        return torch.tensor(np.array(data[f'all_xtals/contam_{tof_bin_in_file}']))