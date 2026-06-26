import sys
import os
import gc
import json
import zlib
import base64
import numpy as np
import SimpleITK as sitk
import pyqtgraph as pg
import pydicom
import traceback
import glob
import time
import uuid
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QSlider,
                             QPushButton, QWidget, QLabel, QHBoxLayout, QLineEdit,
                             QStatusBar, QFileDialog, QTabWidget, QComboBox,
                             QFormLayout, QCheckBox, QGroupBox, QMessageBox,
                             QProgressDialog, QStackedWidget, QListWidget, QDialog,
                             QDialogButtonBox, QDoubleSpinBox,
                             QSplitter, QSizePolicy, QGridLayout, QRadioButton, QButtonGroup,
                             QScrollArea, QFrame)

from src.models import Lesion, ReconstructionQueueItem
from src.utils import (
    add_3d_spherical_lesion,
    apply_library_lesion_to_volume,
    insert_real_lesion_array,
    iter_number_for_recon_result,
    lesion_volume_ml_on_grid,
    lesion_truth_mean_on_grid,
    preview_sphere_volume_ml_on_grid,
    resize_volume_preserve_morphology,
    request_recon_thread_cancel,
)
from src.dialogs import MonteCarloSettingsDialog, RealLesionLibraryDialog
from src.widgets import AdvancedImageViewer, MyColorBarItem
from src.recon_pet import PETReconstructionThread, LesionFinalizationThread
from src.recon_spect import (
    SPECTReconstructionThread,
    SPECTLesionFinalizationThread,
    calculate_spect_bed_z_offsets,
)
from src.analysis_tab import DataAnalysisTab

# Reconstruction algorithm choices exposed in the UI (must match recon_pet / recon_spect support).
PET_RECON_ALGORITHMS = ("OSEM", "BSREM")
SPECT_RECON_ALGORITHMS = ("OSEM", "OSMAPOSL", "BSREM", "KEM")


class DICOMViewer(QMainWindow):
    _LESION_WORKFLOW_FORMAT = "lesion_toolbox_lesion_workflow"
    _LESION_WORKFLOW_VERSION = 1

    def __init__(self):
        super().__init__()
        self.lesion_objects = []
        self.image_array = None
        self.lesion_array = None
        self.final_array = None
        self.registered_ct_array = None
        
        # Split containers for Occurrence Maps
        self.occurrence_pet_array = None
        self.occurrence_ct_array = None
        self.occurrence_pet_spacing = (1.0, 1.0, 1.0)
        self.occurrence_ct_spacing = (1.0, 1.0, 1.0)
        
        self.sphere_center = None
        self.pet_folder_path = None
        self.ct_folder_path = None
        self.projection_file_path = None
        self.selected_modality = None
        self.is_setting_center = False
        self._temp_recon_result = None
        self.reconstruction_queue = []
        self.queue_results = []
        self.current_lesion_recon_name = ""
        self.bed_z_offsets = None
        self.current_coordinates = (0, 0, 0)
        self.color_bar_levels = (0, 1)
        self.master_lut = None
        self.mc_params = {}
        self.voxel_size_mm = None
        self.quant_mode = "Arbitrary"
        self.quantification_params = {
            "cf_to_bqml": 1.0,
            "patient_weight_kg": 70.0,
            "injected_dose_bq": 370000000.0,
            "half_life_s": 6586.2,
            "delta_t_s": 0.0
        }
        self.quant_meta_defaults = {}
        self._recon_setup_hint = None
        self.pending_real_lesion = None
        # Dialog references
        self.occur_dialog = None
        self.library_dialog = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Lesion Insertion Toolbox')
        self.resize(1600, 900)
        self.setMinimumSize(1200, 800)
        self.setGeometry(100, 100, 1600, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self._create_setup_tab()
        self._create_reconstruction_tab()
        self._create_lesion_toolkit_tab()
        self._create_rc_tab()
        self._create_about_tab()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._apply_stylesheet()
        self._modality_changed(self.modality_combo.currentText())
        self.tabs.setCurrentIndex(0)
        self.showMaximized()
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if index == 0:
            if self.selected_modality:
                self.status_bar.showMessage(f"{self.selected_modality} mode selected. Please load data.")
            else:
                self.status_bar.showMessage("Please select a modality to begin.")
        else:
            self.status_bar.clearMessage()
        if index == 3 and hasattr(self, "rc_tab") and hasattr(self.rc_tab, "refresh_sources"):
            self.rc_tab.refresh_sources()

    def _apply_stylesheet(self):
        stylesheet = """
            QMainWindow, QWidget { background-color: #2E3440; color: #D8DEE9; font-family: Arial, sans-serif; }
            QTabWidget::pane { border: 1px solid #4C566A; }
            QTabBar::tab { background: #3B4252; color: #ECEFF4; padding: 10px 8px; border-top-left-radius: 7px; border-top-right-radius: 7px; font-weight: bold; font-size: 9pt; min-width: 140px; }
            QTabBar::tab:selected { background: #88C0D0; color: #2E3440; }
            QTabBar::tab:disabled { background: #2E3440; color: #4C566A; }
            QGroupBox { border: 1px solid #4C566A; border-radius: 5px; margin-top: 1ex; padding: 15px 10px 10px 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; font-weight: bold; font-size: 12pt; }
            QLabel { font-size: 11pt; }
            QCheckBox { font-size: 11pt; color: #D8DEE9; }
            QLineEdit, QComboBox { border: 1px solid #4C566A; border-radius: 4px; padding: 6px; background-color: #434C5E; font-size: 10pt;}
            QPushButton { background-color: #5E81AC; color: #ECEFF4; font-weight: bold; border-radius: 4px; padding: 8px; border: none; font-size: 10pt;}
            QPushButton:hover { background-color: #81A1C1; }
            QPushButton:disabled { background-color: #4C566A; color: #6a758c; }
            QPushButton#realLesionLibBtn { background-color: #D08770; color: #ECEFF4; font-weight: bold; border-radius: 4px; padding: 8px; border: none; font-size: 10pt; }
            QPushButton#realLesionLibBtn:hover { background-color: #E39C87; }
            QPushButton#realLesionLibBtn:disabled { background-color: #4C566A; color: #6a758c; }
            QStatusBar { background-color: #3B4252; color: #ECEFF4; font-weight: bold; }
            QSlider::groove:horizontal { border: 1px solid #4C566A; background: #434C5E; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #D8DEE9; border: 1px solid #4C566A; width: 18px; margin: -5px 0; border-radius: 9px; }
            QListWidget { border: 1px solid #4C566A; border-radius: 4px; background-color: #434C5E; }
            QListWidget::item { padding: 5px; border-bottom: 1px solid #4C566A; }
            QListWidget::item:selected { background-color: #5E81AC; }
            QTextEdit { border: 1px solid #4C566A; border-radius: 4px; background-color: #434C5E; }
            QDialog { background-color: #2E3440; }
            QTableWidget { background-color: #434C5E; border: 1px solid #4C566A; gridline-color: #4C566A; }
            QHeaderView::section { background-color: #4C566A; padding: 4px; border: 1px solid #2E3440; font-weight: bold; }
        """
        self.setStyleSheet(stylesheet)

    def _create_setup_tab(self):
        setup_tab = QWidget()
        layout = QVBoxLayout(setup_tab)
        
        # 1. Modality Selection Group
        modality_group = QGroupBox()
        modality_group_layout = QVBoxLayout(modality_group)
        modality_group_layout.setContentsMargins(10, 5, 10, 10)
        modality_title = QLabel("<font size='+1'>1. Select Modality</font>")
        modality_title.setObjectName("groupBoxTitle")
        modality_group_layout.addWidget(modality_title)
        
        modality_content_widget = QWidget()
        modality_layout = QHBoxLayout(modality_content_widget)
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["-- Select a Modality --", "PET", "SPECT"])
        self.modality_combo.currentTextChanged.connect(self._modality_changed)
        modality_layout.addWidget(QLabel("Imaging Modality:"))
        modality_layout.addWidget(self.modality_combo)
        modality_group_layout.addWidget(modality_content_widget)
        
        # 2. Data Loading Group
        data_group = QGroupBox()
        data_group_layout = QVBoxLayout(data_group)
        data_group_layout.setContentsMargins(10, 5, 10, 10)
        data_title = QLabel("<font size='+1'>2. Load Data</font>")
        data_title.setObjectName("groupBoxTitle")
        data_group_layout.addWidget(data_title)
        
        # --- PET Data Widget ---
        self.pet_data_widget = QWidget()
        pet_data_layout = QFormLayout(self.pet_data_widget)
        pet_data_layout.setHorizontalSpacing(20)
        pet_data_layout.setVerticalSpacing(15)
        
        # PET Row 1: CT Folder
        self.load_button_pet_ct = QPushButton("Browse...")
        self.load_button_pet_ct.setFixedWidth(100)
        self.pet_ct_status = QLabel("No folder selected.")
        self.pet_ct_row = QWidget()
        pet_ct_layout = QHBoxLayout(self.pet_ct_row)
        pet_ct_layout.setContentsMargins(0,0,0,0)
        pet_ct_layout.addWidget(self.pet_ct_status)
        pet_ct_layout.addStretch()
        pet_ct_layout.addWidget(self.load_button_pet_ct)

        # PET Row 2: List Data + Correction Folder
        self.load_button_pet_folder = QPushButton("Browse...")
        self.load_button_pet_folder.setFixedWidth(100)
        self.pet_folder_status = QLabel("No folder selected.")
        self.pet_folder_row = QWidget()
        pet_folder_layout = QHBoxLayout(self.pet_folder_row)
        pet_folder_layout.setContentsMargins(0,0,0,0)
        pet_folder_layout.addWidget(self.pet_folder_status)
        pet_folder_layout.addStretch()
        pet_folder_layout.addWidget(self.load_button_pet_folder)
        
        pet_data_layout.addRow("CT Data:", self.pet_ct_row)
        pet_data_layout.addRow("List Data + Correction:", self.pet_folder_row)
        
        # --- SPECT Data Widget ---
        self.spect_data_widget = QWidget()
        spect_data_layout = QFormLayout(self.spect_data_widget)
        spect_data_layout.setHorizontalSpacing(20)
        spect_data_layout.setVerticalSpacing(15)
        
        # SPECT Row 1: CT
        self.load_button_ct = QPushButton("Browse...")
        self.load_button_ct.setFixedWidth(100)
        self.ct_status = QLabel("No folder selected.")
        self.spect_ct_row = QWidget()
        spect_ct_layout = QHBoxLayout(self.spect_ct_row)
        spect_ct_layout.setContentsMargins(0,0,0,0)
        spect_ct_layout.addWidget(self.ct_status)
        spect_ct_layout.addStretch()
        spect_ct_layout.addWidget(self.load_button_ct)
        
        # SPECT Row 2: Projections
        self.load_button_projection = QPushButton("Browse...")
        self.load_button_projection.setFixedWidth(100)
        self.projection_status = QLabel("No file/folder selected.")
        self.spect_proj_row = QWidget()
        spect_proj_layout = QHBoxLayout(self.spect_proj_row)
        spect_proj_layout.setContentsMargins(0,0,0,0)
        spect_proj_layout.addWidget(self.projection_status)
        spect_proj_layout.addStretch()
        spect_proj_layout.addWidget(self.load_button_projection)
        
        spect_data_layout.addRow("CT Folder:", self.spect_ct_row)
        spect_data_layout.addRow("Projection Data:", self.spect_proj_row)
        
        # Stacked Widget to switch between PET/SPECT
        self.data_stack = QStackedWidget()
        self.data_stack.addWidget(self.pet_data_widget)
        self.data_stack.addWidget(self.spect_data_widget)
        data_group_layout.addWidget(self.data_stack)
        data_group_layout.addStretch()
        
        # Connect Buttons
        self.load_button_pet_folder.clicked.connect(self.load_pet_folder)
        self.load_button_pet_ct.clicked.connect(self.load_pet_ct_folder)
        self.load_button_ct.clicked.connect(self.load_ct_folder)
        self.load_button_projection.clicked.connect(self.load_projection_file)
        
        # Add groups to layout
        layout.addWidget(modality_group)
        layout.addWidget(data_group)
        layout.setStretchFactor(modality_group, 0)
        layout.setStretchFactor(data_group, 1)
        
        self.tabs.addTab(setup_tab, "Setup")

    def _create_reconstruction_tab(self):
        recon_tab = QWidget()
        layout = QVBoxLayout(recon_tab)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- PET Recon Group ---
        self.pet_recon_group = QGroupBox()
        self.pet_recon_group.setObjectName("pet_recon_group")
        pet_main_layout = QVBoxLayout(self.pet_recon_group)
        pet_main_layout.setContentsMargins(10, 5, 10, 10)
        pet_title = QLabel("<font size='+1'><b>PET Reconstruction Parameters</b></font>")
        pet_title.setAlignment(Qt.AlignLeft)
        pet_main_layout.addWidget(pet_title)
        
        pet_content_widget = QWidget()
        pet_layout = QFormLayout(pet_content_widget)
        pet_layout.setVerticalSpacing(15)
        self.scanner_name_input = QLineEdit("discovery_MI")
        self.object_dr_input = QLineEdit("2.78,2.78,2.78")
        self.object_shape_input = QLineEdit("192,192")
        self.psf_input = QLineEdit("4.5")
        
        self.pet_overlap_input = QLineEdit("25")
        self.pet_z_slices_input = QLineEdit("89")
        
        self.pet_algo_combo = QComboBox()
        self.pet_algo_combo.addItems(list(PET_RECON_ALGORITHMS))
        self.pet_beta_input = QLineEdit("25")
        self.pet_gamma_input = QLineEdit("2")
        
        pet_layout.addRow("Scanner Name:", self.scanner_name_input)
        pet_layout.addRow("Object Voxel Size (mm):", self.object_dr_input)
        pet_layout.addRow("Object Shape (XY voxels):", self.object_shape_input)
        pet_layout.addRow("Slices Per Bed (Z):", self.pet_z_slices_input)
        pet_layout.addRow("Bed Overlap (slices):", self.pet_overlap_input)
        pet_layout.addRow("PSF Gaussian Filter (mm):", self.psf_input)
        pet_layout.addRow("Algorithm:", self.pet_algo_combo)
        self.pet_beta_label = QLabel("Prior Beta:")
        pet_layout.addRow(self.pet_beta_label, self.pet_beta_input)
        self.pet_gamma_label = QLabel("Prior Gamma:")
        pet_layout.addRow(self.pet_gamma_label, self.pet_gamma_input)
        pet_main_layout.addWidget(pet_content_widget)
        
        # --- SPECT Recon Group ---
        self.spect_recon_group = QGroupBox()
        self.spect_recon_group.setObjectName("spect_recon_group")
        spect_main_layout = QVBoxLayout(self.spect_recon_group)
        spect_main_layout.setContentsMargins(10, 5, 10, 10)
        spect_title = QLabel("<font size='+1'><b>SPECT Reconstruction Parameters</b></font>")
        spect_title.setAlignment(Qt.AlignLeft)
        spect_main_layout.addWidget(spect_title)
        spect_content_widget = QWidget()
        spect_content_layout = QVBoxLayout(spect_content_widget)
        scanner_params_group = QGroupBox("Scanner Information")
        scanner_form_layout = QFormLayout(scanner_params_group)
        scanner_form_layout.setVerticalSpacing(15)
        self.collimator_combo = QComboBox()
        self.collimator_combo.addItems(['SY-ME', 'SY-LEAP', 'SY-LEHR', 'SY-HE', 'SY-HE1', 'SY-HE2','GV-LEGP', 'GV-LEHR', 'GV-EEGP', 'GV-MEGP', 'GV-HEGP', 'GV-LEUR', 'GV-UEUH','GI-LEHR', 'GI-LEGP', 'GI-MEGP', 'GI-HEGP', 'GI-ELEG', 'GI-LEHX','G8-LHRS', 'G8-LEHS', 'G8-LUHR', 'G8-LEHR', 'G8-ELEG', 'G8-MEGP', 'G8-HEGP','PB-LEGP', 'PB-LEHR', 'PB-CAHR', 'PB-MEGP', 'PB-HEGP','SE-LEHS', 'SE-LEAP', 'SE-LEHR', 'SE-LEUR', 'SE-LEFB', 'SE-ME', 'SE-HE', 'SE-UHE','Custom'])
        self.energy_input = QLineEdit("208")
        self.intrinsic_resolution_input = QLineEdit("0.38")
        self.show_windows_btn = QPushButton("Get Energy Windows")
        self.index_peak_input = QLineEdit("0")
        self.index_lower_input = QLineEdit("1")
        self.index_upper_input = QLineEdit("2")
        scanner_form_layout.addRow("Collimator:", self.collimator_combo)
        scanner_form_layout.addRow("Energy (keV):", self.energy_input)
        scanner_form_layout.addRow("Intrinsic Resolution (mm):", self.intrinsic_resolution_input)
        scanner_form_layout.addRow(self.show_windows_btn)
        scanner_form_layout.addRow("Index Peak:", self.index_peak_input)
        scanner_form_layout.addRow("Index Lower (Scatter):", self.index_lower_input)
        scanner_form_layout.addRow("Index Upper (Scatter):", self.index_upper_input)
        spect_content_layout.addWidget(scanner_params_group)
        algo_select_layout = QFormLayout()
        self.spect_algo_combo = QComboBox()
        self.spect_algo_combo.addItems(list(SPECT_RECON_ALGORITHMS))
        algo_select_layout.addRow("Algorithm:", self.spect_algo_combo)
        spect_content_layout.addLayout(algo_select_layout)
        self.spect_prior_group = QGroupBox("Parameters")
        prior_params_layout = QFormLayout(self.spect_prior_group)
        prior_params_layout.setVerticalSpacing(15)
        self.spect_beta_input = QLineEdit("0.3")
        self.spect_gamma_input = QLineEdit("2")
        self.n_neighbours_input = QLineEdit("8")
        prior_params_layout.addRow("Prior Beta:", self.spect_beta_input)
        prior_params_layout.addRow("Prior Gamma:", self.spect_gamma_input)
        prior_params_layout.addRow("N Neighbours:", self.n_neighbours_input)
        spect_content_layout.addWidget(self.spect_prior_group)
        self.spect_kem_group = QGroupBox("Parameters")
        kem_params_layout = QFormLayout(self.spect_kem_group)
        kem_params_layout.setVerticalSpacing(15)
        self.support_kernel_input = QLineEdit("0.005")
        self.distance_kernel_input = QLineEdit("0.2")
        self.top_n_input = QLineEdit("40")
        self.kernel_gpu_checkbox = QCheckBox()
        self.kernel_gpu_checkbox.setChecked(True)
        kem_params_layout.addRow("Support Kernel Param:", self.support_kernel_input)
        kem_params_layout.addRow("Distance Kernel Param:", self.distance_kernel_input)
        kem_params_layout.addRow("Top N Neighbors:", self.top_n_input)
        kem_params_layout.addRow("Use GPU for Kernel:", self.kernel_gpu_checkbox)
        spect_content_layout.addWidget(self.spect_kem_group)
        spect_main_layout.addWidget(spect_content_widget)
        
        common_group = QGroupBox("Common Settings")
        common_layout = QFormLayout(common_group)
        common_layout.setVerticalSpacing(15)
        self.iterations_input = QLineEdit("2")
        self.subsets_input = QLineEdit("2")
        common_layout.addRow("Iterations:", self.iterations_input)
        common_layout.addRow("Subsets:", self.subsets_input)

        self.voxel_size_label = QLabel("Voxel Size (mm): Not available")
        common_layout.addRow(self.voxel_size_label)

        self.recon_button = QPushButton("Execute Initial Reconstruction")
        layout.addWidget(self.params_container)
        layout.addWidget(common_group)
        layout.addStretch()
        layout.addWidget(self.recon_button)
        self.tabs.addTab(recon_tab, "Reconstruction")
        self.pet_algo_combo.currentTextChanged.connect(self._update_ui_state)
        self.spect_algo_combo.currentTextChanged.connect(self._update_ui_state)
        self.show_windows_btn.clicked.connect(self.show_energy_windows)
        self.recon_button.clicked.connect(self.execute_reconstruction)

    def _create_lesion_recon_params(self, parent_layout):
        recon_params_group = QGroupBox("Reconstruction Parameters")
        common_params_layout = QFormLayout()
        self.lesion_iterations_input = QLineEdit("6")
        self.lesion_subsets_input = QLineEdit("34")
        common_params_layout.addRow("Iterations:", self.lesion_iterations_input)
        common_params_layout.addRow("Subsets:", self.lesion_subsets_input)
        self.save_all_iters_checkbox = QCheckBox("Save image after each iteration")
        common_params_layout.addRow("", self.save_all_iters_checkbox)
        self.lesion_params_stack = QStackedWidget()
        self.lesion_pet_widget = QWidget()
        pet_layout = QFormLayout(self.lesion_pet_widget)
        self.lesion_pet_algo_combo = QComboBox()
        self.lesion_pet_algo_combo.addItems(list(PET_RECON_ALGORITHMS))
        self.lesion_pet_beta_input = QLineEdit("25")
        self.lesion_pet_gamma_input = QLineEdit("2")
        pet_layout.addRow("Algorithm:", self.lesion_pet_algo_combo)
        self.lesion_pet_beta_label = QLabel("Prior Beta:")
        pet_layout.addRow(self.lesion_pet_beta_label, self.lesion_pet_beta_input)
        self.lesion_pet_gamma_label = QLabel("Prior Gamma:")
        pet_layout.addRow(self.lesion_pet_gamma_label, self.lesion_pet_gamma_input)
        self.lesion_spect_widget = QWidget()
        spect_layout = QVBoxLayout(self.lesion_spect_widget)
        spect_layout.setSpacing(8)
        spect_layout.setContentsMargins(0, 0, 0, 0)
        spect_algo_layout = QFormLayout()
        spect_algo_layout.setVerticalSpacing(10)
        self.lesion_spect_algo_combo = QComboBox()
        self.lesion_spect_algo_combo.addItems(list(SPECT_RECON_ALGORITHMS))
        spect_algo_layout.addRow("Algorithm:", self.lesion_spect_algo_combo)
        self.lesion_spect_projection_method_combo = QComboBox()
        self.lesion_spect_projection_method_combo.addItems(["Analytical", "Monte Carlo"])
        spect_algo_layout.addRow("Projection Method:", self.lesion_spect_projection_method_combo)
        spect_layout.addLayout(spect_algo_layout)
        self.lesion_mc_button_row = QWidget()
        mc_row_layout = QHBoxLayout(self.lesion_mc_button_row)
        mc_row_layout.setContentsMargins(0, 0, 0, 0)
        self.mc_config_button = QPushButton("Configure Monte Carlo...")
        self.mc_config_button.clicked.connect(self._open_mc_settings_dialog)
        mc_row_layout.addWidget(self.mc_config_button)
        spect_layout.addWidget(self.lesion_mc_button_row)
        self.lesion_spect_algo_stack = QStackedWidget()
        self.lesion_spect_algo_stack.addWidget(QWidget())  # OSEM — no extra params
        self.lesion_spect_prior_group = QGroupBox("Prior Parameters")
        prior_layout = QFormLayout(self.lesion_spect_prior_group)
        prior_layout.setVerticalSpacing(15)
        prior_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.lesion_spect_beta_input = QLineEdit("0.3")
        self.lesion_spect_gamma_input = QLineEdit("2")
        self.lesion_n_neighbours_input = QLineEdit("8")
        for prior_field in (
            self.lesion_spect_beta_input,
            self.lesion_spect_gamma_input,
            self.lesion_n_neighbours_input,
        ):
            prior_field.setMinimumHeight(28)
            prior_field.setMaximumHeight(28)
        self.lesion_spect_prior_group.setStyleSheet(
            "QLineEdit { padding: 2px 8px; border: 1px solid #4C566A; border-radius: 4px;"
            " background-color: #434C5E; color: #D8DEE9; font-size: 10pt; }"
        )
        prior_layout.addRow("Prior Beta:", self.lesion_spect_beta_input)
        prior_layout.addRow("Prior Gamma:", self.lesion_spect_gamma_input)
        prior_layout.addRow("N Neighbours:", self.lesion_n_neighbours_input)
        self.lesion_spect_algo_stack.addWidget(self.lesion_spect_prior_group)
        self.lesion_spect_kem_group = QGroupBox("KEM Parameters")
        kem_layout = QFormLayout(self.lesion_spect_kem_group)
        kem_layout.setVerticalSpacing(15)
        kem_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.lesion_support_kernel_input = QLineEdit("0.005")
        self.lesion_distance_kernel_input = QLineEdit("0.2")
        self.lesion_top_n_input = QLineEdit("40")
        for kem_field in (
            self.lesion_support_kernel_input,
            self.lesion_distance_kernel_input,
            self.lesion_top_n_input,
        ):
            kem_field.setMinimumHeight(28)
            kem_field.setMaximumHeight(28)
        self.lesion_spect_kem_group.setStyleSheet(
            "QLineEdit { padding: 2px 8px; border: 1px solid #4C566A; border-radius: 4px;"
            " background-color: #434C5E; color: #D8DEE9; font-size: 10pt; }"
        )
        self.lesion_kernel_gpu_checkbox = QCheckBox()
        self.lesion_kernel_gpu_checkbox.setChecked(True)
        kem_layout.addRow("Support Kernel Param:", self.lesion_support_kernel_input)
        kem_layout.addRow("Distance Kernel Param:", self.lesion_distance_kernel_input)
        kem_layout.addRow("Top N Neighbors:", self.lesion_top_n_input)
        kem_layout.addRow("Use GPU for Kernel:", self.lesion_kernel_gpu_checkbox)
        self.lesion_spect_algo_stack.addWidget(self.lesion_spect_kem_group)
        spect_layout.addWidget(self.lesion_spect_algo_stack)
        self.lesion_params_stack.addWidget(self.lesion_pet_widget)
        self.lesion_params_stack.addWidget(self.lesion_spect_widget)
        post_recon_widget = QWidget()
        post_recon_outer = QVBoxLayout(post_recon_widget)
        post_recon_outer.setContentsMargins(0, 0, 0, 0)
        post_recon_outer.setSpacing(6)
        self.lesion_post_recon_filter_checkbox = QCheckBox("Post-reconstruction Gaussian filter")
        post_recon_outer.addWidget(self.lesion_post_recon_filter_checkbox, 0, Qt.AlignLeft)
        self.lesion_post_recon_cutoff_row = QWidget()
        cutoff_form = QFormLayout(self.lesion_post_recon_cutoff_row)
        cutoff_form.setContentsMargins(0, 0, 0, 0)
        self.lesion_post_recon_cutoff_input = QLineEdit("3.0")
        self.lesion_post_recon_cutoff_input.setToolTip("Isotropic Gaussian FWHM in millimeters (applied after reconstruction).")
        cutoff_form.addRow("Filter cutoff (FWHM in mm):", self.lesion_post_recon_cutoff_input)
        post_recon_outer.addWidget(self.lesion_post_recon_cutoff_row)
        self.lesion_post_recon_filter_checkbox.toggled.connect(self._on_lesion_post_recon_filter_toggled)
        self._on_lesion_post_recon_filter_toggled(False)
        recon_layout = QVBoxLayout(recon_params_group)
        recon_layout.addLayout(common_params_layout)
        recon_layout.addWidget(self.lesion_params_stack)
        recon_layout.addWidget(post_recon_widget)
        self.lesion_recon_params_group = recon_params_group
        self.lesion_recon_params_pet_host = QWidget()
        self.lesion_recon_params_pet_layout = QVBoxLayout(self.lesion_recon_params_pet_host)
        self.lesion_recon_params_pet_layout.setContentsMargins(0, 0, 0, 0)
        self.lesion_recon_params_scroll_content = QWidget()
        self.lesion_recon_params_scroll_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lesion_recon_params_scroll_layout = QVBoxLayout(self.lesion_recon_params_scroll_content)
        self.lesion_recon_params_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.lesion_recon_params_scroll = QScrollArea()
        self.lesion_recon_params_scroll.setWidget(self.lesion_recon_params_scroll_content)
        self.lesion_recon_params_scroll.setWidgetResizable(True)
        self.lesion_recon_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lesion_recon_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.lesion_recon_params_scroll.setFrameShape(QFrame.NoFrame)
        self.lesion_recon_params_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.lesion_recon_params_host = QStackedWidget()
        self.lesion_recon_params_host.addWidget(self.lesion_recon_params_pet_host)
        self.lesion_recon_params_host.addWidget(self.lesion_recon_params_scroll)
        parent_layout.addWidget(self.lesion_recon_params_host)

    def _on_lesion_post_recon_filter_toggled(self, checked):
        self.lesion_post_recon_cutoff_row.setVisible(checked)
        self.lesion_post_recon_cutoff_input.setEnabled(checked)
        self._sync_tab3_spect_params_layout()

    def _open_mc_settings_dialog(self):
        dialog = MonteCarloSettingsDialog(self.mc_params, self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            if settings:
                self.mc_params = settings
                self.status_bar.showMessage("Monte Carlo settings updated.", 3000)

    def _create_lesion_toolkit_tab(self):
        self.viewer_tab = QWidget()
        main_layout = QHBoxLayout(self.viewer_tab)

        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        viewer_title_label = QLabel("<p style='font-size:11pt; font-weight:bold;'>Image Viewers</p>")
        viewer_group_box = QGroupBox("")
        viewer_group_box.setObjectName("TitledGroupBody")
        viewer_group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        viewer_group_box_layout = QVBoxLayout(viewer_group_box)
        viewer_group_box_layout.setContentsMargins(8, 8, 8, 8)
        viewer_group_box_layout.setSpacing(8)
        
        occur_btns_layout = QHBoxLayout()
        occur_btns_layout.setContentsMargins(0, 0, 0, 0)
        occur_btns_layout.setSpacing(8)
        self.load_occur_pet_btn = QPushButton("Load Occurrence Map")
        self.load_occur_ct_btn = QPushButton("Load Reference CT")
        
        self.register_occur_btn = QPushButton("Register to Patient Data")
        self.register_occur_btn.setToolTip("Rigidly registers the loaded Reference CT to the Registered CT, applying the same transform to the Occurrence Map.")
        
        occur_btns_layout.addWidget(self.load_occur_pet_btn)
        occur_btns_layout.addWidget(self.load_occur_ct_btn)
        occur_btns_layout.addWidget(self.register_occur_btn)
        occur_btns_layout.addStretch()
        
        viewer_group_box_layout.addLayout(occur_btns_layout, 0)
        viewer_group_box_layout.addSpacing(6)

        # 2. Viewers Row
        viewers_row_layout = QHBoxLayout()
        viewers_row_layout.setContentsMargins(0, 0, 0, 0)
        viewers_row_layout.setSpacing(10)
        
        self.viewer1 = AdvancedImageViewer(self, viewer_id=1)
        self.viewer2 = AdvancedImageViewer(self, viewer_id=2)
        self.viewer3 = AdvancedImageViewer(self, viewer_id=3)
        self.viewers = [self.viewer1, self.viewer2, self.viewer3]

        for viewer in self.viewers:
            viewers_row_layout.addWidget(viewer, 1)

        self.viewer1.content_combo.setCurrentText("Initial Recon")
        self.viewer2.content_combo.setCurrentText("Registered CT")
        self.viewer3.content_combo.setCurrentText("Lesion Mask")

        viewer_group_box_layout.addLayout(viewers_row_layout, 1)
        viewer_group_box_layout.addSpacing(6)

        # Compact, centered slice strip under viewers
        sliders_group = QGroupBox("Slice Controls")
        sliders_grid = QGridLayout(sliders_group)
        sliders_grid.setContentsMargins(14, 8, 14, 10)
        sliders_grid.setHorizontalSpacing(16)
        sliders_grid.setVerticalSpacing(5)
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.axial_slider = QSlider(Qt.Horizontal)
        for s in (self.sagittal_slider, self.coronal_slider, self.axial_slider):
            s.setMaximumHeight(22)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sagittal_label = QLabel("Sagittal: 1")
        self.coronal_label = QLabel("Coronal: 1")
        self.axial_label = QLabel("Axial: 1")
        for lb in (self.sagittal_label, self.coronal_label, self.axial_label):
            lb.setAlignment(Qt.AlignHCenter)
            lb.setStyleSheet("font-size: 10pt; font-weight: 600;")
        self.slice_hover_label = QLabel("Hover: --")
        self.slice_hover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.slice_hover_label.setStyleSheet("color: #D8DEE9; font-size: 9pt;")
        self.slice_hover_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sliders_grid.addWidget(self.sagittal_label, 0, 0)
        sliders_grid.addWidget(self.coronal_label, 0, 1)
        sliders_grid.addWidget(self.axial_label, 0, 2)
        sliders_grid.addWidget(self.sagittal_slider, 1, 0)
        sliders_grid.addWidget(self.coronal_slider, 1, 1)
        sliders_grid.addWidget(self.axial_slider, 1, 2)
        sliders_grid.addWidget(self.slice_hover_label, 1, 3)
        sliders_grid.setColumnStretch(0, 1)
        sliders_grid.setColumnStretch(1, 1)
        sliders_grid.setColumnStretch(2, 1)
        sliders_grid.setColumnStretch(3, 1)
        sliders_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        viewer_group_box_layout.addWidget(sliders_group, 0)

        controls_title_label = QLabel("<p style='font-size:11pt; font-weight:bold;'>Lesion Controls</p>")
        controls_group = self._create_lesion_controls_group()
        controls_group.setMinimumHeight(340)
        controls_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        top_left_widget = QWidget()
        top_left_widget.setMinimumHeight(280)
        top_left_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        top_left_layout = QVBoxLayout(top_left_widget)
        top_left_layout.setContentsMargins(0, 0, 0, 0)
        top_left_layout.setSpacing(4)
        top_left_layout.addWidget(viewer_title_label, 0, Qt.AlignBottom)
        top_left_layout.addWidget(viewer_group_box, 1)

        bottom_left_widget = QWidget()
        bottom_left_layout = QVBoxLayout(bottom_left_widget)
        bottom_left_layout.setContentsMargins(2, 2, 2, 2)
        bottom_left_layout.setSpacing(6)
        bottom_left_layout.addWidget(controls_title_label, 0, Qt.AlignBottom)
        bottom_left_layout.addWidget(controls_group, 0)
        bottom_left_layout.addStretch(1)
        bottom_left_widget.setMinimumHeight(320)
        bottom_left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_vert_splitter = QSplitter(Qt.Vertical)
        left_vert_splitter.addWidget(top_left_widget)
        left_vert_splitter.addWidget(bottom_left_widget)
        left_vert_splitter.setCollapsible(0, False)
        left_vert_splitter.setCollapsible(1, False)
        left_vert_splitter.setChildrenCollapsible(False)
        left_vert_splitter.setStretchFactor(0, 3)
        left_vert_splitter.setStretchFactor(1, 2)
        left_vert_splitter.setSizes([700, 420])
        left_vert_splitter.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        left_panel_layout.addWidget(left_vert_splitter, 1)

        right_panel_widget = QWidget()
        # Wider recon parameters panel (more readable / nicer spacing)
        right_panel_widget.setMinimumWidth(520)
        
        right_panel_layout = QVBoxLayout(right_panel_widget)

        recon_params_title_label = QLabel("<p style='font-size:11pt; font-weight:bold;'>Reconstruction Parameters</p>")
        results_title_label = QLabel("<p style='font-size:11pt; font-weight:bold;'>Results</p>")

        queue_group = self._create_queue_group()
        results_group = self._create_results_group()

        right_panel_layout.addWidget(recon_params_title_label)
        right_panel_layout.addWidget(queue_group)
        right_panel_layout.addWidget(results_title_label)
        right_panel_layout.addWidget(results_group)
        right_panel_layout.addStretch()

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel_widget)
        main_splitter.addWidget(right_panel_widget)
        
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        
        # Keep viewer dominant, but keep a consistently wider right column across resolutions
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        # Avoid hard-coded pixels: set an initial proportional split once sizes exist.
        QTimer.singleShot(0, lambda: main_splitter.setSizes([max(900, int(main_splitter.width() * 0.73)),
                                                            max(520, int(main_splitter.width() * 0.27))]))
        main_layout.addWidget(main_splitter)

        self.tabs.addTab(self.viewer_tab, "Lesion Insertion")

        self.axial_slider.valueChanged.connect(self._update_coords_from_slider)
        self.coronal_slider.valueChanged.connect(self._update_coords_from_slider)
        self.sagittal_slider.valueChanged.connect(self._update_coords_from_slider)

        for viewer in self.viewers:
            viewer.mouse_clicked_signal.connect(self._handle_viewer_click)
            viewer.view_changed_signal.connect(self.update_all_views)
            viewer.manual_levels_signal.connect(self.update_manual_levels)

        self.new_lesion_btn.clicked.connect(self.prepare_for_new_lesion)
        self.set_center_btn.clicked.connect(self.set_center_from_input)
        
        self.radius_mm_input.textChanged.connect(self._update_sphere_volume_display)
        self.radius_mm_input.textChanged.connect(self.update_all_views)
        self.size_scale_input.textChanged.connect(self._update_sphere_volume_display)
        self.object_dr_input.textChanged.connect(self._on_object_dr_input_changed)
        for le in (self.center_x_input, self.center_y_input, self.center_z_input):
            le.textChanged.connect(self._update_sphere_volume_display)

        self.save_lesion_btn.clicked.connect(self.save_current_lesion)
        
        self.save_registered_ct_btn.clicked.connect(self.save_registered_ct)
        self.save_mask_only_btn.clicked.connect(self.save_mask_only)
        self.save_initial_image_btn.clicked.connect(self.save_initial_image)
        
        self.add_to_queue_btn.clicked.connect(self.add_to_reconstruction_queue)
        
        self.clear_queue_btn.clicked.connect(self.clear_reconstruction_queue)
        self.remove_selected_btn.clicked.connect(self.remove_selected_queue_item)
        self.move_up_btn.clicked.connect(self.move_queue_item_up)
        self.move_down_btn.clicked.connect(self.move_queue_item_down)
        self.execute_queue_btn.clicked.connect(self.execute_reconstruction_queue)
        self.stop_queue_btn.clicked.connect(self.stop_reconstruction_queue)
        self.load_result_btn.clicked.connect(self.load_selected_result)
        self.save_result_btn.clicked.connect(self.save_selected_result)
        self.save_workflow_btn.clicked.connect(self.save_lesion_workflow_json)
        self.load_workflow_btn.clicked.connect(self.load_lesion_workflow_json)
        self.lesion_pet_algo_combo.currentTextChanged.connect(self._update_lesion_ui_state)
        self.lesion_spect_algo_combo.currentTextChanged.connect(self._update_lesion_ui_state)
        self.lesion_spect_projection_method_combo.currentTextChanged.connect(self._update_lesion_ui_state)

        self.load_occur_pet_btn.clicked.connect(self.load_occurrence_pet)
        self.load_occur_ct_btn.clicked.connect(self.load_occurrence_ct)
        self.register_occur_btn.clicked.connect(self.register_occurrence_maps)
        
        self._update_sphere_volume_display(self.radius_mm_input.text())
        self._sync_size_scale_visibility()

    def update_manual_levels(self, min_val, max_val):
        """
        Updates the global color scale from a manual viewer change and refreshes views.
        This allows the user to manually type in min/max values on one colorbar
        and have it apply to all other relevant viewers.
        """
        self.color_bar_levels = (min_val, max_val)
        self.update_all_views()

    def _create_lesion_controls_group(self):
        controls_group = QGroupBox("")
        controls_group.setObjectName("TitledGroupBody")
        main_controls_layout = QHBoxLayout(controls_group)
        main_controls_layout.setContentsMargins(10, 10, 10, 10)
        main_controls_layout.setSpacing(14)
        drawing_subgroup = QGroupBox("Add Lesions")
        drawing_subgroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        drawing_layout = QVBoxLayout(drawing_subgroup)
        drawing_layout.setContentsMargins(10, 10, 10, 10)
        drawing_layout.setSpacing(8)
        self.instruction_label = QLabel("Click 'Add New', then click on any Initial Recon view (Axial/Sagittal/Coronal) to set coordinates.")
        self.instruction_label.setWordWrap(True)
        
        self.new_lesion_btn = QPushButton("Add New Lesion")
        self.new_lesion_btn.setMinimumHeight(34)

        coords_widget = QWidget()
        coords_layout = QHBoxLayout(coords_widget)
        coords_layout.setContentsMargins(0,0,0,0)
        coords_layout.setSpacing(6)
        self.center_x_input = QLineEdit()
        self.center_y_input = QLineEdit()
        self.center_z_input = QLineEdit()
        for le in (self.center_x_input, self.center_y_input, self.center_z_input):
            le.setMinimumHeight(28)
        coords_layout.addWidget(QLabel("X:"))
        coords_layout.addWidget(self.center_x_input)
        coords_layout.addWidget(QLabel("Y:"))
        coords_layout.addWidget(self.center_y_input)
        coords_layout.addWidget(QLabel("Z:"))
        coords_layout.addWidget(self.center_z_input)
        self.set_center_btn = QPushButton("Set Center from Input")
        self.set_center_btn.setMinimumHeight(34)
        self.set_center_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.real_lesion_lib_btn = QPushButton("Real Lesion Library")
        self.real_lesion_lib_btn.setObjectName("realLesionLibBtn")
        self.real_lesion_lib_btn.setMinimumHeight(34)
        self.real_lesion_lib_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        center_and_library_row = QWidget()
        center_and_library_layout = QHBoxLayout(center_and_library_row)
        center_and_library_layout.setContentsMargins(0, 0, 0, 0)
        center_and_library_layout.setSpacing(8)
        center_and_library_layout.addWidget(self.set_center_btn, 1)
        center_and_library_layout.addWidget(self.real_lesion_lib_btn, 1)

        params_row_widget = QWidget()
        params_row_layout = QHBoxLayout(params_row_widget)
        params_row_layout.setContentsMargins(0, 0, 0, 0)
        params_row_layout.setSpacing(10)

        self.radius_mm_input = QLineEdit("10.0")
        self.radius_mm_input.setMinimumHeight(28)

        self.intensity_input = QLineEdit("1.0")
        self.intensity_input.setMinimumHeight(28)

        self.size_scale_input = QLineEdit("1.0")
        self.size_scale_input.setMinimumHeight(28)

        self.volume_label = QLabel("Vol (grid): -- mL")
        self.volume_label.setStyleSheet("color: #88C0D0; font-weight: bold;")
        self.volume_label.setToolTip(
            "Voxel count × reconstruction voxel size on the object grid (same as saved lesion / Tab 4)."
        )

        radius_block = QWidget()
        radius_block_layout = QHBoxLayout(radius_block)
        radius_block_layout.setContentsMargins(0, 0, 0, 0)
        radius_block_layout.setSpacing(6)
        radius_block_layout.addWidget(QLabel("Radius (mm):"))
        radius_block_layout.addWidget(self.radius_mm_input, 1)

        intensity_block = QWidget()
        intensity_block_layout = QHBoxLayout(intensity_block)
        intensity_block_layout.setContentsMargins(0, 0, 0, 0)
        intensity_block_layout.setSpacing(6)
        intensity_block_layout.addWidget(QLabel("Intensity:"))
        intensity_block_layout.addWidget(self.intensity_input, 1)

        size_block = QWidget()
        size_block_layout = QHBoxLayout(size_block)
        size_block_layout.setContentsMargins(0, 0, 0, 0)
        size_block_layout.setSpacing(6)
        size_block_layout.addWidget(QLabel("Size Scale:"))
        size_block_layout.addWidget(self.size_scale_input, 1)
        self.size_scale_block = size_block

        params_row_layout.addWidget(radius_block, 1)
        params_row_layout.addWidget(intensity_block, 1)
        params_row_layout.addWidget(self.size_scale_block, 1)
        params_row_layout.addWidget(self.volume_label)

        self.lesion_name_input = QLineEdit()
        self.lesion_name_input.setPlaceholderText("Lesion name (optional)")
        self.lesion_name_input.setMinimumHeight(28)

        self.save_lesion_btn = QPushButton("Save Current Lesion")
        self.save_lesion_btn.setMinimumHeight(34)
        self.save_lesion_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Match center row: two equal-width cells (1:1) so Save matches Real Lesion Library width.
        name_block = QWidget()
        name_block_layout = QHBoxLayout(name_block)
        name_block_layout.setContentsMargins(0, 0, 0, 0)
        name_block_layout.setSpacing(8)
        name_block_layout.addWidget(QLabel("Name:"))
        name_block_layout.addWidget(self.lesion_name_input, 1)

        save_row = QWidget()
        save_row_layout = QHBoxLayout(save_row)
        save_row_layout.setContentsMargins(0, 0, 0, 0)
        save_row_layout.setSpacing(8)
        save_row_layout.addWidget(name_block, 1)
        save_row_layout.addWidget(self.save_lesion_btn, 1)

        drawing_layout.addWidget(self.instruction_label)
        drawing_layout.addWidget(self.new_lesion_btn)
        drawing_layout.addWidget(coords_widget)
        drawing_layout.addWidget(center_and_library_row)
        
        drawing_layout.addWidget(params_row_widget)
        drawing_layout.addWidget(save_row)
        
        management_subgroup = QGroupBox("Management")
        management_subgroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        management_layout = QVBoxLayout(management_subgroup)
        management_layout.setContentsMargins(10, 10, 10, 10)
        management_layout.setSpacing(8)
        self.lesion_select_combo = QComboBox()
        self.lesion_select_combo.setMinimumHeight(28)
        self.edit_selected_btn = QPushButton("Edit Selected Lesion")
        self.edit_selected_btn.setMinimumHeight(32)
        self.remove_sphere_btn = QPushButton("Remove Selected Lesion")
        self.remove_sphere_btn.setMinimumHeight(32)
        self.save_initial_image_btn = QPushButton("Save Initial Image")
        self.save_initial_image_btn.setMinimumHeight(32)
        self.save_registered_ct_btn = QPushButton("Save Registered CT")
        self.save_registered_ct_btn.setMinimumHeight(32)
        self.save_mask_only_btn = QPushButton("Save Lesion Mask")
        self.save_mask_only_btn.setMinimumHeight(32)
        management_layout.addWidget(QLabel("Select Lesion:"))
        management_layout.addWidget(self.lesion_select_combo)
        management_layout.addWidget(self.edit_selected_btn)
        management_layout.addWidget(self.remove_sphere_btn)
        management_layout.addSpacing(8)
        management_layout.addWidget(self.save_initial_image_btn)
        management_layout.addWidget(self.save_registered_ct_btn)
        management_layout.addWidget(self.save_mask_only_btn)
        management_layout.addStretch(1)
        
        main_controls_layout.addWidget(drawing_subgroup, 4)
        main_controls_layout.addWidget(management_subgroup, 4)
        
        self.real_lesion_lib_btn.clicked.connect(self.show_real_lesion_library)
        self.remove_sphere_btn.clicked.connect(self.remove_selected_sphere)
        self.edit_selected_btn.clicked.connect(self.edit_selected_lesion)
        
        return controls_group

    def _on_object_dr_input_changed(self, *_args):
        """Keep stored spacing in sync when PET object voxel size is edited after recon."""
        if self.voxel_size_mm is not None and self.selected_modality == "PET":
            try:
                parts = [float(p.strip()) for p in self.object_dr_input.text().split(",") if p.strip()]
                if len(parts) >= 3:
                    self.voxel_size_mm = parts[:3]
                    self.voxel_size_label.setText(
                        f"Voxel Size (mm): {self.voxel_size_mm[0]:.2f}, "
                        f"{self.voxel_size_mm[1]:.2f}, {self.voxel_size_mm[2]:.2f}"
                    )
            except ValueError:
                pass
        self._update_sphere_volume_display()

    def _get_volume_preview_grid_shape(self):
        """Reconstruction grid used for sphere volume preview (matches inserted lesion grid)."""
        if self.image_array is not None:
            return self.image_array.shape
        try:
            xy = [int(x.strip()) for x in self.object_shape_input.text().split(",")[:2]]
            z = int(self.pet_z_slices_input.text())
            if len(xy) >= 2 and z > 0:
                return (xy[0], xy[1], z)
        except (ValueError, AttributeError, TypeError):
            pass
        return None

    def _resolve_lesion_center_for_preview(self, shape):
        """Center for volume preview: confirmed center, typed coords, or grid middle."""
        if self.sphere_center is not None:
            return self.sphere_center
        try:
            x = int(self.center_x_input.text()) - 1
            y = int(self.center_y_input.text()) - 1
            z = int(self.center_z_input.text()) - 1
            if (
                0 <= x < shape[0]
                and 0 <= y < shape[1]
                and 0 <= z < shape[2]
            ):
                return (x, y, z)
        except (ValueError, TypeError):
            pass
        return (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    @staticmethod
    def _normalize_sphere_lesion_size(lesion):
        """Bake size_scale into radius for spheres; library lesions keep size_scale."""
        if getattr(lesion, "custom_shape", None) is not None:
            return
        scale = float(getattr(lesion, "size_scale", 1.0))
        if abs(scale - 1.0) > 1e-6:
            lesion.radius = float(lesion.radius) * scale
            lesion.size_scale = 1.0

    def _library_lesion_pending(self):
        return getattr(self, "pending_real_lesion", None) is not None

    def _sync_size_scale_visibility(self):
        """Size scale applies to library lesions only; spheres use radius (mm) alone."""
        if hasattr(self, "size_scale_block"):
            self.size_scale_block.setVisible(self._library_lesion_pending())

    def _update_sphere_volume_display(self, _text=None):
        spacing = self.get_effective_voxel_spacing_mm()
        shape = self._get_volume_preview_grid_shape()
        if spacing is None or shape is None:
            self.volume_label.setText("Vol (grid): -- mL")
            return

        center = self._resolve_lesion_center_for_preview(shape)
        pending = getattr(self, "pending_real_lesion", None)

        try:
            if pending is not None:
                size_scale = float(self.size_scale_input.text())
                preview = type("_LibraryPreview", (), {})()
                preview.custom_shape = np.asarray(pending["array"], dtype=np.float32)
                preview.center = self._resolve_lesion_center_for_preview(shape)
                preview.radius = 0
                preview.size_scale = size_scale
                preview.intensity = 1.0
                preview.uniform_intensity = bool(pending.get("uniform_intensity", False))
                preview.use_recon_voxel_spacing = bool(
                    pending.get("use_recon_voxel_spacing", False)
                )
                preview.source_spacing_mm = pending.get("source_spacing_mm")
                vol_ml = lesion_volume_ml_on_grid(shape, preview, spacing)
            else:
                r = float(self.radius_mm_input.text())
                if r <= 0:
                    self.volume_label.setText("Vol (grid): -- mL")
                    return
                vol_ml = preview_sphere_volume_ml_on_grid(
                    r, 1.0, shape, spacing, center=center
                )
        except ValueError:
            self.volume_label.setText("Vol (grid): -- mL")
            return

        if vol_ml is None:
            self.volume_label.setText("Vol (grid): -- mL")
        else:
            self.volume_label.setText(f"Vol (grid): {vol_ml:.3f} mL")

    def _create_rc_tab(self):
        """Creates the tab for advanced interactive analysis."""
        self.rc_tab = DataAnalysisTab(self)
        self.tabs.addTab(self.rc_tab, "Data Analysis")

    def _load_medical_image(self, path):
        try:
            return sitk.ReadImage(path)
        except:
            pass
        try:
            if os.path.isfile(path):
                folder = os.path.dirname(path)
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(folder)
                if series_ids:
                    dicom_names = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
                    reader.SetFileNames(dicom_names)
                    return reader.Execute()
        except:
            pass
        raise IOError(f"Could not load image from {path}")

    def _safe_float(self, value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _parse_dicom_datetime(self, ds, date_keys, time_keys):
        date_str = None
        time_str = None
        for key in date_keys:
            if hasattr(ds, key):
                date_str = str(getattr(ds, key))
                if date_str:
                    break
        for key in time_keys:
            if hasattr(ds, key):
                time_str = str(getattr(ds, key))
                if time_str:
                    break
        if not date_str or not time_str:
            return None
        time_main = time_str.split(".")[0]
        time_main = (time_main + "000000")[:6]
        dt_str = f"{date_str}{time_main}"
        try:
            return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except Exception:
            return None

    def _extract_pet_quant_defaults_from_dataset(self, ds):
        defaults = {}
        weight = self._safe_float(getattr(ds, "PatientWeight", None))
        if weight is not None and weight > 0:
            defaults["patient_weight_kg"] = weight

        rs = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
        if rs and len(rs) > 0:
            r_item = rs[0]
            dose = self._safe_float(getattr(r_item, "RadionuclideTotalDose", None))
            half_life = self._safe_float(getattr(r_item, "RadionuclideHalfLife", None))
            if dose is not None and dose > 0:
                defaults["injected_dose_bq"] = dose
            if half_life is not None and half_life > 0:
                defaults["half_life_s"] = half_life

            inj_dt = None
            inj_dt_raw = getattr(r_item, "RadiopharmaceuticalStartDateTime", None)
            if inj_dt_raw:
                try:
                    inj_dt = datetime.strptime(str(inj_dt_raw).split(".")[0], "%Y%m%d%H%M%S")
                except Exception:
                    inj_dt = None

            if inj_dt is None:
                inj_dt = self._parse_dicom_datetime(
                    ds,
                    ["StudyDate", "SeriesDate", "AcquisitionDate"],
                    ["RadiopharmaceuticalStartTime"]
                )

            acq_dt = self._parse_dicom_datetime(
                ds,
                ["AcquisitionDate", "SeriesDate", "StudyDate"],
                ["AcquisitionTime", "SeriesTime", "StudyTime"]
            )
            if inj_dt and acq_dt:
                defaults["delta_t_s"] = max((acq_dt - inj_dt).total_seconds(), 0.0)

        slope = self._safe_float(getattr(ds, "RescaleSlope", None))
        units = str(getattr(ds, "Units", "")).upper().strip() if hasattr(ds, "Units") else ""
        if slope is not None and slope > 0 and units in {"BQML", "BQ/ML"}:
            defaults["cf_to_bqml"] = slope
        return defaults

    def _try_auto_load_pet_quant_metadata(self, folder_path):
        dicom_candidates = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
        dicom_candidates += glob.glob(os.path.join(folder_path, "**", "*.IMA"), recursive=True)
        if not dicom_candidates:
            return
        try:
            ds = pydicom.dcmread(dicom_candidates[0], stop_before_pixels=True)
            defaults = self._extract_pet_quant_defaults_from_dataset(ds)
            if defaults:
                self.quant_meta_defaults = defaults
                self.quantification_params.update(defaults)
                self.status_bar.showMessage("Loaded default PET quantification fields from DICOM header.", 5000)
        except Exception:
            pass

    def _current_unit_label(self):
        if self.quant_mode == "Bq/mL":
            return "Bq/mL"
        if self.quant_mode == "SUVbw":
            return "SUVbw"
        return "AU"

    def _arb_to_bqml_factor(self):
        return max(self.quantification_params.get("cf_to_bqml", 1.0), 0.0)

    def _suvbw_factor(self):
        cf = self._arb_to_bqml_factor()
        weight_kg = self.quantification_params.get("patient_weight_kg", 0.0)
        injected = self.quantification_params.get("injected_dose_bq", 0.0)
        half_life = self.quantification_params.get("half_life_s", 0.0)
        delta_t = self.quantification_params.get("delta_t_s", 0.0)
        if cf <= 0 or injected <= 0 or weight_kg <= 0:
            return None
        decay_corrected_dose = injected
        if half_life > 0 and delta_t > 0:
            decay_corrected_dose = injected * np.exp(-np.log(2.0) * (delta_t / half_life))
        if decay_corrected_dose <= 0:
            return None
        return cf * ((weight_kg * 1000.0) / decay_corrected_dose)

    def _unit_factor_for_mode(self):
        if self.quant_mode == "Bq/mL":
            return self._arb_to_bqml_factor()
        if self.quant_mode == "SUVbw":
            suv_factor = self._suvbw_factor()
            return suv_factor if suv_factor is not None else 1.0
        return 1.0

    def _convert_array_to_selected_unit(self, arr, content_key):
        if arr is None:
            return None
        if self.selected_modality != "PET":
            return arr
        if content_key not in {"initial", "final"}:
            return arr
        factor = self._unit_factor_for_mode()
        if factor == 1.0:
            return arr
        return (arr * factor).astype(np.float32)

    def _factor_for_save_quant_mode(self, save_mode):
        """Scale factor from stored PET recon (AU) to save_mode. save_mode: Arbitrary | Bq/mL | SUVbw."""
        if save_mode == "Bq/mL":
            return self._arb_to_bqml_factor()
        if save_mode == "SUVbw":
            suv = self._suvbw_factor()
            return suv if suv is not None else None
        return 1.0

    def _convert_pet_volume_for_save_unit(self, arr, save_mode):
        if arr is None:
            return None
        factor = self._factor_for_save_quant_mode(save_mode)
        if factor is None:
            return None
        if factor == 1.0:
            return np.asarray(arr, dtype=np.float32)
        return (arr * factor).astype(np.float32)

    def _save_unit_display_label(self, save_mode):
        if save_mode == "Bq/mL":
            return "Bq/mL"
        if save_mode == "SUVbw":
            return "SUVbw"
        return "AU"

    def _prompt_pet_save_unit_mode(self):
        """
        Ask which units to write for PET NIfTI export (AU / SUVbw / Bq/mL).
        Returns save mode string or None if cancelled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Save — output units")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Save image values as:"))

        rb_au = QRadioButton("AU (arbitrary units, as reconstructed)")
        rb_suv = QRadioButton("SUV (SUVbw)")
        rb_bq = QRadioButton("Bq/mL")

        group = QButtonGroup(dialog)
        group.addButton(rb_au)
        group.addButton(rb_suv)
        group.addButton(rb_bq)

        if self.quant_mode == "Bq/mL":
            rb_bq.setChecked(True)
        elif self.quant_mode == "SUVbw":
            rb_suv.setChecked(True)
        else:
            rb_au.setChecked(True)

        layout.addWidget(rb_au)
        layout.addWidget(rb_suv)
        layout.addWidget(rb_bq)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None

        if rb_bq.isChecked():
            return "Bq/mL"
        if rb_suv.isChecked():
            if self._suvbw_factor() is None:
                QMessageBox.warning(
                    self,
                    "Cannot save as SUV",
                    "SUVbw needs a positive calibration factor (AU→Bq/mL), patient weight, and injected dose.\n"
                    "Use the unit conversion dialog to set these, or choose AU / Bq/mL.",
                )
                return None
            return "SUVbw"
        return "Arbitrary"

    def format_intensity_value_for_display(self, raw_value, content_type):
        if self.selected_modality != "PET" or content_type not in {"Initial Recon", "Final Recon"}:
            return raw_value, "AU"
        # Viewer already receives arrays scaled by _convert_array_to_selected_unit; do not multiply again.
        return raw_value, self._current_unit_label()

    def open_unit_conversion_dialog(self):
        if self.image_array is None:
            QMessageBox.warning(self, "No Image", "Please reconstruct or load an image before unit conversion.")
            return
        if self.selected_modality != "PET":
            QMessageBox.information(self, "Not Available", "Unit conversion to Bq/mL and SUVbw is only enabled for PET.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Image Unit Conversion")
        dialog.setMinimumWidth(430)
        layout = QFormLayout(dialog)

        unit_combo = QComboBox()
        unit_combo.addItems(["Arbitrary", "Bq/mL", "SUVbw"])
        unit_combo.setCurrentText(self.quant_mode)

        cf_spin = QDoubleSpinBox()
        cf_spin.setDecimals(4)
        cf_spin.setRange(0.0, 1e9)
        cf_spin.setSingleStep(0.01)
        cf_spin.setGroupSeparatorShown(True)
        cf_spin.setValue(float(self.quantification_params.get("cf_to_bqml", 1.0)))

        weight_spin = QDoubleSpinBox()
        weight_spin.setDecimals(1)
        weight_spin.setRange(0.0, 500.0)
        weight_spin.setSingleStep(0.1)
        weight_spin.setSuffix(" kg")
        weight_spin.setValue(float(self.quantification_params.get("patient_weight_kg", 70.0)))

        dose_spin = QDoubleSpinBox()
        dose_spin.setDecimals(0)
        dose_spin.setRange(0.0, 1e12)
        dose_spin.setSingleStep(1e6)
        dose_spin.setGroupSeparatorShown(True)
        dose_spin.setSuffix(" Bq")
        dose_spin.setValue(float(self.quantification_params.get("injected_dose_bq", 370000000.0)))

        half_life_spin = QDoubleSpinBox()
        half_life_spin.setDecimals(1)
        half_life_spin.setRange(0.0, 1e9)
        half_life_spin.setSingleStep(1.0)
        half_life_spin.setGroupSeparatorShown(True)
        half_life_spin.setSuffix(" s")
        half_life_spin.setValue(float(self.quantification_params.get("half_life_s", 6586.2)))

        delay_spin = QDoubleSpinBox()
        delay_spin.setDecimals(0)
        delay_spin.setRange(0.0, 1e7)
        delay_spin.setSingleStep(1.0)
        delay_spin.setGroupSeparatorShown(True)
        delay_spin.setSuffix(" s")
        delay_spin.setValue(float(self.quantification_params.get("delta_t_s", 0.0)))

        import_defaults_btn = QPushButton("Load Defaults from Header")
        import_defaults_btn.setStyleSheet("background-color: #8FBCBB; color: #2E3440;")

        def load_defaults():
            if not self.quant_meta_defaults and self.pet_folder_path:
                self._try_auto_load_pet_quant_metadata(self.pet_folder_path)
            if self.quant_meta_defaults:
                cf_spin.setValue(float(self.quant_meta_defaults.get("cf_to_bqml", cf_spin.value())))
                weight_spin.setValue(float(self.quant_meta_defaults.get("patient_weight_kg", weight_spin.value())))
                dose_spin.setValue(float(self.quant_meta_defaults.get("injected_dose_bq", dose_spin.value())))
                half_life_spin.setValue(float(self.quant_meta_defaults.get("half_life_s", half_life_spin.value())))
                delay_spin.setValue(float(self.quant_meta_defaults.get("delta_t_s", delay_spin.value())))
            else:
                QMessageBox.information(dialog, "No Header Defaults", "No usable PET quantification fields were found in available DICOM headers.")

        import_defaults_btn.clicked.connect(load_defaults)

        layout.addRow("Display Units:", unit_combo)
        layout.addRow("Calibration factor (AU -> Bq/mL):", cf_spin)
        layout.addRow("Patient weight (kg):", weight_spin)
        layout.addRow("Injected dose (Bq):", dose_spin)
        layout.addRow("Radionuclide half-life (s):", half_life_spin)
        layout.addRow("Injection -> acquisition delay (s):", delay_spin)
        layout.addRow("", import_defaults_btn)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        updated_params = {
            "cf_to_bqml": cf_spin.value(),
            "patient_weight_kg": weight_spin.value(),
            "injected_dose_bq": dose_spin.value(),
            "half_life_s": half_life_spin.value(),
            "delta_t_s": delay_spin.value()
        }
        requested_mode = unit_combo.currentText()

        self.quantification_params.update(updated_params)
        if requested_mode == "SUVbw" and self._suvbw_factor() is None:
            QMessageBox.warning(self, "Invalid SUV Inputs", "SUVbw conversion needs positive calibration factor, patient weight, and injected dose.")
            return

        self.quant_mode = requested_mode
        for viewer in getattr(self, "viewers", []):
            # Force a full levels reset on unit change so ranges update immediately.
            viewer.last_fg_stats = None
            viewer.last_applied_content_type = None
            viewer.fg_data_ref = None
            if hasattr(viewer, "force_level_reset"):
                viewer.force_level_reset = True
        self._update_global_color_scale()
        self.update_all_views()
        # Ensure histogram dialog reflects new unit range for PET recon viewers only.
        for viewer in getattr(self, "viewers", []):
            if not hasattr(viewer, "hist_widget"):
                continue
            if getattr(viewer, "content_type", None) not in {"Initial Recon", "Final Recon"}:
                continue
            d_min, d_max = getattr(viewer, "fg_stats", (0.0, 1.0))
            if d_max <= d_min:
                d_max = d_min + 1.0
            viewer.hist_widget.setLevels(d_min, d_max)
            viewer.hist_widget.setDefaultLevels(d_min, d_max)
        self.status_bar.showMessage(f"Display unit updated to {self.quant_mode}.", 5000)

    def load_occurrence_pet(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Occurrence Map", "", "Images (*.nii *.nii.gz *.dcm *.ima);;All Files (*)")
        if not path: return
        try:
            sitk_img = self._load_medical_image(path)
            self.occurrence_pet_spacing = sitk_img.GetSpacing()
            arr = sitk.GetArrayFromImage(sitk_img)
            self.occurrence_pet_array = np.transpose(arr, (2, 1, 0)).astype(np.float32)
            
            if self.image_array is None:
                dims = self.occurrence_pet_array.shape
                self.sagittal_slider.setMaximum(dims[0]-1)
                self.coronal_slider.setMaximum(dims[1]-1)
                self.axial_slider.setMaximum(dims[2]-1)
                self.current_coordinates = (dims[0]//2, dims[1]//2, dims[2]//2)
            
            for v in self.viewers:
                if v.content_combo.currentText() == "Occurrence Map" or v.content_combo.currentText() == "None":
                     v.content_combo.setCurrentText("Occurrence Map")
                     break
            
            self.update_all_views()
            self.status_bar.showMessage(f"Loaded Occurrence Map: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load map: {e}")

    def load_occurrence_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference CT", "", "Images (*.nii *.nii.gz *.dcm *.ima);;All Files (*)")
        if not path: return
        try:
            sitk_img = self._load_medical_image(path)
            self.occurrence_ct_spacing = sitk_img.GetSpacing()
            arr = sitk.GetArrayFromImage(sitk_img)
            self.occurrence_ct_array = np.transpose(arr, (2, 1, 0)).astype(np.float32)
            
            if self.image_array is None and self.occurrence_pet_array is None:
                dims = self.occurrence_ct_array.shape
                self.sagittal_slider.setMaximum(dims[0]-1)
                self.coronal_slider.setMaximum(dims[1]-1)
                self.axial_slider.setMaximum(dims[2]-1)
                self.current_coordinates = (dims[0]//2, dims[1]//2, dims[2]//2)

            self.update_all_views()
            self.status_bar.showMessage(f"Loaded Reference CT: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Reference CT: {e}")

    def _sitk_image_centered_on_object_grid(self, arr_zyx, spacing_mm):
        """
        SimpleITK image with spacing and origin centered on the volume — same
        convention as PET recon / registered CT (object grid center at 0 mm).
        arr_zyx: numpy array in (z, y, x) order for sitk.GetImageFromArray.
        """
        image = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
        if spacing_mm is None:
            spacing = (1.0, 1.0, 1.0)
        else:
            parts = [float(s) for s in spacing_mm[:3]]
            while len(parts) < 3:
                parts.append(parts[-1] if parts else 1.0)
            spacing = tuple(parts[:3])
        image.SetSpacing(spacing)
        sz = image.GetSize()
        sp = image.GetSpacing()
        image.SetOrigin([-(sz[i] * sp[i]) / 2.0 for i in range(3)])
        return image

    def register_occurrence_maps(self):
        if self.registered_ct_array is None:
            QMessageBox.warning(self, "No Registered CT", "Please perform a reconstruction and CT registration first.")
            return
        if self.occurrence_ct_array is None:
            QMessageBox.warning(self, "No Reference CT", "Please load a Reference CT map to perform registration.")
            return

        try:
            self.status_bar.showMessage("Registering Data... Please wait.")
            QApplication.processEvents()

            fixed_arr = np.transpose(self.registered_ct_array, (2, 1, 0))
            recon_spacing = self.voxel_size_mm if self.voxel_size_mm else (1.0, 1.0, 1.0)
            fixed_image = self._sitk_image_centered_on_object_grid(fixed_arr, recon_spacing)

            moving_arr = np.transpose(self.occurrence_ct_array, (2, 1, 0))
            moving_image = self._sitk_image_centered_on_object_grid(
                moving_arr, self.occurrence_ct_spacing
            )

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0,
                minStep=1e-4,
                numberOfIterations=200,
                relaxationFactor=0.5,
                gradientMagnitudeTolerance=1e-8
            )
            R.SetOptimizerScalesFromPhysicalShift()

            # TranslationTransform has no rotation center — align volume centers in mm
            # (centered object-grid origin makes both geometric centers ≈ 0 mm).
            fixed_center_idx = [(s - 1) / 2.0 for s in fixed_image.GetSize()]
            moving_center_idx = [(s - 1) / 2.0 for s in moving_image.GetSize()]
            fixed_phys_center = fixed_image.TransformContinuousIndexToPhysicalPoint(fixed_center_idx)
            moving_phys_center = moving_image.TransformContinuousIndexToPhysicalPoint(moving_center_idx)
            translation_vector = [moving_phys_center[i] - fixed_phys_center[i] for i in range(3)]
            initial_transform = sitk.TranslationTransform(3, translation_vector)

            R.SetInitialTransform(initial_transform, inPlace=False)
            R.SetInterpolator(sitk.sitkLinear)
            final_transform = R.Execute(fixed_image, moving_image)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(float(np.min(self.occurrence_ct_array))) 
            resampler.SetTransform(final_transform)

            out_ct = resampler.Execute(moving_image)
            self.occurrence_ct_array = np.transpose(sitk.GetArrayFromImage(out_ct), (2, 1, 0)).astype(np.float32)

            if self.occurrence_pet_array is not None:
                pet_arr = np.transpose(self.occurrence_pet_array, (2, 1, 0))
                pet_img = self._sitk_image_centered_on_object_grid(
                    pet_arr, self.occurrence_pet_spacing
                )
                resampler.SetDefaultPixelValue(0.0)
                out_pet = resampler.Execute(pet_img)
                self.occurrence_pet_array = np.transpose(sitk.GetArrayFromImage(out_pet), (2, 1, 0)).astype(np.float32)
                self.status_bar.showMessage(f"Registration Complete. Final metric: {R.GetMetricValue():.4f}")
            else:
                self.status_bar.showMessage(f"CT Registration Complete. Final metric: {R.GetMetricValue():.4f}")

            self.update_all_views()

        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"An error occurred during registration:\n{e}\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("Registration Failed.")

    def _create_queue_group(self):
        queue_group = QGroupBox("")
        queue_layout = QVBoxLayout(queue_group)
        self._create_lesion_recon_params(queue_layout)
        queue_controls_layout = QHBoxLayout()
        self.queue_name_input = QLineEdit()
        self.queue_name_input.setPlaceholderText("Enter reconstruction name...")
        self.add_to_queue_btn = QPushButton("Add to Queue")
        self.clear_queue_btn = QPushButton("Clear Queue")
        queue_controls_layout.addWidget(QLabel("Name:"))
        queue_controls_layout.addWidget(self.queue_name_input)
        queue_controls_layout.addWidget(self.add_to_queue_btn)
        queue_controls_layout.addWidget(self.clear_queue_btn)
        self.queue_list = QListWidget()
        queue_manage_layout = QHBoxLayout()
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.move_up_btn = QPushButton("Move Up")
        self.move_down_btn = QPushButton("Move Down")
        queue_manage_layout.addWidget(self.remove_selected_btn)
        queue_manage_layout.addWidget(self.move_up_btn)
        queue_manage_layout.addWidget(self.move_down_btn)
        queue_exec_layout = QHBoxLayout()
        self.execute_queue_btn = QPushButton("Execute Queue")
        self.stop_queue_btn = QPushButton("Stop Queue")
        self.stop_queue_btn.setEnabled(False)
        queue_exec_layout.addWidget(self.execute_queue_btn)
        queue_exec_layout.addWidget(self.stop_queue_btn)
        queue_layout.addLayout(queue_controls_layout)
        queue_layout.addWidget(self.queue_list)
        queue_layout.addLayout(queue_manage_layout)
        queue_layout.addLayout(queue_exec_layout)
        # Tab 3 only: queue list expands; recon params keep their natural height.
        queue_layout.setStretch(0, 0)
        queue_layout.setStretch(2, 1)
        return queue_group

    def _create_results_group(self):
        results_group = QGroupBox("")
        results_layout = QVBoxLayout(results_group)
        self.results_list = QListWidget()
        self.results_list.setMaximumHeight(120)
        self.load_result_btn = QPushButton("Load Selected Result")
        self.save_result_btn = QPushButton("Save Selected Result")
        self.save_workflow_btn = QPushButton("Save Workflow (JSON)")
        self.load_workflow_btn = QPushButton("Load Workflow (JSON)")
        results_layout.addWidget(self.results_list)
        results_btn_layout = QHBoxLayout()
        results_btn_layout.addWidget(self.load_result_btn)
        results_btn_layout.addWidget(self.save_result_btn)
        results_layout.addLayout(results_btn_layout)
        workflow_btn_layout = QHBoxLayout()
        workflow_btn_layout.addWidget(self.save_workflow_btn)
        workflow_btn_layout.addWidget(self.load_workflow_btn)
        results_layout.addLayout(workflow_btn_layout)
        return results_group

    def _sanitize_for_json(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            x = float(obj)
            return x if np.isfinite(x) else None
        if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        return str(obj)

    def _encode_array_for_json(self, arr):
        a = np.asarray(arr, dtype=np.float32, order="C")
        raw = a.tobytes()
        comp = zlib.compress(raw, level=6)
        return {
            "shape": [int(x) for x in a.shape],
            "dtype": "float32",
            "encoding": "base64+zlib",
            "data": base64.b64encode(comp).decode("ascii"),
        }

    def _decode_array_from_json(self, payload):
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("Invalid array payload.")
        if payload.get("encoding") != "base64+zlib":
            raise ValueError("Unsupported array encoding.")
        shape = tuple(int(x) for x in payload.get("shape", []))
        if len(shape) != 3:
            raise ValueError("Only 3D arrays are supported.")
        comp = base64.b64decode(str(payload.get("data", "")).encode("ascii"))
        raw = zlib.decompress(comp)
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size != int(np.prod(shape)):
            raise ValueError("Decoded array size mismatch.")
        return arr.reshape(shape).copy()

    def _serialize_lesions_for_workflow(self):
        out = []
        for i, lesion in enumerate(self.lesion_objects or []):
            entry = {
                "index": i,
                "id": str(getattr(lesion, "id", "")),
                "name": str(getattr(lesion, "name", f"Lesion_{i+1}")),
                "center": [int(v) for v in getattr(lesion, "center", (0, 0, 0))[:3]],
                "radius": float(getattr(lesion, "radius", 0.0)),
                "intensity": float(getattr(lesion, "intensity", 0.0)),
                "size_scale": float(getattr(lesion, "size_scale", 1.0)),
                "uniform_intensity": bool(getattr(lesion, "uniform_intensity", False)),
                "use_recon_voxel_spacing": bool(
                    getattr(lesion, "use_recon_voxel_spacing", False)
                ),
                "source_spacing_mm": (
                    [float(s) for s in getattr(lesion, "source_spacing_mm", None)[:3]]
                    if getattr(lesion, "source_spacing_mm", None) is not None
                    else None
                ),
                "custom_shape": None,
            }
            custom_shape = getattr(lesion, "custom_shape", None)
            if custom_shape is not None:
                entry["custom_shape"] = self._encode_array_for_json(custom_shape)
                entry["custom_shape_is_source"] = bool(
                    getattr(lesion, "custom_shape_is_source", True)
                )
            out.append(entry)
        return out

    def _serialize_bed_z_offsets_for_workflow(self):
        """JSON-safe {bed_index: z_offset_voxels} for multibed SPECT workflow saves."""
        offsets = self.bed_z_offsets
        if offsets is None:
            return None
        if isinstance(offsets, dict):
            items = sorted(offsets.items(), key=lambda kv: int(kv[0]))
            return self._sanitize_for_json({str(int(k)): int(v) for k, v in items})
        if isinstance(offsets, (list, tuple)):
            return self._sanitize_for_json({str(i): int(v) for i, v in enumerate(offsets)})
        return None

    def _deserialize_bed_z_offsets_from_workflow(self, raw):
        """Restore {bed_index: z_offset_voxels}; detect legacy broken list(keys) saves."""
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {int(k): int(v) for k, v in raw.items()}
        if isinstance(raw, list):
            # Old bug: list({0: z0, 1: z1, ...}) saved only keys [0, 1, 2, ...].
            if len(raw) > 0 and all(int(raw[i]) == i for i in range(len(raw))):
                return None
            return {i: int(v) for i, v in enumerate(raw)}
        return None

    def _try_restore_bed_z_offsets_from_projections(self):
        """Recompute multibed z offsets from projection DICOMs when workflow JSON lacks them."""
        if self.selected_modality != "SPECT":
            return
        path = self.projection_file_path
        if not path or not os.path.isdir(path):
            return
        try:
            files_nm = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".dcm", ".ima"))
            )
            if len(files_nm) < 2:
                return
            self.bed_z_offsets = calculate_spect_bed_z_offsets(files_nm)
        except Exception:
            pass

    def _restore_lesions_from_workflow(self, lesion_defs):
        restored = []
        skipped = 0
        for i, d in enumerate(lesion_defs or []):
            try:
                center = tuple(int(v) for v in (d.get("center") or [0, 0, 0])[:3])
                while len(center) < 3:
                    center = center + (0,)
                custom_shape = self._decode_array_from_json(d.get("custom_shape")) if d.get("custom_shape") else None
                src = d.get("source_spacing_mm")
                lesion = Lesion(
                    center,
                    float(d.get("radius", 0.0)),
                    float(d.get("intensity", 0.0)),
                    custom_shape=custom_shape,
                    size_scale=float(d.get("size_scale", 1.0)),
                    uniform_intensity=bool(d.get("uniform_intensity", False)),
                    use_recon_voxel_spacing=bool(d.get("use_recon_voxel_spacing", False)),
                    source_spacing_mm=src,
                )
                lesion.name = str(d.get("name", f"Lesion_{i+1}"))
                lesion.id = str(d.get("id", "")) or f"workflow_{i+1}"
                if custom_shape is not None:
                    lesion.custom_shape_is_source = bool(d.get("custom_shape_is_source", True))
                else:
                    self._normalize_sphere_lesion_size(lesion)
                restored.append(lesion)
            except Exception:
                skipped += 1
        self.lesion_objects = restored
        return len(restored), skipped

    def _serialize_queue_item_for_workflow(self, item):
        return {
            "name": str(getattr(item, "name", "")),
            "modality": getattr(item, "modality", None),
            "algo": getattr(item, "algo", None),
            "n_iters": getattr(item, "n_iters", None),
            "n_subsets": getattr(item, "n_subsets", None),
            "projection_method": getattr(item, "projection_method", "ANALYTICAL"),
            "save_all_iterations": bool(getattr(item, "save_all_iterations", False)),
            "iter_number": getattr(item, "iter_number", None),
            "params": self._sanitize_for_json(getattr(item, "params", {}) or {}),
            "status": str(getattr(item, "status", "Queued")),
            "result": self._encode_array_for_json(item.result) if getattr(item, "result", None) is not None else None,
        }

    def _deserialize_queue_item_from_workflow(self, data):
        params = dict(data.get("params") or {})
        item = ReconstructionQueueItem(
            name=str(data.get("name", "")),
            modality=data.get("modality"),
            algo=data.get("algo"),
            n_iters=data.get("n_iters"),
            n_subsets=data.get("n_subsets"),
            projection_method=data.get("projection_method", "ANALYTICAL"),
            save_all_iterations=bool(data.get("save_all_iterations", False)),
            **params,
        )
        item.status = str(data.get("status", "Queued"))
        iter_number = data.get("iter_number")
        item.iter_number = int(iter_number) if iter_number is not None else None
        result_payload = data.get("result")
        item.result = self._decode_array_from_json(result_payload) if result_payload else None
        if item.result is not None and item.iter_number is None:
            item.iter_number = iter_number_for_recon_result(
                item.name, n_iters=item.n_iters, save_all_iterations=item.save_all_iterations
            )
        return item

    def _refresh_results_list(self):
        self.results_list.clear()
        for result_item in self.queue_results:
            status = str(getattr(result_item, "status", "Queued"))
            prefix = "✅" if status == "Completed" else "❌"
            self.results_list.addItem(f"{prefix} {result_item.name} - {status}")

    def _collect_lesion_workflow_ui_state(self):
        viewer_modes = [viewer.content_combo.currentText() for viewer in getattr(self, "viewers", [])]
        return {
            "queue_name": self.queue_name_input.text(),
            "lesion_iterations": self.lesion_iterations_input.text(),
            "lesion_subsets": self.lesion_subsets_input.text(),
            "save_all_iterations": self.save_all_iters_checkbox.isChecked(),
            "lesion_pet_algo": self.lesion_pet_algo_combo.currentText(),
            "lesion_pet_beta": self.lesion_pet_beta_input.text(),
            "lesion_pet_gamma": self.lesion_pet_gamma_input.text(),
            "lesion_spect_algo": self.lesion_spect_algo_combo.currentText(),
            "lesion_spect_projection_method": self.lesion_spect_projection_method_combo.currentText(),
            "lesion_spect_beta": self.lesion_spect_beta_input.text(),
            "lesion_spect_gamma": self.lesion_spect_gamma_input.text(),
            "lesion_n_neighbours": self.lesion_n_neighbours_input.text(),
            "lesion_support_kernel": self.lesion_support_kernel_input.text(),
            "lesion_distance_kernel": self.lesion_distance_kernel_input.text(),
            "lesion_top_n": self.lesion_top_n_input.text(),
            "kernel_gpu": self.kernel_gpu_checkbox.isChecked(),
            "post_recon_enabled": self.lesion_post_recon_filter_checkbox.isChecked(),
            "post_recon_cutoff": self.lesion_post_recon_cutoff_input.text(),
            "center_x": self.center_x_input.text(),
            "center_y": self.center_y_input.text(),
            "center_z": self.center_z_input.text(),
            "radius_mm": self.radius_mm_input.text(),
            "intensity": self.intensity_input.text(),
            "size_scale": self.size_scale_input.text(),
            "lesion_name": self.lesion_name_input.text(),
            "selected_lesion_index": self.lesion_select_combo.currentIndex(),
            "selected_queue_index": self.queue_list.currentRow(),
            "selected_result_index": self.results_list.currentRow(),
            "viewer_modes": viewer_modes,
            "slider_values": {
                "sagittal": self.sagittal_slider.value(),
                "coronal": self.coronal_slider.value(),
                "axial": self.axial_slider.value(),
            },
        }

    def _restore_lesion_workflow_ui_state(self, state):
        state = state or {}
        self.queue_name_input.setText(str(state.get("queue_name", "")))
        self.lesion_iterations_input.setText(str(state.get("lesion_iterations", self.lesion_iterations_input.text())))
        self.lesion_subsets_input.setText(str(state.get("lesion_subsets", self.lesion_subsets_input.text())))
        self.save_all_iters_checkbox.setChecked(bool(state.get("save_all_iterations", False)))

        for combo, key in (
            (self.lesion_pet_algo_combo, "lesion_pet_algo"),
            (self.lesion_spect_algo_combo, "lesion_spect_algo"),
            (self.lesion_spect_projection_method_combo, "lesion_spect_projection_method"),
        ):
            txt = state.get(key)
            if txt is not None:
                idx = combo.findText(str(txt))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        self.lesion_pet_beta_input.setText(str(state.get("lesion_pet_beta", self.lesion_pet_beta_input.text())))
        self.lesion_pet_gamma_input.setText(str(state.get("lesion_pet_gamma", self.lesion_pet_gamma_input.text())))
        self.lesion_spect_beta_input.setText(str(state.get("lesion_spect_beta", self.lesion_spect_beta_input.text())))
        self.lesion_spect_gamma_input.setText(str(state.get("lesion_spect_gamma", self.lesion_spect_gamma_input.text())))
        self.lesion_n_neighbours_input.setText(str(state.get("lesion_n_neighbours", self.lesion_n_neighbours_input.text())))
        self.lesion_support_kernel_input.setText(str(state.get("lesion_support_kernel", self.lesion_support_kernel_input.text())))
        self.lesion_distance_kernel_input.setText(str(state.get("lesion_distance_kernel", self.lesion_distance_kernel_input.text())))
        self.lesion_top_n_input.setText(str(state.get("lesion_top_n", self.lesion_top_n_input.text())))
        self.kernel_gpu_checkbox.setChecked(bool(state.get("kernel_gpu", self.kernel_gpu_checkbox.isChecked())))
        self.lesion_post_recon_filter_checkbox.setChecked(bool(state.get("post_recon_enabled", False)))
        self.lesion_post_recon_cutoff_input.setText(str(state.get("post_recon_cutoff", self.lesion_post_recon_cutoff_input.text())))
        self.center_x_input.setText(str(state.get("center_x", "")))
        self.center_y_input.setText(str(state.get("center_y", "")))
        self.center_z_input.setText(str(state.get("center_z", "")))
        self.radius_mm_input.setText(str(state.get("radius_mm", self.radius_mm_input.text())))
        self.intensity_input.setText(str(state.get("intensity", self.intensity_input.text())))
        self.size_scale_input.setText(str(state.get("size_scale", self.size_scale_input.text())))
        self.lesion_name_input.setText(str(state.get("lesion_name", "")))

        viewer_modes = state.get("viewer_modes") or []
        for viewer, mode in zip(getattr(self, "viewers", []), viewer_modes):
            idx = viewer.content_combo.findText(str(mode))
            if idx >= 0:
                viewer.content_combo.setCurrentIndex(idx)

        slider_values = state.get("slider_values") or {}
        for slider, key in (
            (self.sagittal_slider, "sagittal"),
            (self.coronal_slider, "coronal"),
            (self.axial_slider, "axial"),
        ):
            if key in slider_values:
                slider.setValue(max(0, min(slider.maximum(), int(slider_values[key]))))

        sel_lesion_idx = int(state.get("selected_lesion_index", -1))
        if 0 <= sel_lesion_idx < self.lesion_select_combo.count():
            self.lesion_select_combo.setCurrentIndex(sel_lesion_idx)
        sel_queue_idx = int(state.get("selected_queue_index", -1))
        if 0 <= sel_queue_idx < self.queue_list.count():
            self.queue_list.setCurrentRow(sel_queue_idx)
        sel_result_idx = int(state.get("selected_result_index", -1))
        if 0 <= sel_result_idx < self.results_list.count():
            self.results_list.setCurrentRow(sel_result_idx)

    def save_lesion_workflow_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save lesion insertion workflow",
            "lesion_workflow.json",
            "JSON workflow (*.json);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        payload = {
            "format": self._LESION_WORKFLOW_FORMAT,
            "version": self._LESION_WORKFLOW_VERSION,
            "selected_modality": self.selected_modality,
            "paths": {
                "pet_folder_path": self.pet_folder_path,
                "ct_folder_path": self.ct_folder_path,
                "projection_file_path": self.projection_file_path,
            },
            "quant_mode": self.quant_mode,
            "quantification_params": self._sanitize_for_json(self.quantification_params),
            "quant_meta_defaults": self._sanitize_for_json(self.quant_meta_defaults),
            "voxel_size_mm": self._sanitize_for_json(list(self.voxel_size_mm) if self.voxel_size_mm is not None else None),
            "bed_z_offsets": self._serialize_bed_z_offsets_for_workflow(),
            "mc_params": self._sanitize_for_json(self.mc_params),
            "current_coordinates": self._sanitize_for_json(list(self.current_coordinates)),
            "color_bar_levels": self._sanitize_for_json(list(self.color_bar_levels)),
            "arrays": {
                "image_array": self._encode_array_for_json(self.image_array) if self.image_array is not None else None,
                "registered_ct_array": self._encode_array_for_json(self.registered_ct_array) if self.registered_ct_array is not None else None,
                "lesion_array": self._encode_array_for_json(self.lesion_array) if self.lesion_array is not None else None,
                "final_array": self._encode_array_for_json(self.final_array) if self.final_array is not None else None,
                "occurrence_pet_array": self._encode_array_for_json(self.occurrence_pet_array) if self.occurrence_pet_array is not None else None,
                "occurrence_ct_array": self._encode_array_for_json(self.occurrence_ct_array) if self.occurrence_ct_array is not None else None,
            },
            "occurrence_spacing": {
                "pet": self._sanitize_for_json(list(self.occurrence_pet_spacing)),
                "ct": self._sanitize_for_json(list(self.occurrence_ct_spacing)),
            },
            "lesions": self._serialize_lesions_for_workflow(),
            "queue_items": [self._serialize_queue_item_for_workflow(item) for item in self.reconstruction_queue],
            "queue_results": [self._serialize_queue_item_for_workflow(item) for item in self.queue_results],
            "ui_state": self._collect_lesion_workflow_ui_state(),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            QMessageBox.information(
                self,
                "Workflow saved",
                f"Saved lesion insertion workflow to:\n{path}\n\n"
                f"Lesions: {len(self.lesion_objects)}\n"
                f"Queued reconstructions: {len(self.reconstruction_queue)}\n"
                f"Saved results: {len(self.queue_results)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save workflow.\n\n{e}")

    def load_lesion_workflow_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load lesion insertion workflow",
            "",
            "JSON workflow (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not read workflow JSON.\n\n{e}")
            return

        if data.get("format") != self._LESION_WORKFLOW_FORMAT:
            QMessageBox.warning(self, "Wrong file", "This JSON is not a lesion insertion workflow file.")
            return

        ver = int(data.get("version", 0))
        if ver != self._LESION_WORKFLOW_VERSION:
            QMessageBox.warning(
                self,
                "Version mismatch",
                f"This file is version {ver}; this app expects {self._LESION_WORKFLOW_VERSION}. Loading may be incomplete.",
            )

        modality = data.get("selected_modality")
        if modality in {"PET", "SPECT"}:
            self.modality_combo.setCurrentText(modality)

        paths = data.get("paths") or {}
        self.pet_folder_path = paths.get("pet_folder_path")
        self.ct_folder_path = paths.get("ct_folder_path")
        self.projection_file_path = paths.get("projection_file_path")

        self.pet_folder_status.setText(f"✅ {os.path.basename(self.pet_folder_path)}" if self.pet_folder_path else "No folder selected.")
        self.pet_ct_status.setText(f"✅ {os.path.basename(self.ct_folder_path)}" if self.selected_modality == "PET" and self.ct_folder_path else "No folder selected.")
        self.ct_status.setText(f"✅ {os.path.basename(self.ct_folder_path)}" if self.selected_modality == "SPECT" and self.ct_folder_path else "No folder selected.")
        self.projection_status.setText(f"✅ {os.path.basename(self.projection_file_path)}" if self.projection_file_path else "No file/folder selected.")

        self.quant_mode = str(data.get("quant_mode", "Arbitrary"))
        self.quantification_params = dict(data.get("quantification_params") or self.quantification_params)
        self.quant_meta_defaults = dict(data.get("quant_meta_defaults") or {})
        vx = data.get("voxel_size_mm")
        self.voxel_size_mm = tuple(float(v) for v in vx[:3]) if isinstance(vx, list) and len(vx) >= 3 else None
        self.bed_z_offsets = self._deserialize_bed_z_offsets_from_workflow(data.get("bed_z_offsets"))
        if self.bed_z_offsets is None:
            self._try_restore_bed_z_offsets_from_projections()
        self.mc_params = dict(data.get("mc_params") or {})

        arrays = data.get("arrays") or {}
        self.image_array = self._decode_array_from_json(arrays.get("image_array")) if arrays.get("image_array") else None
        self.registered_ct_array = self._decode_array_from_json(arrays.get("registered_ct_array")) if arrays.get("registered_ct_array") else None
        self.final_array = self._decode_array_from_json(arrays.get("final_array")) if arrays.get("final_array") else None
        self.occurrence_pet_array = self._decode_array_from_json(arrays.get("occurrence_pet_array")) if arrays.get("occurrence_pet_array") else None
        self.occurrence_ct_array = self._decode_array_from_json(arrays.get("occurrence_ct_array")) if arrays.get("occurrence_ct_array") else None

        occ_spacing = data.get("occurrence_spacing") or {}
        pet_spacing = occ_spacing.get("pet")
        ct_spacing = occ_spacing.get("ct")
        if isinstance(pet_spacing, list) and len(pet_spacing) >= 3:
            self.occurrence_pet_spacing = tuple(float(v) for v in pet_spacing[:3])
        if isinstance(ct_spacing, list) and len(ct_spacing) >= 3:
            self.occurrence_ct_spacing = tuple(float(v) for v in ct_spacing[:3])

        restored_lesions, skipped_lesions = self._restore_lesions_from_workflow(data.get("lesions") or [])

        lesion_array_payload = arrays.get("lesion_array")
        if lesion_array_payload:
            self.lesion_array = self._decode_array_from_json(lesion_array_payload)
        elif self.image_array is not None:
            self.lesion_array = np.zeros_like(self.image_array, dtype=np.float32)
        else:
            self.lesion_array = None

        if self.image_array is not None:
            dims = self.image_array.shape
            self.sagittal_slider.setMaximum(dims[0] - 1)
            self.coronal_slider.setMaximum(dims[1] - 1)
            self.axial_slider.setMaximum(dims[2] - 1)
            coords = data.get("current_coordinates") or [dims[0] // 2, dims[1] // 2, dims[2] // 2]
            self.current_coordinates = tuple(int(v) for v in coords[:3])
            while len(self.current_coordinates) < 3:
                self.current_coordinates = self.current_coordinates + (0,)
            self._refresh_lesion_combo()
            if self.lesion_objects:
                self._rebuild_lesion_mask_display()
            else:
                self._update_global_color_scale()
                self.update_all_views()
        else:
            self.current_coordinates = tuple(int(v) for v in (data.get("current_coordinates") or [0, 0, 0])[:3])
            self._update_global_color_scale()
            self.update_all_views()

        self.reconstruction_queue = []
        skipped_queue_items = 0
        for item_data in data.get("queue_items") or []:
            try:
                self.reconstruction_queue.append(self._deserialize_queue_item_from_workflow(item_data))
            except Exception:
                skipped_queue_items += 1
        self._refresh_queue_list()

        self.queue_results = []
        skipped_results = 0
        for item_data in data.get("queue_results") or []:
            try:
                self.queue_results.append(self._deserialize_queue_item_from_workflow(item_data))
            except Exception:
                skipped_results += 1
        self._refresh_results_list()

        self._restore_lesion_workflow_ui_state(data.get("ui_state") or {})
        self._update_ui_state()
        self._update_lesion_ui_state()
        self.tabs.setCurrentIndex(2)

        parts = [
            f"Loaded workflow from:\n{path}",
            f"Restored lesions: {restored_lesions}" + (f" (skipped {skipped_lesions})" if skipped_lesions else ""),
            f"Restored queued reconstructions: {len(self.reconstruction_queue)}" + (f" (skipped {skipped_queue_items})" if skipped_queue_items else ""),
            f"Restored saved results: {len(self.queue_results)}" + (f" (skipped {skipped_results})" if skipped_results else ""),
        ]
        QMessageBox.information(self, "Workflow loaded", "\n\n".join(parts))

    def _prepare_quantitative_dicom(self, sitk_float_image):
        stats = sitk.StatisticsImageFilter()
        stats.Execute(sitk_float_image)
        min_val, max_val = stats.GetMinimum(), stats.GetMaximum()
        pixel_type = sitk.sitkInt16
        int_min, int_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        slope = (max_val - min_val) / (int_max - int_min) if max_val > min_val else 1.0
        intercept = min_val - int_min * slope
        int_image = sitk.Cast(sitk.IntensityWindowing(sitk_float_image, windowMinimum=min_val, windowMaximum=max_val, outputMinimum=int_min, outputMaximum=int_max), pixel_type)
        int_image.SetMetaData("0028|1052", str(intercept))
        int_image.SetMetaData("0028|1053", str(slope))
        return int_image

    def save_initial_image(self):
        if self.image_array is None:
            QMessageBox.warning(self, "No Initial Image", "No initial image has been reconstructed.")
            return
        save_mode = "Arbitrary"
        if self.selected_modality == "PET":
            chosen = self._prompt_pet_save_unit_mode()
            if chosen is None:
                return
            save_mode = chosen
        path, _ = QFileDialog.getSaveFileName(self, "Save Initial Image", "Initial_Reconstruction", "NIfTI Files (*.nii *.nii.gz)")
        if not path: return
        try:
            if not (path.lower().endswith('.nii') or path.lower().endswith('.nii.gz')): path += '.nii'
            if self.selected_modality == "PET":
                save_array = self._convert_pet_volume_for_save_unit(self.image_array, save_mode)
                if save_array is None:
                    return
            else:
                save_array = np.asarray(self.image_array, dtype=np.float32)
            image_to_save = np.transpose(save_array, (2, 1, 0))
            sitk_image = sitk.GetImageFromArray(image_to_save.astype(np.float32))
            unit_lbl = self._save_unit_display_label(save_mode) if self.selected_modality == "PET" else "counts"
            sitk_image.SetMetaData("descrip", f"Initial recon saved in {unit_lbl}")
            if self.selected_modality == "PET":
                sitk_image.SetMetaData("quant_mode", save_mode)
                sitk_image.SetMetaData("cf_to_bqml", str(self.quantification_params.get("cf_to_bqml", 1.0)))
                sitk_image.SetMetaData("patient_weight_kg", str(self.quantification_params.get("patient_weight_kg", 0.0)))
                sitk_image.SetMetaData("injected_dose_bq", str(self.quantification_params.get("injected_dose_bq", 0.0)))
                sitk_image.SetMetaData("half_life_s", str(self.quantification_params.get("half_life_s", 0.0)))
                sitk_image.SetMetaData("delta_t_s", str(self.quantification_params.get("delta_t_s", 0.0)))
            else:
                sitk_image.SetMetaData("quant_mode", "SPECT")
            sitk.WriteImage(sitk_image, path)
            QMessageBox.information(self, "Save Successful", f"Initial image saved to {os.path.basename(path)}")
            self.status_bar.showMessage(f"Initial image saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the initial image: {e}\n{traceback.format_exc()}")
            self.status_bar.showMessage(f"Error saving initial image: {e}")

    def save_registered_ct(self):
        if self.registered_ct_array is None:
            QMessageBox.warning(self, "No Registered CT Data", "No registered CT data is available to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Registered CT", "Registered_CT", "NIfTI Files (*.nii *.nii.gz)")
        if not path: return
        try:
            self.status_bar.showMessage("Writing quantitative NIfTI file...")
            QApplication.processEvents()
            image_to_save = np.transpose(self.registered_ct_array, (2, 1, 0))
            float_sitk_image = sitk.GetImageFromArray(image_to_save.astype(np.float32))
            int_sitk_image = self._prepare_quantitative_dicom(float_sitk_image)
            int_sitk_image.CopyInformation(float_sitk_image)
            sitk.WriteImage(int_sitk_image, path)
            QMessageBox.information(self, "Save Successful", f"Registered CT saved successfully to:\n{os.path.basename(path)}")
            self.status_bar.showMessage(f"Registered CT NIfTI saved.")
        except Exception as e:
            error_message = f"Could not save the registered CT NIfTI file:\n{e}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Save Error", error_message)
            self.status_bar.showMessage(f"Error saving registered CT: {e}")

    def save_mask_only(self):
        if self.lesion_array is None or not self.lesion_array.any():
            QMessageBox.warning(self, "No Lesion Mask", "No lesion mask has been created.")
            return
        suggested_name = f"Lesion_Mask_{len(self.lesion_objects)}"
        path, _ = QFileDialog.getSaveFileName(self, "Save Lesion Mask", suggested_name, "NIfTI Files (*.nii *.nii.gz)")
        if not path: return
        try:
            if not (path.lower().endswith('.nii') or path.lower().endswith('.nii.gz')): path += '.nii'
            mask_to_save = np.transpose(self.lesion_array, (2, 1, 0))
            sitk_image = sitk.GetImageFromArray(mask_to_save.astype(np.float32))
            sitk.WriteImage(sitk_image, path)
            QMessageBox.information(self, "Save Successful", f"Lesion mask saved to {os.path.basename(path)}")
            self.status_bar.showMessage(f"Lesion mask saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the lesion mask: {e}\n{traceback.format_exc()}")
            self.status_bar.showMessage(f"Error saving lesion mask: {e}")

    def _get_lesion_summary(self):
        if not self.lesion_objects: return "No lesions"
        return "; ".join([f"L{i}:({l.center[0]+1},{l.center[1]+1},{l.center[2]+1}),R{l.radius},I{l.intensity:.2f}" for i, l in enumerate(self.lesion_objects, 1)])[:1024]

    def _create_about_tab(self):
        about_tab = QWidget()
        layout = QVBoxLayout(about_tab)
        layout.addStretch(1)
        title = QLabel("Lesion Insertion Toolbox")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        desc = QLabel("<p style='font-size:12pt; text-align:center;'>This software facilitates the insertion of artificial lesions into PET and SPECT imaging data for research purposes.<br><br>Built using Python, PyQt5, and the PyTomography library.</p>")
        desc.setWordWrap(True)
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(desc)
        layout.addStretch(1)
        self.tabs.addTab(about_tab, "About")

    def _clear_setup_data_paths(self):
        """Drop loaded folder paths and modality-specific recon state (Setup tab reset)."""
        self.pet_folder_path = None
        self.ct_folder_path = None
        self.projection_file_path = None
        self.registered_ct_array = None
        self.bed_z_offsets = None
        self.final_array = None
        self.lesion_array = None
        self.voxel_size_mm = None
        self.pending_real_lesion = None

    def _validate_pet_data_folder(self, path):
        """
        Check a PET data folder before enabling reconstruction.
        Returns (ok, error_message).
        """
        try:
            names = os.listdir(path)
        except OSError as exc:
            return False, f"Cannot read folder:\n{exc}"

        blf_files = [f for f in names if f.lower().endswith(".blf")]
        h5_files = [f for f in names if f.lower().endswith(".h5")]

        if not blf_files:
            return False, (
                "No listmode (.blf) files were found in this folder.\n\n"
                "PET reconstruction requires at least one .blf listmode file."
            )
        if not h5_files:
            return False, (
                "No corrections (.h5) files were found in this folder.\n\n"
                "PET reconstruction requires at least one .h5 corrections file."
            )
        if len(blf_files) > 1 and len(h5_files) < len(blf_files):
            return False, (
                f"Found {len(blf_files)} .blf bed(s) but only {len(h5_files)} .h5 file(s).\n\n"
                "Multi-bed PET needs a corrections .h5 for each bed (or clearly paired filenames)."
            )
        return True, ""

    def _modality_changed(self, modality):
        self.selected_modality = None if modality == "-- Select a Modality --" else modality
        self.image_array = None
        self.lesion_objects = []
        self.reconstruction_queue = []
        self.queue_results = []
        self._clear_setup_data_paths()
        self.pet_folder_status.setText("No folder selected.")
        self.pet_ct_status.setText("No folder selected.")
        self.ct_status.setText("No folder selected.")
        self.projection_status.setText("No file/folder selected.")
        self.lesion_select_combo.clear()
        self.queue_list.clear()
        self.results_list.clear()
        self.occurrence_pet_array = None
        self.occurrence_ct_array = None
        self.quant_mode = "Arbitrary"
        self.quantification_params = {
            "cf_to_bqml": 1.0,
            "patient_weight_kg": 70.0,
            "injected_dose_bq": 370000000.0,
            "half_life_s": 6586.2,
            "delta_t_s": 0.0
        }
        self.quant_meta_defaults = {}
        
        self.update_all_views()
        self._update_ui_state()
        self._update_lesion_ui_state()
        self._on_tab_changed(self.tabs.currentIndex())

    def _update_ui_state(self):
        has_modality = self.selected_modality is not None
        self.data_stack.setVisible(has_modality)
        self.params_container.setVisible(True)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        if not has_modality:
            self.recon_button.setEnabled(False)
            for i in reversed(range(self.params_layout.count())):
                child = self.params_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            if self._recon_setup_hint is None:
                self._recon_setup_hint = QLabel(
                    "<p style='font-size:11pt;'>Select <b>PET</b> or <b>SPECT</b> in the <b>Setup</b> tab and load your data there. "
                    "Algorithm-specific reconstruction parameters will appear in this area.</p>"
                )
                self._recon_setup_hint.setWordWrap(True)
            self.params_layout.addWidget(self._recon_setup_hint)
            return
        is_pet = (self.selected_modality == "PET")
        for i in reversed(range(self.params_layout.count())):
            child = self.params_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        if is_pet:
            self.data_stack.setCurrentWidget(self.pet_data_widget)
            self.params_layout.addWidget(self.pet_recon_group)
            is_bsrem = self.pet_algo_combo.currentText() == "BSREM"
            self.pet_beta_label.setVisible(is_bsrem)
            self.pet_beta_input.setVisible(is_bsrem)
            self.pet_gamma_label.setVisible(is_bsrem)
            self.pet_gamma_input.setVisible(is_bsrem)
        else:
            self.data_stack.setCurrentWidget(self.spect_data_widget)
            self.params_layout.addWidget(self.spect_recon_group)
            algo = self.spect_algo_combo.currentText()
            self.spect_prior_group.setVisible(algo in ["OSMAPOSL", "BSREM"])
            self.spect_kem_group.setVisible(algo == "KEM")
        if is_pet:
             recon_ready = bool(self.pet_folder_path)
        else:
            recon_ready = bool(self.ct_folder_path and self.projection_file_path)
        self.recon_button.setEnabled(recon_ready)
        can_save_lesion = self.is_setting_center or self.sphere_center is not None
        self.save_lesion_btn.setEnabled(can_save_lesion)
        self.lesion_name_input.setEnabled(can_save_lesion)
        manage_enabled = len(self.lesion_objects) > 0
        self.lesion_select_combo.setEnabled(manage_enabled)
        self.edit_selected_btn.setEnabled(manage_enabled)
        self.remove_sphere_btn.setEnabled(manage_enabled)
        if hasattr(self, 'unit_conversion_btn'):
            self.unit_conversion_btn.setEnabled(self.image_array is not None and self.selected_modality == "PET")
        if hasattr(self, 'save_registered_ct_btn'):
            self.save_registered_ct_btn.setEnabled(self.registered_ct_array is not None)
        self._sync_size_scale_visibility()

    def _update_lesion_ui_state(self):
        if not hasattr(self, 'lesion_params_stack'): return
        is_pet = (self.selected_modality == "PET")
        self.lesion_params_stack.setCurrentIndex(0 if is_pet else 1)
        self._mount_tab3_recon_params_for_modality(is_pet)
        if is_pet:
            is_bsrem = self.lesion_pet_algo_combo.currentText() == "BSREM"
            self.lesion_pet_beta_label.setVisible(is_bsrem)
            self.lesion_pet_beta_input.setVisible(is_bsrem)
            self.lesion_pet_gamma_label.setVisible(is_bsrem)
            self.lesion_pet_gamma_input.setVisible(is_bsrem)
        else:
            algo = self.lesion_spect_algo_combo.currentText()
            is_mc_selected = self.lesion_spect_projection_method_combo.currentText() == "Monte Carlo"
            if algo in ["OSMAPOSL", "BSREM"]:
                self.lesion_spect_algo_stack.setCurrentIndex(1)
            elif algo == "KEM":
                self.lesion_spect_algo_stack.setCurrentIndex(2)
            else:
                self.lesion_spect_algo_stack.setCurrentIndex(0)
            self.lesion_mc_button_row.setVisible(is_mc_selected)
            self._sync_tab3_spect_params_layout()
        self.add_to_queue_btn.setEnabled(len(self.lesion_objects) > 0)
        self.execute_queue_btn.setEnabled(len(self.reconstruction_queue) > 0)

    def _mount_tab3_recon_params_for_modality(self, is_pet):
        """PET: plain panel (no scroll). SPECT: scrollable panel."""
        if not hasattr(self, "lesion_recon_params_group"):
            return
        group = self.lesion_recon_params_group
        group.setParent(None)
        if is_pet:
            self.lesion_recon_params_pet_layout.addWidget(group)
            self.lesion_recon_params_host.setCurrentIndex(0)
        else:
            self.lesion_recon_params_scroll_layout.addWidget(group)
            self.lesion_recon_params_host.setCurrentIndex(1)

    def _sync_tab3_spect_params_layout(self):
        """Tab 3 SPECT only: size scrollable recon-params panel to its current content."""
        if getattr(self, "selected_modality", None) == "PET":
            return
        if not hasattr(self, "lesion_recon_params_scroll"):
            return
        if hasattr(self, "lesion_spect_algo_stack"):
            page = self.lesion_spect_algo_stack.currentWidget()
            page_h = page.sizeHint().height() if page is not None else 0
            self.lesion_spect_algo_stack.setMinimumHeight(0)
            self.lesion_spect_algo_stack.setMaximumHeight(max(page_h, 0))
        if hasattr(self, "lesion_spect_widget"):
            self.lesion_spect_widget.updateGeometry()
        if hasattr(self, "lesion_params_stack"):
            self.lesion_params_stack.updateGeometry()
        if hasattr(self, "lesion_recon_params_group"):
            self.lesion_recon_params_group.updateGeometry()
            self.lesion_recon_params_group.adjustSize()
        self.lesion_recon_params_scroll_content.setMinimumHeight(0)
        self.lesion_recon_params_scroll_content.updateGeometry()
        self.lesion_recon_params_scroll_content.adjustSize()

        content_h = self.lesion_recon_params_group.sizeHint().height()
        spect_cap = 420

        if content_h <= spect_cap:
            self.lesion_recon_params_scroll.setMinimumHeight(content_h)
            self.lesion_recon_params_scroll.setMaximumHeight(content_h)
            self.lesion_recon_params_scroll_content.setMinimumHeight(0)
        else:
            self.lesion_recon_params_scroll.setMinimumHeight(spect_cap)
            self.lesion_recon_params_scroll.setMaximumHeight(spect_cap)
            self.lesion_recon_params_scroll_content.setMinimumHeight(content_h)

        self.lesion_recon_params_scroll.updateGeometry()

    def load_pet_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select PET Data Folder (containing .BLF and .h5 files)")
        if not path:
            return
        ok, err = self._validate_pet_data_folder(path)
        if not ok:
            QMessageBox.warning(self, "Invalid PET Data Folder", err)
            return
        self.pet_folder_path = path
        self.pet_folder_status.setText(f"✅ {os.path.basename(path)}")
        self._try_auto_load_pet_quant_metadata(path)
        self._update_ui_state()

    def load_pet_ct_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select CT Data Folder (Optional)")
        if path:
            self.ct_folder_path = path 
            self.pet_ct_status.setText(f"✅ {os.path.basename(path)}")
            self._update_ui_state()

    def load_ct_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select CT Folder")
        if path:
            self.ct_folder_path = path
            self.ct_status.setText(f"✅ {os.path.basename(path)}")
            self._update_ui_state()

    def load_projection_file(self):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Select SPECT Study Type")
            msg_box.setText("Single-bed or multi-bed study?")
            single_button = msg_box.addButton("Single-Bed (File)", QMessageBox.ActionRole)
            multi_button = msg_box.addButton("Multi-Bed (Folder)", QMessageBox.ActionRole)
            msg_box.addButton(QMessageBox.Cancel)
            msg_box.exec_()
            path = None
            if msg_box.clickedButton() == single_button:
                path, _ = QFileDialog.getOpenFileName(self, "Select Projection DICOM File", "", "DICOM Files (*.dcm *.IMA);;All Files (*)")
            elif msg_box.clickedButton() == multi_button:
                path = QFileDialog.getExistingDirectory(self, "Select Folder with Projection DICOMs")
            if path:
                self.projection_file_path = path
                self.projection_status.setText(f"✅ {os.path.basename(path)}")
                self._update_ui_state()

    def _confirm_repeat_initial_recon(self):
        """Ask before repeat initial recon when existing workflow data would be cleared on success."""
        if self.image_array is None:
            return True
        reply = QMessageBox.question(
            self,
            "Replace Initial Reconstruction?",
            "Running initial reconstruction again will replace the current volume and, "
            "if reconstruction succeeds, clear all downstream work:\n\n"
            "• Saved lesions and lesion masks\n"
            "• Reconstruction queue and completed results\n"
            "• Data analysis scenarios and results\n\n"
            "If you cancel reconstruction, your current work will be kept.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def execute_reconstruction(self):
        if not self._confirm_repeat_initial_recon():
            return
        try:
            n_iters = int(self.iterations_input.text())
            n_subsets = int(self.subsets_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Iterations and Subsets must be integers.")
            return
        total_steps = n_iters
        if self.selected_modality == "PET" and self.pet_folder_path:
            try:
                files = [f for f in os.listdir(self.pet_folder_path) if f.lower().endswith('.blf')]
                num_beds = len(files)
                if num_beds > 1: total_steps = num_beds * n_iters
            except Exception: pass
        elif self.selected_modality == "SPECT" and self.projection_file_path and os.path.isdir(self.projection_file_path):
            try:
                num_beds = len([f for f in os.listdir(self.projection_file_path) if f.lower().endswith(('.dcm', '.IMA'))])
                total_steps = num_beds * n_iters
            except Exception: pass

        self.progress = QProgressDialog("Starting Initial Reconstruction...", "Cancel", 0, total_steps, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumWidth(450)
        self.progress.show()

        def _cancel_initial_reconstruction():
            request_recon_thread_cancel(getattr(self, "recon_thread", None))

        self.progress.canceled.connect(_cancel_initial_reconstruction)

        def update_dialog(value, label_text):
            if hasattr(self, 'progress') and self.progress.isVisible():
                self.progress.setValue(value)
                self.progress.setLabelText(label_text)

        if self.selected_modality == "PET":
            try:
                dr = tuple(map(float, self.object_dr_input.text().split(',')[:3]))
                psf = float(self.psf_input.text())
                algo = self.pet_algo_combo.currentText()
                beta = float(self.pet_beta_input.text()) if algo == "BSREM" else None
                gamma = float(self.pet_gamma_input.text()) if algo == "BSREM" else None
                shape_xy = tuple(map(int, self.object_shape_input.text().split(',')))
                if len(shape_xy) == 3: shape_xy = shape_xy[0:2]
                overlap = int(self.pet_overlap_input.text())
                z_slices = int(self.pet_z_slices_input.text())
                self.recon_thread = PETReconstructionThread(
                    self, self.pet_folder_path, self.scanner_name_input.text(), dr, shape_xy, psf, n_iters, n_subsets, algo, beta, gamma,
                    overlap=overlap, z_slices_per_bed=z_slices
                )
            except (ValueError, IndexError):
                QMessageBox.critical(self, "Input Error", "Invalid format for PET parameters.")
                self.progress.close()
                return
        elif self.selected_modality == "SPECT":
            try:
                algo = self.spect_algo_combo.currentText()
                beta = float(self.spect_beta_input.text()) if algo in ["BSREM", "OSMAPOSL"] else None
                gamma = float(self.spect_gamma_input.text()) if algo in ["BSREM", "OSMAPOSL"] else None
                n_neighbours = int(self.n_neighbours_input.text()) if algo in ["BSREM", "OSMAPOSL"] else 8
                support_kernel = float(self.support_kernel_input.text()) if algo == "KEM" else 0.005
                distance_kernel = float(self.distance_kernel_input.text()) if algo == "KEM" else 0.4
                top_n = int(self.top_n_input.text()) if algo == "KEM" else 40
                kernel_gpu = self.kernel_gpu_checkbox.isChecked() if algo == "KEM" else True
                collimator = self.collimator_combo.currentText()
                energy = int(self.energy_input.text())
                intrinsic_res = float(self.intrinsic_resolution_input.text())
                idx_peak = int(self.index_peak_input.text())
                idx_lower = int(self.index_lower_input.text())
                idx_upper = int(self.index_upper_input.text())
                self.recon_thread = SPECTReconstructionThread(self, self.projection_file_path, self.ct_folder_path, algo, n_iters, n_subsets, beta, gamma, n_neighbours, support_kernel, distance_kernel, top_n, kernel_gpu, collimator, energy, intrinsic_res, idx_peak, idx_lower, idx_upper)
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Invalid format for SPECT parameters.")
                self.progress.close()
                return
        else: return

        self.recon_thread.progress_signal.connect(update_dialog)
        self.recon_thread.finished_signal.connect(self.reconstruction_finished)
        self.recon_thread.start()

    def reconstruction_finished(self, result, offsets, status):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        if status == "Cancelled":
            self.status_bar.showMessage("Initial reconstruction cancelled.")
            self._update_ui_state()
            return
        if status == "Success":
            if hasattr(self, 'progress'): self.progress.setValue(self.progress.maximum())
            self._temp_recon_result = (result, offsets)
            QTimer.singleShot(100, self._process_successful_recon)
        else:
            QMessageBox.critical(self, "Reconstruction Error", f"An error occurred: {status}")
            self._update_ui_state()

    def _reset_workflow_after_initial_recon(self):
        """Clear lesion, queue, and analysis state so repeat initial recon matches the first run."""
        self.lesion_objects = []
        self.pending_real_lesion = None
        self.reconstruction_queue = []
        self.queue_results = []
        self.final_array = None
        self.sphere_center = None
        self.is_setting_center = False
        self.current_lesion_recon_name = ""

        self.occurrence_pet_array = None
        self.occurrence_ct_array = None
        self.occurrence_pet_spacing = (1.0, 1.0, 1.0)
        self.occurrence_ct_spacing = (1.0, 1.0, 1.0)

        if hasattr(self, "queue_list"):
            self.queue_list.clear()
        if hasattr(self, "results_list"):
            self.results_list.clear()
        if hasattr(self, "queue_name_input"):
            self.queue_name_input.clear()

        if hasattr(self, "center_x_input"):
            self.center_x_input.clear()
        if hasattr(self, "center_y_input"):
            self.center_y_input.clear()
        if hasattr(self, "center_z_input"):
            self.center_z_input.clear()
        if hasattr(self, "lesion_name_input"):
            self.lesion_name_input.clear()
        if hasattr(self, "radius_mm_input"):
            self.radius_mm_input.setText("10.0")
        self._reset_lesion_entry_defaults()

        if hasattr(self, "lesion_post_recon_filter_checkbox"):
            self.lesion_post_recon_filter_checkbox.setChecked(False)
            self._on_lesion_post_recon_filter_toggled(False)

        for viewer in getattr(self, "viewers", []):
            viewer.roi.setVisible(False)

        self._refresh_lesion_combo()

        for dialog_attr in ("library_dialog", "occur_dialog"):
            dialog = getattr(self, dialog_attr, None)
            if dialog is not None:
                try:
                    dialog.close()
                except Exception:
                    pass
                setattr(self, dialog_attr, None)

        if hasattr(self, "rc_tab") and hasattr(self.rc_tab, "clear_all_workflow_state"):
            self.rc_tab.clear_all_workflow_state()

    def _process_successful_recon(self):
        if hasattr(self, 'progress'): self.progress.close()
        self._reset_workflow_after_initial_recon()
        result, offsets = self._temp_recon_result
        if self.selected_modality == "SPECT":
            self.image_array = np.flip(result, axis=1)
        else:
            self.image_array = result

        try:
            if self.selected_modality == "PET":
                dr_str_list = self.object_dr_input.text().split(',')
                self.voxel_size_mm = [float(d) for d in dr_str_list]
            else:
                from pytomography.io.SPECT import dicom
                meta_file = self.projection_file_path
                if os.path.isdir(meta_file):
                    dicom_files = sorted([f for f in os.listdir(meta_file) if f.lower().endswith(('.dcm', '.IMA'))])
                    if not dicom_files: raise FileNotFoundError("No DICOM files in multi-bed folder.")
                    meta_file = os.path.join(meta_file, dicom_files[0])
                object_meta, _ = dicom.get_metadata(meta_file)
                self.voxel_size_mm = [val * 10 for val in object_meta.dr]
            self.voxel_size_label.setText(f"Voxel Size (mm): {self.voxel_size_mm[0]:.2f}, {self.voxel_size_mm[1]:.2f}, {self.voxel_size_mm[2]:.2f}")
        except Exception as e:
            self.voxel_size_mm = None
            self.voxel_size_label.setText("Voxel Size (mm): Error reading.")

        self._update_sphere_volume_display()

        self.bed_z_offsets = offsets
        self.registered_ct_array = None

        if self.ct_folder_path: 
            try:
                from pytomography.io.shared import open_multifile, _get_affine_multifile
                import numpy.linalg as npl
                from scipy.ndimage import affine_transform
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.status_bar.showMessage("Performing CT Registration...")
                QApplication.processEvents()

                files_CT = [os.path.join(self.ct_folder_path, f) for f in os.listdir(self.ct_folder_path)]
                CT_HU = open_multifile(files_CT).cpu().numpy()
                M_CT = _get_affine_multifile(files_CT)

                if self.selected_modality == "SPECT":
                    from pytomography.io.SPECT import dicom as spect_dicom
                    is_multibed = os.path.isdir(self.projection_file_path)
                    if is_multibed:
                        nm_files = sorted([os.path.join(self.projection_file_path, f) for f in os.listdir(self.projection_file_path) if f.lower().endswith(('.dcm', '.IMA'))])
                        registered_ct_beds = []
                        for file_NM_i in nm_files:
                            object_meta_i, _ = spect_dicom.get_metadata(file_NM_i, index_peak=int(self.index_peak_input.text()))
                            M_NM_i = spect_dicom._get_affine_spect_projections(file_NM_i)
                            M_i = npl.inv(M_CT) @ M_NM_i
                            registered_bed_i = affine_transform(CT_HU, M_i, output_shape=object_meta_i.shape, mode='constant', cval=-1024, order=1)
                            registered_ct_beds.append(torch.tensor(registered_bed_i).to(device))
                        registered_ct_tensor = spect_dicom.stitch_multibed(recons=torch.stack(registered_ct_beds), files_NM=nm_files)
                        registered_ct = registered_ct_tensor.cpu().numpy()
                    else:
                        M_NM = spect_dicom._get_affine_spect_projections(self.projection_file_path)
                        M = npl.inv(M_CT) @ M_NM
                        registered_ct = affine_transform(CT_HU, M, output_shape=self.image_array.shape, mode='constant', cval=-1024, order=1)
                    self.registered_ct_array = np.flip(registered_ct, axis=1).copy()

                elif self.selected_modality == "PET":
                    vx, vy, vz = self.voxel_size_mm
                    nx, ny, nz = self.image_array.shape 
                    M_PET = np.eye(4)
                    M_PET[0, 0] = vx
                    M_PET[1, 1] = vy
                    M_PET[2, 2] = vz
                    M_PET[0, 3] = -(nx * vx) / 2.0
                    M_PET[1, 3] = -(ny * vy) / 2.0
                    M_PET[2, 3] = -(nz * vz) / 2.0
                    ctx, cty, ctz = CT_HU.shape
                    M_CT_centered = M_CT.copy()
                    rotation = M_CT[:3, :3]
                    center_vox_3 = np.array([ctx/2.0, cty/2.0, ctz/2.0])
                    new_origin = - (rotation @ center_vox_3)
                    M_CT_centered[:3, 3] = new_origin
                    M = npl.inv(M_CT_centered) @ M_PET
                    registered_ct = affine_transform(CT_HU, M, output_shape=self.image_array.shape, mode='constant', cval=-1024, order=1)
                    self.registered_ct_array = np.flip(registered_ct, axis=1).copy()

                if self.registered_ct_array is not None:
                    self.status_bar.showMessage("CT Registration successful (Center Aligned, Y-Flipped).")

            except Exception as e:
                self.registered_ct_array = None
                print(f"Registration Warning: {e}")
                self.status_bar.showMessage(f"Registration Warning: {e}")
                QMessageBox.warning(
                    self,
                    "CT Registration Failed",
                    "CT could not be registered to the reconstruction grid.\n\n"
                    f"{e}\n\n"
                    "Reconstruction finished, but the CT overlay on Tab 3 will be unavailable "
                    "until registration succeeds. Check that the CT folder contains valid DICOM "
                    "and matches this study.",
                )

        self.lesion_array = np.zeros_like(self.image_array, dtype=np.float32)
        self.final_array = None
        dims = self.image_array.shape
        self.sagittal_slider.setMaximum(dims[0] - 1)
        self.coronal_slider.setMaximum(dims[1] - 1)
        self.axial_slider.setMaximum(dims[2] - 1)
        self.current_coordinates = (dims[0]//2, dims[1]//2, dims[2]//2)
        self._update_global_color_scale()
        self.update_all_views()
        self.tabs.setCurrentIndex(2)
        self._update_ui_state()
        self._update_lesion_ui_state()
        self.status_bar.showMessage("Initial Reconstruction Completed Successfully!")
        self._temp_recon_result = None

    def add_to_reconstruction_queue(self):
        if not self.lesion_objects:
            QMessageBox.warning(self, "No Lesions", "Please add and save at least one lesion before adding to the queue.")
            return
        name = self.queue_name_input.text().strip() or f"Recon_{len(self.reconstruction_queue) + 1}"
        try:
            n_iters = int(self.lesion_iterations_input.text())
            n_subsets = int(self.lesion_subsets_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Iterations and Subsets must be integers.")
            return
        save_all_iterations = self.save_all_iters_checkbox.isChecked()
        post_recon_fwhm_mm = None
        if self.lesion_post_recon_filter_checkbox.isChecked():
            try:
                post_recon_fwhm_mm = float(self.lesion_post_recon_cutoff_input.text())
                if post_recon_fwhm_mm <= 0:
                    QMessageBox.critical(self, "Input Error", "Filter cutoff (mm) must be positive.")
                    return
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Filter cutoff (mm) must be a valid number.")
                return
        params = {}
        try:
            if self.selected_modality == "PET":
                algo = self.lesion_pet_algo_combo.currentText()
                projection_method = 'ANALYTICAL'
                params.update({
                    'psf_value': float(self.psf_input.text()),
                    'beta': float(self.lesion_pet_beta_input.text()) if algo == "BSREM" else None,
                    'gamma': float(self.lesion_pet_gamma_input.text()) if algo == "BSREM" else None,
                })
            else:
                algo = self.lesion_spect_algo_combo.currentText()
                projection_method = self.lesion_spect_projection_method_combo.currentText()
                params.update({
                    'beta': float(self.lesion_spect_beta_input.text()) if algo in ["BSREM", "OSMAPOSL"] else None,
                    'gamma': float(self.lesion_spect_gamma_input.text()) if algo in ["BSREM", "OSMAPOSL"] else None,
                    'n_neighbours': int(self.lesion_n_neighbours_input.text()) if algo in ["BSREM", "OSMAPOSL"] else 8,
                    'support_kernel': float(self.lesion_support_kernel_input.text()) if algo == "KEM" else 0.005,
                    'distance_kernel': float(self.lesion_distance_kernel_input.text()) if algo == "KEM" else 0.4,
                    'top_n': int(self.lesion_top_n_input.text()) if algo == "KEM" else 40,
                    'kernel_gpu': self.lesion_kernel_gpu_checkbox.isChecked() if algo == "KEM" else True,
                    'collimator': self.collimator_combo.currentText(),
                    'energy': int(self.energy_input.text()),
                    'intrinsic_res': float(self.intrinsic_resolution_input.text()),
                    'idx_peak': int(self.index_peak_input.text()),
                    'idx_lower': int(self.index_lower_input.text()),
                    'idx_upper': int(self.index_upper_input.text())
                })
                if projection_method == 'Monte Carlo':
                    if not self.mc_params:
                        dialog = MonteCarloSettingsDialog()
                        self.mc_params = dialog.get_defaults()
                    params.update(self.mc_params)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Invalid numeric value in reconstruction parameters.")
            return
        params['post_recon_fwhm_mm'] = post_recon_fwhm_mm
        queue_item = ReconstructionQueueItem(name=name, modality=self.selected_modality, algo=algo, n_iters=n_iters, n_subsets=n_subsets, projection_method=projection_method, save_all_iterations=save_all_iterations, **params)
        self.reconstruction_queue.append(queue_item)
        self.queue_list.addItem(str(queue_item))
        self.queue_name_input.clear()
        self._update_lesion_ui_state()
        self.status_bar.showMessage(f"Added '{name}' to the reconstruction queue.")

    def clear_reconstruction_queue(self):
        if not self.reconstruction_queue: return
        if QMessageBox.question(self, "Confirm Clear", "Clear all items from queue?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.reconstruction_queue.clear()
            self.queue_list.clear()
            self._update_lesion_ui_state()
            self.status_bar.showMessage("Reconstruction queue cleared.")

    def execute_reconstruction_queue(self):
        if not self.reconstruction_queue: return
        self.current_queue_index = 0
        self.queue_results = []
        self.results_list.clear()
        self.total_queue_items = len(self.reconstruction_queue)
        self._queue_stop_requested = False
        self.queue_progress = QProgressDialog("Preparing Reconstruction Queue...", "Cancel", 0, 100, self)
        self.queue_progress.setWindowModality(Qt.WindowModal)
        self.queue_progress.setMinimumWidth(500)
        self.queue_progress.setWindowTitle("Reconstruction Queue Progress")
        self.queue_progress.setAutoClose(False)
        self.queue_progress.setAutoReset(False)
        self.queue_progress.canceled.connect(self.stop_reconstruction_queue)
        self.queue_progress.show()
        self.execute_queue_btn.setEnabled(False)
        self.stop_queue_btn.setEnabled(True)
        self.status_bar.showMessage("Processing Reconstruction Queue...")
        self._process_next_queue_item()

    def _update_queue_progress(self, iter_num, total_iters, name):
        if hasattr(self, 'queue_progress') and self.queue_progress.isVisible():
            item_label = f"Processing '{name}' (Item {self.current_queue_index + 1} of {self.total_queue_items})"
            self.queue_progress.setLabelText(item_label)
            if total_iters > 0:
                self.queue_progress.setMaximum(total_iters)
                self.queue_progress.setValue(iter_num)
            if self.queue_progress.wasCanceled():
                self.stop_reconstruction_queue()

    def _make_result_item_from_queue(self, queue_item, result_name, result_array, status):
        """Copy recon metadata from the queue item that produced this result."""
        result_item = ReconstructionQueueItem(
            name=result_name,
            modality=getattr(queue_item, "modality", None),
            algo=getattr(queue_item, "algo", None),
            n_iters=getattr(queue_item, "n_iters", None),
            n_subsets=getattr(queue_item, "n_subsets", None),
            projection_method=getattr(queue_item, "projection_method", "ANALYTICAL"),
            save_all_iterations=bool(getattr(queue_item, "save_all_iterations", False)),
            **(getattr(queue_item, "params", None) or {}),
        )
        if status == "Success":
            result_item.result = result_array
            result_item.status = "Completed"
        else:
            result_item.result = None
            result_item.status = f"Failed: {status}"
        result_item.iter_number = iter_number_for_recon_result(
            result_name,
            n_iters=result_item.n_iters,
            save_all_iterations=result_item.save_all_iterations,
        )
        return result_item

    def _handle_new_result(self, result_array, result_name, status):
        queue_item = self.reconstruction_queue[self.current_queue_index]
        result_item = self._make_result_item_from_queue(queue_item, result_name, result_array, status)
        self.queue_results.append(result_item)
        self.results_list.addItem(f"{'✅' if result_item.status == 'Completed' else '❌'} {result_item.name} - {result_item.status}")

    def _process_next_queue_item(self):
        if getattr(self, "_queue_stop_requested", False):
            self._queue_processing_finished(stopped=True)
            return
        if self.current_queue_index >= len(self.reconstruction_queue):
            self._queue_processing_finished()
            return
        item = self.reconstruction_queue[self.current_queue_index]
        self.current_lesion_recon_name = item.name
        try:
            if item.modality == "PET":
                self.current_recon_thread = LesionFinalizationThread(
                    parent=self, lesion_objects=self.lesion_objects, listmode_file=self.pet_folder_path, corrections_file=self.pet_folder_path,
                    psf_value=item.params.get('psf_value', 4.5), n_iters=item.n_iters, n_subsets=item.n_subsets, recon_method=item.algo,
                    beta=item.params.get('beta'), gamma=item.params.get('gamma'), save_all_iterations=item.save_all_iterations, base_name=item.name,
                    initial_image=self.image_array, post_recon_fwhm_mm=item.params.get('post_recon_fwhm_mm')
                )
            else:
                self.current_recon_thread = SPECTLesionFinalizationThread(
                    parent=self, lesion_objects=self.lesion_objects, projection_file_path=self.projection_file_path, ct_folder_path=self.ct_folder_path,
                    n_iters=item.n_iters, n_subsets=item.n_subsets, recon_method=item.algo,
                    beta=item.params.get('beta'), gamma=item.params.get('gamma'), n_neighbours=item.params.get('n_neighbours', 8),
                    support_kernel=item.params.get('support_kernel', 0.005), distance_kernel=item.params.get('distance_kernel', 0.4),
                    top_n=item.params.get('top_n', 40), kernel_gpu=item.params.get('kernel_gpu', True),
                    collimator=item.params.get('collimator', 'SY-ME'), energy=item.params.get('energy', 208),
                    intrinsic_res=item.params.get('intrinsic_res', 0.38), idx_peak=item.params.get('idx_peak', 0),
                    idx_lower=item.params.get('idx_lower', 1), idx_upper=item.params.get('idx_upper', 2),
                    bed_z_offsets=self.bed_z_offsets, projection_method=item.projection_method,
                    save_all_iterations=item.save_all_iterations, base_name=item.name, initial_image=self.image_array,
                    mc_n_events=item.params.get('mc_n_events', 1e7), mc_n_parallel=item.params.get('mc_n_parallel', 8),
                    mc_isotope_names=item.params.get('mc_isotope_names', ['lu177']), mc_isotope_ratios=item.params.get('mc_isotope_ratios', [1.0]),
                    mc_crystal_thickness=item.params.get('mc_crystal_thickness', 0.9525), mc_cover_thickness=item.params.get('mc_cover_thickness', 0.1),
                    mc_backscatter_thickness=item.params.get('mc_backscatter_thickness', 6.6), mc_energy_res_140=item.params.get('mc_energy_res_140', 10),
                    mc_adv_collimator=item.params.get('mc_adv_collimator', True),
                    post_recon_fwhm_mm=item.params.get('post_recon_fwhm_mm')
                )
            self.current_recon_thread.progress_signal.connect(self._update_queue_progress)
            self.current_recon_thread.result_ready_signal.connect(self._handle_new_result)
            self.current_recon_thread.item_finished_signal.connect(self._queue_item_finished)
            self.current_recon_thread.start()
        except Exception as e:
            self._handle_new_result(None, item.name, f"Thread creation error: {e}")
            self.current_queue_index += 1
            QTimer.singleShot(100, self._process_next_queue_item)

    def _cleanup_finished_recon_thread(self):
        """Tear down the just-finished recon thread and reclaim memory before the
        next queue item starts.

        The thread is created with ``parent=self``, so Qt keeps every finished
        thread alive as a child of the window unless we explicitly delete it.
        Combined with PyTorch's CPU/CUDA caching allocators retaining the
        previous item's buffers, this lets memory accumulate across queue items
        until a later (larger) bed exhausts RAM and the OS hard-kills the
        process. Releasing the thread and forcing the allocators to drop their
        cached blocks keeps each item starting from a clean state.
        """
        thread = getattr(self, "current_recon_thread", None)
        if thread is not None:
            try:
                if thread.isRunning():
                    thread.wait()
            except Exception:
                pass
            try:
                thread.progress_signal.disconnect()
                thread.result_ready_signal.disconnect()
                thread.item_finished_signal.disconnect()
            except Exception:
                pass
            try:
                thread.setParent(None)
                thread.deleteLater()
            except Exception:
                pass
            self.current_recon_thread = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _queue_item_finished(self, status):
        if status == "Cancelled" or getattr(self, "_queue_stop_requested", False):
            self._cleanup_finished_recon_thread()
            self._queue_processing_finished(stopped=True)
            return
        if status != "Success":
            item = self.reconstruction_queue[self.current_queue_index]
            # Keep results_list and queue_results aligned for row-based load/save.
            failed_item = ReconstructionQueueItem(
                name=item.name,
                modality=item.modality,
                algo=item.algo,
                n_iters=item.n_iters,
                n_subsets=item.n_subsets,
                projection_method=getattr(item, "projection_method", "ANALYTICAL"),
                save_all_iterations=bool(getattr(item, "save_all_iterations", False)),
                **(getattr(item, "params", None) or {}),
            )
            failed_item.result = None
            failed_item.status = f"Failed: {status}"
            failed_item.iter_number = 0
            self.queue_results.append(failed_item)
            self.results_list.addItem(f"❌ {failed_item.name} - {failed_item.status}")
        self._cleanup_finished_recon_thread()
        self.current_queue_index += 1
        QTimer.singleShot(100, self._process_next_queue_item)

    def _queue_processing_finished(self, stopped=False):
        if hasattr(self, 'queue_progress') and self.queue_progress:
            self.queue_progress.close()
        self.execute_queue_btn.setEnabled(True)
        self.stop_queue_btn.setEnabled(False)
        self._queue_stop_requested = False
        if stopped:
            self.reconstruction_queue.clear()
            self.queue_list.clear()
            self.status_bar.showMessage("Reconstruction queue stopped by user.")
            return
        self.reconstruction_queue.clear()
        self.queue_list.clear()
        completed = sum(1 for r in self.queue_results if r.status == "Completed")
        QMessageBox.information(self, "Queue Complete", f"Queue processing finished. Generated {completed} result(s).")
        self.status_bar.showMessage("Reconstruction queue finished.")

    def stop_reconstruction_queue(self):
        self._queue_stop_requested = True
        request_recon_thread_cancel(getattr(self, "current_recon_thread", None))
        if hasattr(self, 'queue_progress') and self.queue_progress:
            self.queue_progress.setLabelText("Stopping after current iteration...")
        self.stop_queue_btn.setEnabled(False)
        self.status_bar.showMessage("Stopping reconstruction queue...")

    def load_selected_result(self):
        current_row = self.results_list.currentRow()
        if current_row < 0: return
        result_item = self.queue_results[current_row]
        if result_item.result is None:
            QMessageBox.warning(self, "No Data", f"Result '{result_item.name}' has no data to load.")
            return
        self.final_array = result_item.result.copy()
        for viewer in getattr(self, "viewers", []):
            if hasattr(viewer, "force_level_reset"):
                viewer.force_level_reset = True
                viewer.last_applied_content_type = None
                viewer.last_fg_stats = None
                viewer.fg_data_ref = None
        self._update_global_color_scale()
        self.update_all_views()
        self.status_bar.showMessage(f"Loaded Reconstruction Result: {result_item.name}")

    def save_selected_result(self):
        current_row = self.results_list.currentRow()
        if current_row < 0: return
        result_item = self.queue_results[current_row]
        if result_item.result is None:
            QMessageBox.warning(self, "No Data", "Result has no data to save.")
            return
        save_mode = "Arbitrary"
        if self.selected_modality == "PET":
            chosen = self._prompt_pet_save_unit_mode()
            if chosen is None:
                return
            save_mode = chosen
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", f"{result_item.name}", "NIfTI Files (*.nii *.nii.gz)")
        if not path: return
        try:
            if not (path.lower().endswith('.nii') or path.lower().endswith('.nii.gz')): path += '.nii'
            arr = np.asarray(result_item.result, dtype=np.float32)
            if self.selected_modality == "PET":
                arr = self._convert_pet_volume_for_save_unit(arr, save_mode)
                if arr is None:
                    return
            image_to_save = np.transpose(arr, (2, 1, 0))
            sitk_image = sitk.GetImageFromArray(image_to_save.astype(np.float32))
            if self.selected_modality == "PET":
                sitk_image.SetMetaData("descrip", f"Lesion recon saved in {self._save_unit_display_label(save_mode)}")
                sitk_image.SetMetaData("quant_mode", save_mode)
                sitk_image.SetMetaData("cf_to_bqml", str(self.quantification_params.get("cf_to_bqml", 1.0)))
                sitk_image.SetMetaData("patient_weight_kg", str(self.quantification_params.get("patient_weight_kg", 0.0)))
                sitk_image.SetMetaData("injected_dose_bq", str(self.quantification_params.get("injected_dose_bq", 0.0)))
                sitk_image.SetMetaData("half_life_s", str(self.quantification_params.get("half_life_s", 0.0)))
                sitk_image.SetMetaData("delta_t_s", str(self.quantification_params.get("delta_t_s", 0.0)))
            else:
                sitk_image.SetMetaData("descrip", "SPECT lesion recon (raw counts)")
                sitk_image.SetMetaData("quant_mode", "SPECT")
            sitk.WriteImage(sitk_image, path)
            QMessageBox.information(self, "Save Successful", f"Result saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file: {e}\n{traceback.format_exc()}")

    def save_current_lesion(self):
        if self.sphere_center is None:
            QMessageBox.warning(self, "No Center", "Please set a lesion center before saving.")
            return
        try:
            intensity = float(self.intensity_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Intensity must be a valid number.")
            return

        custom_name = self.lesion_name_input.text().strip()

        # If a real lesion was chosen from library, commit it only when Save is clicked.
        if self.pending_real_lesion is not None:
            try:
                size_scale = float(self.size_scale_input.text())
                if size_scale <= 0:
                    raise ValueError("Size Scale must be positive.")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", str(e))
                return
            lesion_array = self.pending_real_lesion["array"].copy()
            uniform_intensity = bool(self.pending_real_lesion.get("uniform_intensity", False))
            lesion = Lesion(
                self.sphere_center, 0, intensity,
                custom_shape=lesion_array,
                uniform_intensity=uniform_intensity,
                use_recon_voxel_spacing=bool(
                    self.pending_real_lesion.get("use_recon_voxel_spacing", False)
                ),
                source_spacing_mm=self.pending_real_lesion.get("source_spacing_mm"),
            )
            lesion.size_scale = size_scale
            lesion.custom_shape_is_source = True
            lesion.name = custom_name or self.pending_real_lesion["name"]
            lesion.id = str(uuid.uuid4())
        else:
            try:
                radius_mm = float(self.radius_mm_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Radius (mm) must be a valid number for sphere lesions.")
                return
            if radius_mm <= 0:
                QMessageBox.warning(self, "Invalid Input", "Radius (mm) must be positive.")
                return
            if self.voxel_size_mm is None:
                QMessageBox.critical(self, "Voxel Size Error", "Voxel size not available. Please perform an initial reconstruction first.")
                return
            radius_px = radius_mm / self.voxel_size_mm[0]
            lesion = Lesion(self.sphere_center, radius_px, intensity, custom_shape=None)
            lesion.size_scale = 1.0
            lesion.name = custom_name or "Sphere"
            lesion.id = str(uuid.uuid4())
        
        self.lesion_objects.append(lesion)
        self._refresh_lesion_combo()

        self.pending_real_lesion = None
        self.sphere_center = None
        self.lesion_name_input.clear()
        self._reset_lesion_entry_defaults()
        self._rebuild_lesion_mask_display()
        self._update_ui_state()
        self._update_lesion_ui_state()
        self.status_bar.showMessage(f"Saved lesion {len(self.lesion_objects)}. Ready to add another.")

    def _reset_lesion_entry_defaults(self):
        """Clear library pre-fill so sphere (and next lesion) defaults are not affected."""
        self.intensity_input.setText("1.0")
        self.size_scale_input.setText("1.0")
        self._sync_size_scale_visibility()

    def remove_selected_sphere(self):
        idx = self.lesion_select_combo.currentIndex()
        if idx < 0: return
        
        selected_id = self.lesion_select_combo.currentData()
        
        found = False
        if selected_id:
            for i, l in enumerate(self.lesion_objects):
                if str(getattr(l, 'id', '')) == str(selected_id):
                    del self.lesion_objects[i]
                    found = True
                    break
        
        if not found and not selected_id and idx < len(self.lesion_objects):
             del self.lesion_objects[idx]
        
        self._refresh_lesion_combo()
        self._rebuild_lesion_mask_display()
        self._update_ui_state()
        self._update_lesion_ui_state()
        self.status_bar.showMessage("Removed selected lesion.")

    def edit_selected_lesion(self):
        idx = self.lesion_select_combo.currentIndex()
        if idx < 0 or idx >= len(self.lesion_objects):
            QMessageBox.warning(self, "No Lesion Selected", "Please select a lesion to edit.")
            return

        lesion = self.lesion_objects[idx]
        is_library = getattr(lesion, "custom_shape", None) is not None
        if not is_library:
            self._normalize_sphere_lesion_size(lesion)

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Selected Lesion")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        name_input = QLineEdit(str(getattr(lesion, "name", "")))
        center_x_input = QLineEdit(str(int(lesion.center[0]) + 1))
        center_y_input = QLineEdit(str(int(lesion.center[1]) + 1))
        center_z_input = QLineEdit(str(int(lesion.center[2]) + 1))
        intensity_input = QLineEdit(f"{float(getattr(lesion, 'intensity', 1.0)):.3f}".rstrip("0").rstrip("."))
        size_scale_input = None
        radius_input = QLineEdit()

        if not is_library and self.voxel_size_mm is not None:
            radius_mm = float(lesion.radius) * float(self.voxel_size_mm[0])
            radius_input.setText(f"{radius_mm:.3f}".rstrip("0").rstrip("."))
            radius_input.setEnabled(True)
        elif not is_library:
            radius_input.setText(f"{float(lesion.radius):.3f}".rstrip("0").rstrip("."))
            radius_input.setEnabled(True)
        else:
            radius_input.setPlaceholderText("N/A for real lesion")
            radius_input.setToolTip("Real lesion size is controlled by Size Scale.")
            radius_input.setEnabled(False)
            size_scale_input = QLineEdit(
                f"{float(getattr(lesion, 'size_scale', 1.0)):.3f}".rstrip("0").rstrip(".")
            )

        form.addRow("Name:", name_input)
        form.addRow("X:", center_x_input)
        form.addRow("Y:", center_y_input)
        form.addRow("Z:", center_z_input)
        if not is_library:
            form.addRow("Radius (mm):", radius_input)
        form.addRow("Intensity Scale:" if is_library else "Intensity:", intensity_input)
        if size_scale_input is not None:
            form.addRow("Size Scale:", size_scale_input)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            x = int(center_x_input.text()) - 1
            y = int(center_y_input.text()) - 1
            z = int(center_z_input.text()) - 1
            intensity = float(intensity_input.text())

            if self.image_array is not None:
                dims = self.image_array.shape
                if not (0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]):
                    QMessageBox.warning(
                        self,
                        "Invalid Coordinates",
                        f"Coordinates must be inside the image bounds: "
                        f"x=1..{dims[0]}, y=1..{dims[1]}, z=1..{dims[2]}."
                    )
                    return

            if not is_library:
                radius_mm = float(radius_input.text())
                if radius_mm <= 0:
                    raise ValueError("Radius must be positive.")
                if self.voxel_size_mm is None:
                    QMessageBox.critical(self, "Voxel Size Error", "Voxel size not available. Please perform an initial reconstruction first.")
                    return
                lesion.radius = radius_mm / float(self.voxel_size_mm[0])
                lesion.size_scale = 1.0
            else:
                size_scale = float(size_scale_input.text())
                if size_scale <= 0:
                    raise ValueError("Size Scale must be positive.")
                lesion.size_scale = size_scale
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        lesion.center = (x, y, z)
        lesion.intensity = intensity
        lesion.name = name_input.text().strip() or getattr(lesion, "name", "Lesion")

        selected_id = str(getattr(lesion, "id", ""))
        self._refresh_lesion_combo(selected_id=selected_id)
        self._rebuild_lesion_mask_display()
        self._update_ui_state()
        self._update_lesion_ui_state()
        self.status_bar.showMessage(f"Updated lesion '{lesion.name}'.")

    def _parse_lesion_center_from_inputs(self):
        """Parse 1-based X/Y/Z fields to a 0-based in-bounds voxel center, or None on error."""
        if self.image_array is None:
            return None
        try:
            x = int(self.center_x_input.text()) - 1
            y = int(self.center_y_input.text()) - 1
            z = int(self.center_z_input.text()) - 1
        except ValueError:
            QMessageBox.warning(
                self,
                "Lesion Center Required",
                "Enter valid integer coordinates (X, Y, Z) and click "
                "'Set Center from Input', or click on an Initial Recon view "
                "before placing a lesion.",
            )
            return None
        dims = self.image_array.shape
        if not (0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]):
            QMessageBox.critical(
                self,
                "Out of Bounds",
                f"Coordinates outside image dimensions.\n"
                f"Valid range: x=1..{dims[0]}, y=1..{dims[1]}, z=1..{dims[2]}.",
            )
            return None
        return (x, y, z)

    def _ensure_lesion_center_for_placement(self):
        """Return a valid lesion center for save/library insert, or None after showing an error."""
        if self.image_array is None:
            QMessageBox.warning(self, "No Image", "Please perform an initial reconstruction first.")
            return None
        dims = self.image_array.shape
        if self.sphere_center is not None:
            x, y, z = self.sphere_center
            if 0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]:
                return self.sphere_center
            QMessageBox.critical(
                self,
                "Out of Bounds",
                "The current lesion center is outside the image. "
                "Set a new center before continuing.",
            )
            return None
        center = self._parse_lesion_center_from_inputs()
        if center is None:
            return None
        self.sphere_center = center
        self.current_coordinates = center
        self.center_x_input.setText(str(center[0] + 1))
        self.center_y_input.setText(str(center[1] + 1))
        self.center_z_input.setText(str(center[2] + 1))
        return center

    def show_real_lesion_library(self):
        if self.library_dialog is None:
            self.library_dialog = RealLesionLibraryDialog(self)
            self.library_dialog.lesion_selected_signal.connect(self.insert_real_lesion)

        self.library_dialog.show()
        self.library_dialog.raise_()
        self.library_dialog.activateWindow()

    def insert_real_lesion(
        self,
        lesion_array,
        region_name,
        scale_factor=1.0,
        size_scale=1.0,
        uniform_intensity=False,
        use_recon_voxel_spacing=False,
        source_spacing_mm=None,
    ):
        if self.image_array is None:
            QMessageBox.warning(self, "No Image", "Please perform an initial reconstruction first.")
            return False

        if self._ensure_lesion_center_for_placement() is None:
            return False

        lesion_array = lesion_array.copy()
        lesion_array[lesion_array < 0] = 0.0
        self.pending_real_lesion = {
            "array": lesion_array,
            "name": region_name,
            "uniform_intensity": bool(uniform_intensity),
            "use_recon_voxel_spacing": bool(use_recon_voxel_spacing),
            "source_spacing_mm": (
                tuple(float(s) for s in source_spacing_mm[:3])
                if source_spacing_mm is not None
                else None
            ),
        }

        self.intensity_input.setText(f"{float(scale_factor):.3f}".rstrip("0").rstrip("."))
        self.size_scale_input.setText(f"{float(size_scale):.3f}".rstrip("0").rstrip("."))
        self.lesion_name_input.setText(region_name)

        self._update_sphere_volume_display()
        self.update_all_views()
        self._update_ui_state()
        self._update_lesion_ui_state()
        self._sync_size_scale_visibility()
        self.status_bar.showMessage(
            f"Real lesion '{region_name}' is ready at (x={self.sphere_center[0]+1}, y={self.sphere_center[1]+1}, z={self.sphere_center[2]+1}). "
            "Click 'Save Current Lesion' to commit."
        )
        return True

    def _refresh_lesion_combo(self, selected_id=None):
        current_id = selected_id if selected_id is not None else self.lesion_select_combo.currentData()
        self.lesion_select_combo.clear()
        
        # Pre-calc spacing / grid for volume reporting
        spacing = self.get_effective_voxel_spacing_mm()
        grid_shape = self.image_array.shape if self.image_array is not None else None

        for l in self.lesion_objects:
            if not hasattr(l, 'id') or l.id is None: 
                l.id = str(uuid.uuid4())

            self._normalize_sphere_lesion_size(l)
            
            name = getattr(l, 'name', 'Lesion')
            
            # Vol Calc: grid-aligned mask on the reconstruction object grid
            vol_mm3 = 0.0
            if grid_shape is not None and spacing is not None:
                vol_mm3 = lesion_volume_ml_on_grid(grid_shape, l, spacing) * 1000.0
            
            vol_ml = vol_mm3 / 1000.0
            coords = f"(x={l.center[0]+1}, y={l.center[1]+1}, z={l.center[2]+1})"
            is_library = l.custom_shape is not None
            is_uniform_library = is_library and getattr(l, "uniform_intensity", False)
            is_real_library = is_library and not is_uniform_library

            if is_uniform_library:
                intensity_part = f"Intensity={l.intensity:.2f}"
            elif is_real_library:
                truth_mean = (
                    lesion_truth_mean_on_grid(grid_shape, l)
                    if grid_shape is not None
                    else float(l.intensity)
                )
                intensity_part = (
                    f"Intensity scale={l.intensity:.2f}, Mean={truth_mean:.2f}"
                )
            else:
                intensity_part = f"Intensity={l.intensity:.2f}"

            parts = [
                f"{name}: {coords}",
                f"Vol={vol_ml:.3f}mL",
                intensity_part,
            ]
            if is_library:
                scale = float(getattr(l, "size_scale", 1.0))
                parts.append(f"Size Scale={scale:.2f}")
            if is_uniform_library:
                parts.append("Uniform intensity")
            if not is_library and self.voxel_size_mm:
                r_mm = float(l.radius) * float(self.voxel_size_mm[0])
                parts.append(f"Radius={r_mm:.2f} mm")
            text = ", ".join(parts)

            self.lesion_select_combo.addItem(text, l.id)

        if current_id:
            for i in range(self.lesion_select_combo.count()):
                if str(self.lesion_select_combo.itemData(i)) == str(current_id):
                    self.lesion_select_combo.setCurrentIndex(i)
                    break

    def _rebuild_lesion_mask_display(self):
        if self.image_array is None: return
        self.lesion_array.fill(0)
        
        for i, l in enumerate(self.lesion_objects):
            try:
                current_scale = getattr(l, 'size_scale', 1.0)
                if hasattr(l, 'custom_shape') and l.custom_shape is not None:
                    self.lesion_array = apply_library_lesion_to_volume(self.lesion_array, l)
                else:
                    effective_radius = l.radius * current_scale
                    self.lesion_array = add_3d_spherical_lesion(
                        self.lesion_array, l.center, effective_radius, l.intensity
                    )
            except Exception as e:
                print(f"Error drawing lesion {i}: {e}")
                continue 
        
        self._update_global_color_scale()
        self.update_all_views()

    def move_queue_item_up(self):
        row = self.queue_list.currentRow()
        if row > 0:
            self.reconstruction_queue[row], self.reconstruction_queue[row-1] = self.reconstruction_queue[row-1], self.reconstruction_queue[row]
            self._refresh_queue_list()
            self.queue_list.setCurrentRow(row - 1)

    def move_queue_item_down(self):
        row = self.queue_list.currentRow()
        if 0 <= row < len(self.reconstruction_queue) - 1:
            self.reconstruction_queue[row], self.reconstruction_queue[row+1] = self.reconstruction_queue[row+1], self.reconstruction_queue[row]
            self._refresh_queue_list()
            self.queue_list.setCurrentRow(row + 1)

    def remove_selected_queue_item(self):
        current_row = self.queue_list.currentRow()
        if current_row >= 0:
            item = self.reconstruction_queue.pop(current_row)
            self.queue_list.takeItem(current_row)
            self.status_bar.showMessage(f"Removed '{item.name}' from queue.")
            self._update_lesion_ui_state()

    def _refresh_queue_list(self):
        self.queue_list.clear()
        for item in self.reconstruction_queue:
            self.queue_list.addItem(str(item))

    def _update_global_color_scale(self):
        if self.image_array is None:
            self.color_bar_levels = (0, 1)
            return
        
        all_data = [self._convert_array_to_selected_unit(self.image_array, "initial")]
        if self.lesion_array is not None: all_data.append(self.lesion_array)
        if self.final_array is not None: all_data.append(self._convert_array_to_selected_unit(self.final_array, "final"))

        valid_data = [d for d in all_data if d is not None and d.size > 0]
        if not valid_data:
            self.color_bar_levels = (0, 1)
            return
        min_val = min(np.min(d) for d in valid_data)
        max_val = max(np.max(d) for d in valid_data)
        self.color_bar_levels = (min_val, max_val if max_val > min_val else min_val + 1)

    def update_all_views(self):
        self.master_lut = pg.colormap.get('viridis').getLookupTable()

        if self.image_array is None:
            images = {'initial': None, 'ct': None, 'lesion': None, 'final': None}
        else:
            images = {
                'initial': self._convert_array_to_selected_unit(self.image_array, "initial"),
                'ct': self.registered_ct_array,
                'lesion': self.lesion_array,
                'final': self._convert_array_to_selected_unit(self.final_array, "final"),
                'occurrence_pet': self.occurrence_pet_array,
                'occurrence_ct': self.occurrence_ct_array
            }
        
        for viewer in self.viewers:
            viewer.update_view(self.current_coordinates, images, self.color_bar_levels, self.master_lut)

        self.update_sliders_and_labels(self.current_coordinates)

    def update_sliders_and_labels(self, coords):
        x, y, z = coords
        self.sagittal_slider.blockSignals(True)
        self.coronal_slider.blockSignals(True)
        self.axial_slider.blockSignals(True)
        self.sagittal_slider.setValue(x)
        self.coronal_slider.setValue(y)
        self.axial_slider.setValue(z)
        self.sagittal_slider.blockSignals(False)
        self.coronal_slider.blockSignals(False)
        self.axial_slider.blockSignals(False)
        sx_total = self.sagittal_slider.maximum() + 1
        cy_total = self.coronal_slider.maximum() + 1
        az_total = self.axial_slider.maximum() + 1
        self.sagittal_label.setText(f"Sagittal: {x+1}/{sx_total}")
        self.coronal_label.setText(f"Coronal: {y+1}/{cy_total}")
        self.axial_label.setText(f"Axial: {z+1}/{az_total}")

    def _update_coords_from_slider(self):
        x = self.sagittal_slider.value()
        y = self.coronal_slider.value()
        z = self.axial_slider.value()
        self.current_coordinates = (x, y, z)
        self.update_all_views()

    def _calculate_coords_from_view(self, viewer, view_x, view_y):
        orientation = viewer.orientation
        slice_idx = viewer.slice_index
        if orientation == 'Axial':
            return (view_x, view_y, slice_idx)
        elif orientation == 'Sagittal':
            return (slice_idx, view_x, view_y)
        elif orientation == 'Coronal':
            return (view_x, slice_idx, view_y)
        return self.current_coordinates

    def _handle_viewer_click(self, sender_viewer, x_pos, y_pos):
        self.current_coordinates = self._calculate_coords_from_view(sender_viewer, x_pos, y_pos)
        self.update_all_views()
        if sender_viewer.content_type == "Initial Recon":
            x, y, z = self.current_coordinates
            self.center_x_input.setText(str(x + 1))
            self.center_y_input.setText(str(y + 1))
            self.center_z_input.setText(str(z + 1))
            self.status_bar.showMessage(f"Coordinates ({x+1}, {y+1}, {z+1}) set. Press 'Set Center' to confirm.")

    def get_effective_voxel_spacing_mm(self):
        """
        (dx, dy, dz) in mm for volumes that share the reconstruction object grid.
        Prefer spacing set after reconstruction (tab 2); otherwise the Object Voxel Size field.
        """
        if self.voxel_size_mm is not None:
            try:
                t = [float(x) for x in self.voxel_size_mm[:3]]
            except (TypeError, ValueError, IndexError):
                t = []
            if len(t) >= 3:
                return tuple(t[:3])
            if len(t) == 2:
                return (t[0], t[1], t[1])
            if len(t) == 1:
                return (t[0], t[0], t[0])
        inp = getattr(self, "object_dr_input", None)
        if inp is None:
            return None
        try:
            parts = [p.strip() for p in inp.text().split(",") if p.strip()]
            if not parts:
                return None
            vals = [float(parts[min(i, len(parts) - 1)]) for i in range(3)]
            return tuple(vals)
        except (ValueError, IndexError):
            return None

    def _get_current_radius_for_display_px(self):
        sp = self.get_effective_voxel_spacing_mm()
        if sp is None:
            return 0
        try:
            radius_mm = float(self.radius_mm_input.text())
            return radius_mm / sp[0]
        except (ValueError, ZeroDivisionError):
            return 0

    def show_energy_windows(self):
        if not self.projection_file_path:
            QMessageBox.warning(self, "File Not Found", "Please load SPECT projection data first.")
            return
        file_to_check = self.projection_file_path
        if os.path.isdir(file_to_check):
            dicom_files = sorted([f for f in os.listdir(file_to_check) if f.lower().endswith(('.dcm', '.IMA'))])
            if not dicom_files:
                QMessageBox.warning(self, "No DICOMs", "No DICOM files found in folder.")
                return
            file_to_check = os.path.join(self.projection_file_path, dicom_files[0])
        try:
            ds = pydicom.dcmread(file_to_check, stop_before_pixels=True)
            info_str = "Found Energy Window Information:\n\n"
            if hasattr(ds, 'EnergyWindowInformationSequence'):
                for i, window in enumerate(ds.EnergyWindowInformationSequence):
                    name = window.EnergyWindowName if 'EnergyWindowName' in window else f'Window {i+1}'
                    info_str += f"Index {i}: {name}\n"
                    if 'EnergyWindowRangeSequence' in window:
                        for r in window.EnergyWindowRangeSequence:
                            info_str += f"  - Range: {r.EnergyWindowLowerLimit:.1f} - {r.EnergyWindowUpperLimit:.1f} keV\n"
            else:
                info_str = "Could not auto-detect energy windows. Common for Lu-177:\n- Idx 0: Peak\n- Idx 1/2: Scatter"
            QMessageBox.information(self, "Energy Windows", info_str)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read DICOM energy windows: {e}")

    def prepare_for_new_lesion(self):
        self.center_x_input.clear()
        self.center_y_input.clear()
        self.center_z_input.clear()
        self.lesion_name_input.clear()
        self._reset_lesion_entry_defaults()
        self.sphere_center = None
        self.pending_real_lesion = None
        for viewer in self.viewers:
            viewer.roi.setVisible(False)
        self._sync_size_scale_visibility()
        self.center_x_input.setFocus()
        self.status_bar.showMessage("Enter new lesion coordinates or click on the 'Initial Image' in Axial/Sagittal/Coronal view.")

    def set_center_from_input(self):
        if self.image_array is None:
            QMessageBox.warning(self, "No Image", "Please perform an initial reconstruction first.")
            return
        center = self._parse_lesion_center_from_inputs()
        if center is None:
            return
        self.sphere_center = center
        self.current_coordinates = center
        self.status_bar.showMessage(f"Lesion center set to (x={center[0]+1}, y={center[1]+1}, z={center[2]+1}). Adjust radius and save.")
        self._update_sphere_volume_display()
        self.update_all_views()
        self._update_ui_state()

if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = DICOMViewer()
    window.show()
    sys.exit(app.exec_())