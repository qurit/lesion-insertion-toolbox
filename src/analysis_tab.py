import numpy as np
import scipy.ndimage
import os
import json
import zlib
import base64
import SimpleITK as sitk
import hashlib

import pyqtgraph as pg

from math import cos, sin, pi

from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QEvent
from PyQt5.QtGui import QColor, QBrush, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QComboBox,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QMessageBox,
    QHeaderView,
    QFileDialog,
    QSplitter,
    QCheckBox,
    QTabWidget,
    QDialog,
)

try:
    # Normal path when running the app from the project root
    from src.utils import (
        build_lesion_mask_on_grid,
        build_lesion_truth_on_grid,
        expected_lesion_mean_on_grid,
        iter_number_for_recon_result,
        lesion_volume_ml_on_grid,
        lesion_volume_spacing_mm,
        parse_iteration_from_scenario_name,
        reference_lesion_voxels_on_grid,
    )
    from src.models import Lesion
    from src.dialogs import NoiseRoiPickerDialog
except ModuleNotFoundError:
    # Allows running this file directly (e.g. for quick UI experiments).
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.utils import (
        build_lesion_mask_on_grid,
        build_lesion_truth_on_grid,
        expected_lesion_mean_on_grid,
        iter_number_for_recon_result,
        lesion_volume_ml_on_grid,
        lesion_volume_spacing_mm,
        parse_iteration_from_scenario_name,
        reference_lesion_voxels_on_grid,
    )
    from src.models import Lesion
    from src.dialogs import NoiseRoiPickerDialog


class DataAnalysisTab(QWidget):
    """Tab-4 analysis workspace for multi-lesion / multi-scenario comparisons."""

    # Distinct ScatterPlotItem symbols used to distinguish lesions in the analysis plots.
    # Ordered so the first ten lesions all get clearly different shapes (no two
    # triangle variants in a row). After the pool is exhausted, symbols repeat
    # with an increasing size tier to stay distinguishable.
    _LESION_SYMBOL_POOL = (
        "o",      # circle
        "s",      # square
        "t",      # triangle-down
        "d",      # diamond
        "p",      # pentagon
        "h",      # hexagon
        "star",   # star
        "+",      # plus
        "x",      # cross
        "t1",     # triangle-up
        "t2",     # triangle-right
        "t3",     # triangle-left
    )

    def __init__(self, parent_main_window):
        super().__init__(parent_main_window)
        self.parent_window = parent_main_window
        self._current_records = []
        self._records_mode = None
        self._records_signature = None
        self._force_blank_plot = False
        self._scenario_data = {}
        self._imported_scenarios = {}
        self._scenario_meta = {}
        self._scenario_meta_overrides = {}
        self._last_selected_scenario_count = 0
        self._last_selected_lesion_count = 0
        self._legend_aliases = {}
        self._figure_tabs_context = []
        self._legend_refresh_in_progress = False
        self._legend_replot_timer = QTimer(self)
        self._legend_replot_timer.setSingleShot(True)
        self._legend_replot_timer.timeout.connect(self.plot_current_results)
        self._noise_roi_center = None
        self._noise_roi_center_mm = None
        self._noise_roi_radius_vox = 5.0
        self._noise_roi_radius_mm = 14.0
        self._noise_roi_template_key = None
        self._metric_labels = {
            "__none__": "None",
            "__sample_index__": "Sample Index",
            "volume_ml": "Lesion Volume (mL)",
            "rc_mean": "Recovery Coefficient",
            "bias_percent": "Bias %",
            "abs_bias_percent": "Absolute Bias (%)",
            "noise_cv_percent": "Noise (%)",
            "measured_mean": "Measured Mean Intensity",
            "expected_mean": "Expected Mean Intensity",
            "cnr": "Contrast-to-Noise Ratio (CNR)",
            "lesion_radius_px": "Lesion Radius (px)",
            "iter_number": "Iteration Number",
            "filter_fwhm_mm": "Filter FWHM (mm)",
            "peak": "Peak Uptake (SUVmax)",
            "mean": "Mean Intensity",
            "std": "Standard Deviation",
            "cv_percent": "Coefficient of Variation (%)",
            "entropy": "Entropy",
            "energy": "Energy",
            "p10": "10th Percentile",
            "median": "Median",
            "p90": "90th Percentile (SUV90)",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "overshoot_percent": "Overshoot %",
            "undershoot_percent": "Undershoot %",
            "ga_percent": "Gibbs Activity GA %",
            "edge_window_mm": "Edge Window w (mm)",
            "ground_truth_peak": "Reference Peak Uptake (SUVmax)",
            "ground_truth_mean": "Reference Mean Intensity",
            "ground_truth_std": "Reference Standard Deviation",
            "ground_truth_cv_percent": "Reference Coefficient of Variation (%)",
            "ground_truth_entropy": "Reference Entropy",
            "ground_truth_energy": "Reference Energy",
            "ground_truth_p10": "Reference 10th Percentile",
            "ground_truth_median": "Reference Median",
            "ground_truth_p90": "Reference 90th Percentile (SUV90)",
            "ground_truth_skewness": "Reference Skewness",
            "ground_truth_kurtosis": "Reference Kurtosis",
            "dsi_percent": "Detection Stability Index",
        }
        self._axis_options = {
            "Recovery Coefficients": {
                "x": ["iter_number", "volume_ml"],
                "y": ["rc_mean", "measured_mean", "bias_percent"],
            },
            "Noise-Bias Curve": {
                "x": ["bias_percent", "noise_cv_percent", "iter_number", "volume_ml"],
                "y": ["noise_cv_percent", "bias_percent"],
            },
            "Post-Recon Filtering Effect": {
                "x": ["filter_fwhm_mm"],
                "y": ["rc_mean", "cnr", "noise_cv_percent", "bias_percent", "measured_mean"],
            },
            "Gibbs Effect": {
                "x": ["iter_number", "volume_ml", "expected_mean", "measured_mean"],
                "y": [
                    "overshoot_percent",
                    "undershoot_percent",
                    "ga_percent",
                ],
            },
            "Radiomics": {
                "x": ["iter_number", "volume_ml", "__none__"],
                "y": [
                    "__none__",
                    "dsi_percent",
                    "peak",
                    "ground_truth_peak",
                    "p90",
                    "ground_truth_p90",
                    "entropy",
                    "ground_truth_entropy",
                    "mean",
                    "ground_truth_mean",
                    "std",
                    "ground_truth_std",
                    "cv_percent",
                    "ground_truth_cv_percent",
                    "energy",
                    "ground_truth_energy",
                    "p10",
                    "ground_truth_p10",
                    "median",
                    "ground_truth_median",
                    "skewness",
                    "ground_truth_skewness",
                    "kurtosis",
                    "ground_truth_kurtosis",
                ],
            },
        }
        self._default_axes = {
            "Recovery Coefficients": ("iter_number", "rc_mean"),
            "Noise-Bias Curve": ("bias_percent", "noise_cv_percent"),
            "Post-Recon Filtering Effect": ("filter_fwhm_mm", "cnr"),
            "Gibbs Effect": ("iter_number", "overshoot_percent"),
            "Radiomics": ("iter_number", "dsi_percent"),
        }
        self._build_ui()
        self.refresh_sources()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setMinimumWidth(680)

        controls_group = QGroupBox("Analysis Controls")
        controls_group.setMinimumWidth(660)
        controls_layout = QVBoxLayout(controls_group)

        selectors_row = QHBoxLayout()

        scenario_group = QGroupBox("Reconstruction Scenarios")
        scenario_layout = QVBoxLayout(scenario_group)
        self.scenario_search = QLineEdit()
        self.scenario_search.setPlaceholderText("Search scenarios...")
        scenario_layout.addWidget(self.scenario_search)
        scenario_btns = QHBoxLayout()
        self.scenario_select_all_btn = QPushButton("Select all")
        self.scenario_select_none_btn = QPushButton("Select none")
        self.scenario_import_btn = QPushButton("Import Image")
        self.scenario_export_btn = QPushButton("Export Image")
        scenario_btns.addWidget(self.scenario_select_all_btn)
        scenario_btns.addWidget(self.scenario_select_none_btn)
        scenario_btns.addWidget(self.scenario_import_btn)
        scenario_btns.addWidget(self.scenario_export_btn)
        scenario_layout.addLayout(scenario_btns)
        self.scenario_list = QListWidget()
        self.scenario_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.scenario_list.setMinimumHeight(220)
        scenario_layout.addWidget(self.scenario_list, 1)
        selectors_row.addWidget(scenario_group, 2)

        lesion_group = QGroupBox("Lesions")
        lesion_layout = QVBoxLayout(lesion_group)
        self.lesion_search = QLineEdit()
        self.lesion_search.setPlaceholderText("Search lesions...")
        lesion_layout.addWidget(self.lesion_search)
        lesion_btns = QHBoxLayout()
        self.lesion_select_all_btn = QPushButton("Select all")
        self.lesion_select_none_btn = QPushButton("Select none")
        lesion_btns.addWidget(self.lesion_select_all_btn)
        lesion_btns.addWidget(self.lesion_select_none_btn)
        lesion_layout.addLayout(lesion_btns)
        self.lesion_list = QListWidget()
        self.lesion_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lesion_list.setMinimumHeight(220)
        lesion_layout.addWidget(self.lesion_list, 1)
        selectors_row.addWidget(lesion_group, 1)

        controls_layout.addLayout(selectors_row)

        mode_group = QGroupBox("Mode")
        mode_form = QFormLayout(mode_group)
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(
            [
                "Recovery Coefficients",
                "Noise-Bias Curve",
                "Post-Recon Filtering Effect",
                "Gibbs Effect",
                "Radiomics",
            ]
        )
        self.radiomics_feature_combo = QComboBox()
        # Multi-checkable model: the combo defines which first-order features are
        # pooled into the Detection Stability Index (DSI). Each item is checkable
        # and toggling does not close the popup. Item 0 acts as "Select all / none"
        # for convenience. The default selection is the literature-grounded
        # detection panel { SUVmax (peak), SUV90 (p90), first-order entropy }
        # (EANM 2.0 / PERCIST for SUVmax; Tixier 2011 / Pfaehler 2019 for SUV90 as
        # the noise-robust peak surrogate; Tixier 2011 / Cook 2013 / Hatt 2017 for
        # first-order entropy as the canonical distribution-separation feature).
        # The user can pick any combination; this default is what the paper's
        # methods section reports.
        self._radiomics_feature_keys = [
            "peak", "p90", "entropy", "mean", "std", "cv_percent",
            "energy", "p10", "median", "skewness", "kurtosis",
        ]
        self._radiomics_default_panel = {"peak", "p90", "entropy"}
        rf_model = QStandardItemModel(self.radiomics_feature_combo)
        select_all_item = QStandardItem("(Select all / none)")
        select_all_item.setData("__select_all__", Qt.UserRole)
        select_all_item.setCheckable(True)
        # Master starts PartiallyChecked because only a subset of features
        # is checked by default; clicking it once will toggle all.
        select_all_item.setCheckState(Qt.PartiallyChecked)
        rf_model.appendRow(select_all_item)
        for k in self._radiomics_feature_keys:
            item = QStandardItem(self._metric_labels.get(k, k))
            item.setData(k, Qt.UserRole)
            item.setCheckable(True)
            item.setCheckState(Qt.Checked if k in self._radiomics_default_panel else Qt.Unchecked)
            rf_model.appendRow(item)
        self.radiomics_feature_combo.setModel(rf_model)
        self.radiomics_feature_combo.setEditable(True)
        self.radiomics_feature_combo.lineEdit().setReadOnly(True)
        self.radiomics_feature_combo.lineEdit().setFocusPolicy(Qt.NoFocus)
        self.radiomics_feature_combo.setToolTip(
            "First-order features that contribute to the Detection Stability Index (DSI). "
            "Default (paper) panel: SUVmax (peak), SUV90 (p90), first-order entropy — "
            "the literature-supported detection-representing features. "
            "Click to add/remove features; the first row toggles all. "
            "Within each record, features whose ground-truth value is degenerate "
            "(e.g. entropy / energy / std on a uniform-intensity sphere) are skipped "
            "automatically so DSI averages only over well-defined features."
        )
        # Track previous check-states so itemChanged can detect transitions and
        # propagate the master-toggle without infinite recursion.
        self._radiomics_combo_updating = False
        rf_model.itemChanged.connect(self._on_radiomics_feature_item_changed)
        self.radiomics_feature_combo.view().viewport().installEventFilter(self)
        self.radiomics_feature_combo.installEventFilter(self)
        self._refresh_radiomics_feature_combo_text()
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        self.show_legend_checkbox = QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        mode_form.addRow("Analysis type:", self.analysis_mode_combo)
        mode_form.addRow("X-axis:", self.x_axis_combo)
        mode_form.addRow("Y-axis:", self.y_axis_combo)
        mode_form.addRow("Radiomics plot feature:", self.radiomics_feature_combo)
        self.filter_controls_widget = QWidget()
        filter_form = QFormLayout(self.filter_controls_widget)
        filter_form.setContentsMargins(0, 0, 0, 0)
        self.filter_start_input = QLineEdit("0")
        self.filter_stop_input = QLineEdit("8")
        self.filter_step_input = QLineEdit("1")
        filter_form.addRow("Filter sweep start (mm):", self.filter_start_input)
        filter_form.addRow("Filter sweep stop (mm):", self.filter_stop_input)
        filter_form.addRow("Filter sweep step (mm):", self.filter_step_input)
        mode_form.addRow("", self.filter_controls_widget)
        mode_form.addRow("", self.show_legend_checkbox)

        self.noise_bias_scale_widget = QWidget()
        nbs_lay = QVBoxLayout(self.noise_bias_scale_widget)
        nbs_lay.setContentsMargins(0, 0, 0, 0)
        nbs_row = QHBoxLayout()
        self.noise_bias_scale_combo = QComboBox()
        self.noise_bias_scale_combo.addItem("Fit data (expand differences)", "fit")
        self.noise_bias_scale_combo.addItem("From origin (0 to max)", "zero")
        self.noise_bias_scale_combo.addItem("Manual limits", "manual")
        self.noise_bias_scale_combo.setToolTip(
            "Noise-Bias Curve only: how to set the X/Y axis ranges.\n"
            "• Fit data — zoom to the plotted values (best for bias vs noise).\n"
            "• From origin — keep (0,0) as lower corner like a full-range plot.\n"
            "• Manual — type min/max below; leave a field empty to auto-fill from data."
        )
        nbs_row.addWidget(QLabel("Noise-bias axes:"))
        nbs_row.addWidget(self.noise_bias_scale_combo, 1)
        nbs_lay.addLayout(nbs_row)
        self.noise_bias_manual_widget = QWidget()
        nb_man = QHBoxLayout(self.noise_bias_manual_widget)
        nb_man.setContentsMargins(0, 0, 0, 0)
        self.noise_bias_xmin_edit = QLineEdit()
        self.noise_bias_xmax_edit = QLineEdit()
        self.noise_bias_ymin_edit = QLineEdit()
        self.noise_bias_ymax_edit = QLineEdit()
        for ed, ph in (
            (self.noise_bias_xmin_edit, "X min"),
            (self.noise_bias_xmax_edit, "X max"),
            (self.noise_bias_ymin_edit, "Y min"),
            (self.noise_bias_ymax_edit, "Y max"),
        ):
            ed.setPlaceholderText(ph)
            ed.setMaximumWidth(92)
        nb_man.addWidget(QLabel("X:"))
        nb_man.addWidget(self.noise_bias_xmin_edit)
        nb_man.addWidget(self.noise_bias_xmax_edit)
        nb_man.addSpacing(8)
        nb_man.addWidget(QLabel("Y:"))
        nb_man.addWidget(self.noise_bias_ymin_edit)
        nb_man.addWidget(self.noise_bias_ymax_edit)
        nb_man.addStretch()
        nbs_lay.addWidget(self.noise_bias_manual_widget)
        self.noise_bias_manual_widget.setVisible(False)
        self.noise_reference_mode_combo = QComboBox()
        self.noise_reference_mode_combo.addItem("Global Liver ROI", "global_liver")
        self.noise_reference_mode_combo.addItem("Per Lesion Ring", "per_lesion_ring")
        self.noise_reference_mode_combo.setCurrentIndex(0)
        self.noise_reference_mode_combo.setToolTip(
            "How background noise (CV%) is measured for Noise-Bias Curve.\n"
            "• Global liver ROI — one spherical ROI in liver; same noise for all lesions.\n"
            "• Per-lesion ring — 3-voxel shell around each lesion."
        )
        nbs_lay.addWidget(self.noise_reference_mode_combo)
        self.noise_roi_picker_widget = QWidget()
        roi_picker_lay = QVBoxLayout(self.noise_roi_picker_widget)
        roi_picker_lay.setContentsMargins(0, 4, 0, 0)
        self.noise_roi_define_btn = QPushButton("Define liver noise ROI…")
        self.noise_roi_define_btn.setToolTip(
            "Place the liver ROI on initial reconstruction + lesion mask (preview only). "
            "Saved center/radius (mm) are reused on every selected reconstruction; "
            "noise CV% always reads voxels from that scenario's volume, not the preview."
        )
        roi_picker_lay.addWidget(self.noise_roi_define_btn)
        nbs_lay.addWidget(self.noise_roi_picker_widget)
        self.noise_roi_picker_widget.setVisible(False)
        mode_form.addRow("", self.noise_bias_scale_widget)

        self.noise_roi_define_btn.clicked.connect(self._open_noise_roi_dialog)

        self.noise_bias_scale_combo.currentIndexChanged.connect(self._on_noise_bias_scale_changed)
        self.noise_reference_mode_combo.currentIndexChanged.connect(self._on_noise_reference_mode_changed)
        for _ed in (
            self.noise_bias_xmin_edit,
            self.noise_bias_xmax_edit,
            self.noise_bias_ymin_edit,
            self.noise_bias_ymax_edit,
        ):
            _ed.editingFinished.connect(self.plot_current_results)

        controls_layout.addWidget(mode_group)

        buttons_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Sources")
        self.run_btn = QPushButton("Run Analysis")
        self.export_csv_btn = QPushButton("Export CSV")
        self.save_session_btn = QPushButton("Save Session (JSON)")
        self.load_session_btn = QPushButton("Load Session (JSON)")
        buttons_row.addWidget(self.refresh_btn)
        buttons_row.addWidget(self.run_btn)
        buttons_row.addWidget(self.export_csv_btn)
        buttons_row.addWidget(self.save_session_btn)
        buttons_row.addWidget(self.load_session_btn)
        buttons_row.addStretch()
        controls_layout.addLayout(buttons_row)
        left_layout.addWidget(controls_group, 1)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        output_group = QGroupBox("Results")
        output_layout = QVBoxLayout(output_group)
        results_top_row = QHBoxLayout()
        results_top_row.addStretch()
        self.new_figure_btn = QPushButton("New Figure")
        results_top_row.addWidget(self.new_figure_btn)
        output_layout.addLayout(results_top_row)
        self.figure_tabs = QTabWidget()
        self.figure_tabs.setTabsClosable(True)
        self.figure_tabs.tabCloseRequested.connect(self._close_figure_tab)
        self.figure_tabs.setMinimumHeight(360)
        output_layout.addWidget(self.figure_tabs, 2)
        self._add_figure_tab("Figure 1")
        self.analysis_summary_label = QLabel("Run summary will appear here.")
        self.analysis_summary_label.setStyleSheet("color: #A3BE8C; font-size: 10pt;")
        output_layout.addWidget(self.analysis_summary_label)
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(False)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.setStyleSheet(
            "QTableWidget { background-color: #434C5E; alternate-background-color: #434C5E; gridline-color: #4C566A; }"
            "QTableWidget::item { background-color: #434C5E; color: #D8DEE9; }"
            "QTableWidget::item:selected { background-color: #5E81AC; color: #ECEFF4; }"
        )
        output_layout.addWidget(self.results_table, 3)
        right_layout.addWidget(output_group, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 760])

        self.refresh_btn.clicked.connect(self.refresh_sources)
        self.run_btn.clicked.connect(self.run_analysis)
        self.new_figure_btn.clicked.connect(lambda: self._add_figure_tab(f"Figure {self.figure_tabs.count() + 1}"))
        self.export_csv_btn.clicked.connect(self.export_current_table_csv)
        self.save_session_btn.clicked.connect(self.save_analysis_session_json)
        self.load_session_btn.clicked.connect(self.load_analysis_session_json)
        self.scenario_import_btn.clicked.connect(self.import_scenario_image)
        self.scenario_export_btn.clicked.connect(self.export_selected_scenarios_nifti)
        self.scenario_select_all_btn.clicked.connect(lambda: self._select_all_in_list(self.scenario_list))
        self.scenario_select_none_btn.clicked.connect(lambda: self._select_none_in_list(self.scenario_list))
        self.lesion_select_all_btn.clicked.connect(lambda: self._select_all_in_list(self.lesion_list))
        self.lesion_select_none_btn.clicked.connect(lambda: self._select_none_in_list(self.lesion_list))
        self.scenario_search.textChanged.connect(lambda _: self._apply_search_filter(self.scenario_list, self.scenario_search.text()))
        self.lesion_search.textChanged.connect(lambda _: self._apply_search_filter(self.lesion_list, self.lesion_search.text()))
        self.scenario_list.itemSelectionChanged.connect(self._analysis_inputs_changed)
        self.lesion_list.itemSelectionChanged.connect(self._analysis_inputs_changed)
        for _filter_ed in (
            self.filter_start_input,
            self.filter_stop_input,
            self.filter_step_input,
        ):
            _filter_ed.editingFinished.connect(self._analysis_inputs_changed)
        self.show_legend_checkbox.toggled.connect(self.plot_current_results)
        self.analysis_mode_combo.currentTextChanged.connect(self._on_analysis_mode_changed)
        self.x_axis_combo.currentTextChanged.connect(self.plot_current_results)
        self.y_axis_combo.currentTextChanged.connect(self.plot_current_results)
        self.radiomics_feature_combo.currentIndexChanged.connect(self._sync_radiomics_axis_choice)
        self._update_mode_visibility()

    def _start_legend_item_edit(self, item):
        # Item can be invalid if a full replot cleared the legend mid-gesture.
        try:
            lw = item.listWidget()
        except RuntimeError:
            return
        if lw is None:
            return
        try:
            key = item.data(Qt.UserRole)
        except RuntimeError:
            return
        if not key or str(key).startswith("__iter_"):
            return
        try:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            lw.editItem(item)
        except RuntimeError:
            return

    def _legend_item_changed(self, item):
        if self._legend_refresh_in_progress:
            return
        try:
            key = item.data(Qt.UserRole)
        except RuntimeError:
            return
        if not key or str(key).startswith("__iter_"):
            return
        new_name = item.text().strip()
        if not new_name:
            new_name = str(key)
            try:
                item.setText(new_name)
            except RuntimeError:
                return
        self._legend_aliases[str(key)] = new_name
        self._legend_replot_timer.stop()
        self._legend_replot_timer.start(450)

    def _update_mode_visibility(self):
        mode = self.analysis_mode_combo.currentText()
        is_filter = mode == "Post-Recon Filtering Effect"
        is_radiomics = mode == "Radiomics"
        self.filter_controls_widget.setVisible(is_filter)
        self.radiomics_feature_combo.setVisible(is_radiomics)
        axis_cfg = self._axis_options.get(mode, {"x": ["volume_ml"], "y": ["rc_mean"]})
        current_x = self.x_axis_combo.currentData()
        current_y = self.y_axis_combo.currentData()
        self.x_axis_combo.blockSignals(True)
        self.y_axis_combo.blockSignals(True)
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        for k in axis_cfg["x"]:
            self.x_axis_combo.addItem(self._metric_labels.get(k, k), k)
        for k in axis_cfg["y"]:
            self.y_axis_combo.addItem(self._metric_labels.get(k, k), k)
        mode_defaults = self._default_axes.get(mode, (axis_cfg["x"][0], axis_cfg["y"][0]))
        x_default = mode_defaults[0] if mode_defaults[0] in axis_cfg["x"] else axis_cfg["x"][0]
        y_default = mode_defaults[1] if mode_defaults[1] in axis_cfg["y"] else axis_cfg["y"][0]
        if current_x in axis_cfg["x"] and mode != "Noise-Bias Curve":
            x_default = current_x
        if current_y in axis_cfg["y"] and mode != "Noise-Bias Curve":
            y_default = current_y
        self.x_axis_combo.setCurrentIndex(max(0, self.x_axis_combo.findData(x_default)))
        self.y_axis_combo.setCurrentIndex(max(0, self.y_axis_combo.findData(y_default)))
        self.x_axis_combo.blockSignals(False)
        self.y_axis_combo.blockSignals(False)
        if is_radiomics:
            self._sync_radiomics_axis_choice()
        is_nb = mode == "Noise-Bias Curve"
        self.noise_bias_scale_widget.setVisible(is_nb)
        if is_nb:
            self._sync_noise_bias_manual_visibility()
            self._sync_noise_reference_lesion_visibility()

    def _on_noise_reference_mode_changed(self):
        self._sync_noise_reference_lesion_visibility()
        self._analysis_inputs_changed()

    def _sync_noise_reference_lesion_visibility(self):
        if not hasattr(self, "noise_roi_picker_widget"):
            return
        use_global = (
            hasattr(self, "noise_reference_mode_combo")
            and self.noise_reference_mode_combo.currentData() == "global_liver"
        )
        self.noise_roi_picker_widget.setVisible(use_global)

    def _on_noise_bias_scale_changed(self):
        self._sync_noise_bias_manual_visibility()
        self.plot_current_results()

    def _sync_noise_bias_manual_visibility(self):
        if not hasattr(self, "noise_bias_manual_widget"):
            return
        mode = self.noise_bias_scale_combo.currentData() if hasattr(self, "noise_bias_scale_combo") else None
        self.noise_bias_manual_widget.setVisible(mode == "manual")

    def _parse_noise_bias_optional_float(self, line_edit):
        s = str(line_edit.text()).strip() if line_edit is not None else ""
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _finite_min_max(self, arr):
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi <= lo:
            c = lo
            span = max(abs(c), 1.0) * 0.05
            return c - span, c + span
        return lo, hi

    def _noise_bias_padded_range(self, lo, hi, key, pad_frac=0.08):
        """Expand [lo,hi] by pad_frac of span; iteration axis gets half-step padding."""
        if key == "iter_number":
            span = max(hi - lo, 1e-9)
            pad = max(0.5, min(0.15 * span, 2.0))
            return max(0.5, lo - pad), hi + pad
        span = max(hi - lo, 1e-9)
        pad = max(span * pad_frac, 1e-6)
        return lo - pad, hi + pad

    def _noise_bias_nonneg_floor(self, key, lo):
        if key in ("bias_percent", "noise_cv_percent", "volume_ml"):
            return max(0.0, lo)
        return lo

    def _noise_bias_axis_ranges(
        self,
        x_flat,
        y_flat,
        x_key,
        y_key,
    ):
        """
        Returns (x_min, x_max, y_min, y_max) for Noise-Bias Curve view.
        """
        xf_lo, xf_hi = self._finite_min_max(x_flat)
        yf_lo, yf_hi = self._finite_min_max(y_flat)
        xf_lo = self._noise_bias_nonneg_floor(x_key, xf_lo)
        yf_lo = self._noise_bias_nonneg_floor(y_key, yf_lo)
        xf_lo, xf_hi = self._noise_bias_padded_range(xf_lo, xf_hi, x_key)
        yf_lo, yf_hi = self._noise_bias_padded_range(yf_lo, yf_hi, y_key)
        if x_key in ("noise_cv_percent", "volume_ml"):
            xf_lo = max(0.0, xf_lo)
        elif x_key == "bias_percent":
            xf_lo = max(0.0, xf_lo)
        if y_key in ("noise_cv_percent", "volume_ml"):
            yf_lo = max(0.0, yf_lo)

        mode = self.noise_bias_scale_combo.currentData() if hasattr(self, "noise_bias_scale_combo") else "fit"
        if mode == "zero":
            zx_lo = 0.0 if x_key == "iter_number" else max(0.0, self._noise_bias_nonneg_floor(x_key, 0.0))
            zy_lo = 0.0 if y_key == "iter_number" else max(0.0, self._noise_bias_nonneg_floor(y_key, 0.0))
            xm = float(np.nanmax(x_flat)) if np.asarray(x_flat).size else 0.0
            ym = float(np.nanmax(y_flat)) if np.asarray(y_flat).size else 0.0
            return max(0.0, zx_lo), max(1.0, xm) * 1.05, max(0.0, zy_lo), max(1.0, ym) * 1.05

        if mode == "manual":
            x_lo = self._parse_noise_bias_optional_float(self.noise_bias_xmin_edit)
            x_hi = self._parse_noise_bias_optional_float(self.noise_bias_xmax_edit)
            y_lo = self._parse_noise_bias_optional_float(self.noise_bias_ymin_edit)
            y_hi = self._parse_noise_bias_optional_float(self.noise_bias_ymax_edit)
            if x_lo is None:
                x_lo = xf_lo
            if x_hi is None:
                x_hi = xf_hi
            if y_lo is None:
                y_lo = yf_lo
            if y_hi is None:
                y_hi = yf_hi
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
            if y_lo > y_hi:
                y_lo, y_hi = y_hi, y_lo
            return x_lo, x_hi, y_lo, y_hi

        return xf_lo, xf_hi, yf_lo, yf_hi

    def _sync_radiomics_axis_choice(self):
        # Qt re-syncs the combo's edit line whenever the current row changes
        # (e.g. when popup navigation highlights another row); re-apply our
        # summary so the displayed text reflects the multi-selection.
        self._refresh_radiomics_feature_combo_text()
        if self.analysis_mode_combo.currentText() != "Radiomics":
            return
        # Keep Y-axis fully user-controlled (supports "None").
        self.plot_current_results()

    def _selected_radiomics_features(self):
        """Return the user-checked first-order feature keys (no '__select_all__').

        Falls back to ['mean'] when nothing is checked so that single-feature
        plotting (X=None, Y=None) still has something to show.
        """
        keys = []
        try:
            model = self.radiomics_feature_combo.model()
        except (AttributeError, RuntimeError):
            return list(getattr(self, "_radiomics_feature_keys", ["mean"]))
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data == "__select_all__":
                continue
            if item.checkState() == Qt.Checked:
                keys.append(str(data))
        if keys:
            return keys
        return ["mean"] if "mean" in getattr(self, "_radiomics_feature_keys", []) else list(getattr(self, "_radiomics_feature_keys", []))[:1]

    def _set_selected_radiomics_features(self, keys):
        """Restore a multi-selection (used by session load and the master toggle)."""
        wanted = set(str(k) for k in (keys or []))
        try:
            model = self.radiomics_feature_combo.model()
        except (AttributeError, RuntimeError):
            return
        self._radiomics_combo_updating = True
        try:
            n_feat = 0
            n_checked = 0
            for i in range(model.rowCount()):
                item = model.item(i)
                if item is None:
                    continue
                data = item.data(Qt.UserRole)
                if data == "__select_all__":
                    continue
                n_feat += 1
                state = Qt.Checked if str(data) in wanted else Qt.Unchecked
                item.setCheckState(state)
                if state == Qt.Checked:
                    n_checked += 1
            master = model.item(0)
            if master is not None and master.data(Qt.UserRole) == "__select_all__":
                if n_checked == 0:
                    master.setCheckState(Qt.Unchecked)
                elif n_checked == n_feat:
                    master.setCheckState(Qt.Checked)
                else:
                    master.setCheckState(Qt.PartiallyChecked)
        finally:
            self._radiomics_combo_updating = False
        self._refresh_radiomics_feature_combo_text()

    def _on_radiomics_feature_item_changed(self, item):
        """Mirror the master toggle, keep the line edit summary fresh, and refresh outputs."""
        if self._radiomics_combo_updating or item is None:
            return
        data = item.data(Qt.UserRole)
        if data == "__select_all__":
            wanted = self._radiomics_feature_keys if item.checkState() != Qt.Unchecked else []
            self._set_selected_radiomics_features(wanted)
        else:
            self._radiomics_combo_updating = True
            try:
                model = self.radiomics_feature_combo.model()
                n_feat = 0
                n_checked = 0
                for i in range(model.rowCount()):
                    it = model.item(i)
                    if it is None or it.data(Qt.UserRole) == "__select_all__":
                        continue
                    n_feat += 1
                    if it.checkState() == Qt.Checked:
                        n_checked += 1
                master = model.item(0)
                if master is not None and master.data(Qt.UserRole) == "__select_all__":
                    if n_checked == 0:
                        master.setCheckState(Qt.Unchecked)
                    elif n_checked == n_feat:
                        master.setCheckState(Qt.Checked)
                    else:
                        master.setCheckState(Qt.PartiallyChecked)
            finally:
                self._radiomics_combo_updating = False
            self._refresh_radiomics_feature_combo_text()
        self._refresh_radiomics_outputs()

    def _refresh_radiomics_feature_combo_text(self):
        """Show a 'N/M selected' summary in the combo's read-only line edit."""
        try:
            le = self.radiomics_feature_combo.lineEdit()
        except (AttributeError, RuntimeError):
            return
        if le is None:
            return
        total = len(getattr(self, "_radiomics_feature_keys", []) or [])
        sel = self._selected_radiomics_features()
        n = len(sel)
        if total == 0:
            le.setText("(no features)")
        elif n == total:
            le.setText(f"All features ({total})")
        elif n == 0:
            le.setText("No features selected")
        elif n <= 3:
            labels = [self._metric_labels.get(k, k) for k in sel]
            le.setText(", ".join(labels))
        else:
            le.setText(f"{n} / {total} features selected")

    def _refresh_radiomics_outputs(self):
        """Re-populate the Radiomics table and replot when the DSI panel changes."""
        if self.analysis_mode_combo.currentText() != "Radiomics":
            return
        records_mode = self._records_mode or self.analysis_mode_combo.currentText()
        if records_mode == "Radiomics" and self._current_records:
            self._populate_table(records_mode, self._current_records)
        self.plot_current_results()

    def _dsi_for_record(self, rec, feature_keys=None):
        """Detection Stability Index for one record over the chosen feature panel.

        DSI_c = (1/|S_eff|) Σ_{i∈S_eff} |x_{i,c} − x^GT_i| / |x^GT_i| × 100%

        S_eff is the subset of selected features whose ground-truth value is
        well-defined for this record (i.e. finite and not degenerate-near-zero).
        Features whose reference value is degenerate (e.g. entropy / energy /
        std on a uniform-intensity sphere) are skipped at this average so they
        cannot inflate or suppress DSI.

        DSI is therefore meaningful for both patient-derived library lesions
        (all panel features typically contribute) and uniform spheres (where
        the panel naturally reduces to the SUV-class features SUVmax, SUV90,
        SUVmean — those are perfectly well-defined on a sphere). Returns NaN
        only when no panel feature is well-defined for this record.
        """
        if feature_keys is None:
            feature_keys = self._selected_radiomics_features()
        vals = []
        for k in feature_keys:
            v = rec.get(f"stability_{k}")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                vals.append(fv)
        return float(np.mean(vals)) if vals else float("nan")

    def eventFilter(self, source, event):
        """Keep the radiomics-feature combo popup open and toggle item check-state on click."""
        try:
            combo = self.radiomics_feature_combo
        except AttributeError:
            return super().eventFilter(source, event)
        if combo is None:
            return super().eventFilter(source, event)
        try:
            view = combo.view()
            viewport = view.viewport() if view is not None else None
        except (AttributeError, RuntimeError):
            view = None
            viewport = None
        if viewport is not None and source is viewport and event.type() == QEvent.MouseButtonRelease:
            try:
                idx = view.indexAt(event.pos())
            except (AttributeError, RuntimeError):
                idx = None
            if idx is not None and idx.isValid():
                item = combo.model().itemFromIndex(idx)
                if item is not None and item.isCheckable():
                    new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                    item.setCheckState(new_state)
                return True
        return super().eventFilter(source, event)

    def _reset_results_view(self):
        if hasattr(self, "results_table") and self.results_table is not None:
            self.results_table.setSortingEnabled(False)
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.results_table.setSortingEnabled(True)
        if hasattr(self, "analysis_summary_label") and self.analysis_summary_label is not None:
            self.analysis_summary_label.setText("Run summary will appear here.")

    def _invalidate_analysis_results(self, summary_message=None):
        """Clear stored metrics, table, and plots — required after analysis inputs change."""
        self._current_records = []
        self._records_mode = None
        self._records_signature = None
        self._force_blank_plot = True
        self._reset_results_view()
        if summary_message and hasattr(self, "analysis_summary_label"):
            self.analysis_summary_label.setText(summary_message)

    def _analysis_inputs_changed(self):
        """Drop stale table/plot when scenarios, lesions, or mode-specific inputs change."""
        if self._records_signature is None:
            return
        if self._current_analysis_signature() == self._records_signature:
            return
        self._invalidate_analysis_results(
            "Settings changed. Click Run Analysis to refresh results."
        )
        self.plot_current_results()

    def clear_all_workflow_state(self):
        """Reset Tab 4 to the same baseline as after a first initial reconstruction."""
        self._scenario_data = {}
        self._imported_scenarios = {}
        self._scenario_meta = {}
        self._scenario_meta_overrides = {}
        self._legend_aliases = {}
        self._noise_roi_center = None
        self._noise_roi_center_mm = None
        self._noise_roi_radius_vox = 5.0
        self._noise_roi_radius_mm = 14.0
        self._noise_roi_template_key = None
        self._invalidate_analysis_results()
        self.refresh_sources()
        self.plot_current_results()

    def _on_analysis_mode_changed(self):
        self._invalidate_analysis_results(
            "Settings changed. Click Run Analysis to refresh results."
        )
        self._update_mode_visibility()
        self.plot_current_results()

    def _add_figure_tab(self, title):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        top = QHBoxLayout()
        top.addStretch()
        save_btn = QPushButton("Save Figure")
        top.addWidget(save_btn)
        lay.addLayout(top)

        capture_widget = QWidget()
        plot_row = QHBoxLayout(capture_widget)
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground((46, 52, 64))
        plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._apply_analysis_plot_axis_style(plot_widget)
        plot_row.addWidget(plot_widget, 4)

        legend_list = QListWidget()
        legend_list.setMinimumWidth(240)
        legend_list.setMaximumWidth(320)
        legend_list.setIconSize(QSize(22, 18))
        legend_list.setToolTip(
            "Algorithms: colored squares. Lesions: marker shape. Double-click a row to rename."
        )
        legend_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        legend_list.setStyleSheet(
            "QListWidget { background-color: #3B4252; border: 1px solid #4C566A; }"
            "QListWidget::item { color: #D8DEE9; padding: 4px; }"
        )
        plot_row.addWidget(legend_list, 1)
        lay.addWidget(capture_widget, 1)

        idx = self.figure_tabs.addTab(tab, title)
        self._figure_tabs_context.insert(
            idx,
            {
                "tab": tab,
                "plot": plot_widget,
                "legend": legend_list,
                "capture": capture_widget,
            },
        )
        self.figure_tabs.setCurrentIndex(idx)

        save_btn.clicked.connect(lambda _=None, cw=capture_widget: self._save_figure_widget(cw))
        legend_list.itemDoubleClicked.connect(self._start_legend_item_edit)
        legend_list.itemChanged.connect(self._legend_item_changed)
        self._reset_results_view()

    def _close_figure_tab(self, index):
        if self.figure_tabs.count() <= 1:
            QMessageBox.information(self, "Cannot Close", "At least one figure tab must remain open.")
            return
        widget = self.figure_tabs.widget(index)
        self.figure_tabs.removeTab(index)
        if 0 <= index < len(self._figure_tabs_context):
            self._figure_tabs_context.pop(index)
        if widget is not None:
            widget.deleteLater()

    def _select_all_in_list(self, list_widget):
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(True)

    def _select_none_in_list(self, list_widget):
        list_widget.clearSelection()

    def _apply_search_filter(self, list_widget, text):
        t = (text or "").strip().lower()
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def import_scenario_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import scenario image(s)",
            "",
            "Supported images (*.npy *.nii *.nii.gz *.mha *.mhd *.nrrd);;NumPy array (*.npy);;NIfTI (*.nii *.nii.gz);;MetaImage (*.mha *.mhd);;NRRD (*.nrrd)",
        )
        if not paths:
            return
        imported = 0
        errors = []
        for path in paths:
            try:
                lower = path.lower()
                if lower.endswith(".npy"):
                    arr = np.load(path)
                else:
                    img = sitk.ReadImage(path)
                    arr = sitk.GetArrayFromImage(img)  # z,y,x
                    if arr.ndim != 3:
                        raise ValueError(f"Only 3D images are supported, got ndim={arr.ndim}.")
                    arr = np.transpose(arr, (2, 1, 0))  # x,y,z to match toolbox arrays
                arr = np.asarray(arr, dtype=np.float32)
                name = f"Imported: {os.path.basename(path)}"
                self._imported_scenarios[name] = arr
                imported += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if imported > 0:
            self.refresh_sources()
            self._select_all_in_list(self.scenario_list)

        if errors:
            msg = "\n".join(errors[:8])
            if len(errors) > 8:
                msg += f"\n... and {len(errors) - 8} more"
            QMessageBox.warning(
                self,
                "Import completed with issues",
                f"Imported {imported}/{len(paths)} image(s).\n\nIssues:\n{msg}",
            )

    def export_current_table_csv(self):
        if self.results_table.columnCount() == 0 or self.results_table.rowCount() == 0:
            QMessageBox.information(self, "No data", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results to CSV", "analysis_results.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            headers = [self.results_table.horizontalHeaderItem(c).text() for c in range(self.results_table.columnCount())]
            lines = [",".join([h.replace(',', ' ') for h in headers])]
            for r in range(self.results_table.rowCount()):
                row_vals = []
                for c in range(self.results_table.columnCount()):
                    item = self.results_table.item(r, c)
                    row_vals.append((item.text() if item else "").replace(",", " "))
                lines.append(",".join(row_vals))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            QMessageBox.information(self, "Export complete", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not export CSV.\n\n{e}")

    _SESSION_FORMAT = "lesion_toolbox_analysis_session"
    _SESSION_VERSION = 1

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

    def _selected_scenario_names_ordered(self):
        names = []
        seen = set()
        for it in self.scenario_list.selectedItems():
            t = it.text()
            if t not in seen:
                seen.add(t)
                names.append(t)
        return names

    def _selected_lesion_names_ordered(self):
        all_lesions = getattr(self.parent_window, "lesion_objects", [])
        out = []
        seen = set()
        for item in self.lesion_list.selectedItems():
            idx = item.data(Qt.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(all_lesions):
                ln = str(getattr(all_lesions[idx], "name", f"Lesion_{idx+1}"))
                if ln not in seen:
                    seen.add(ln)
                    out.append(ln)
        return out

    def _serialize_lesions_for_session(self):
        out = []
        all_lesions = getattr(self.parent_window, "lesion_objects", []) or []
        for i, l in enumerate(all_lesions):
            center = tuple(int(v) for v in getattr(l, "center", (0, 0, 0)))
            entry = {
                "index": i,
                "id": str(getattr(l, "id", "")),
                "name": str(getattr(l, "name", f"Lesion_{i+1}")),
                "center": [int(center[0]), int(center[1]), int(center[2])],
                "radius": float(getattr(l, "radius", 0.0)),
                "intensity": float(getattr(l, "intensity", 0.0)),
                "size_scale": float(getattr(l, "size_scale", 1.0)),
                "uniform_intensity": bool(getattr(l, "uniform_intensity", False)),
                "use_recon_voxel_spacing": bool(
                    getattr(l, "use_recon_voxel_spacing", False)
                ),
            }
            src = getattr(l, "source_spacing_mm", None)
            if src is not None:
                entry["source_spacing_mm"] = [float(s) for s in src[:3]]
            cs = getattr(l, "custom_shape", None)
            if cs is not None:
                arr = np.asarray(cs, dtype=np.float32)
                entry["custom_shape"] = self._sanitize_for_json(arr.tolist())
                entry["custom_shape_is_source"] = bool(
                    getattr(l, "custom_shape_is_source", True)
                )
            else:
                entry["custom_shape"] = None
            out.append(entry)
        return out

    def _encode_array_for_json(self, arr):
        """Compact 3D float array payload for session JSON."""
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
        if not isinstance(payload, dict):
            raise ValueError("Invalid array payload.")
        if payload.get("encoding") != "base64+zlib":
            raise ValueError("Unsupported array encoding.")
        shape = tuple(int(x) for x in payload.get("shape", []))
        if len(shape) != 3:
            raise ValueError("Only 3D scenario arrays are supported.")
        comp = base64.b64decode(str(payload.get("data", "")).encode("ascii"))
        raw = zlib.decompress(comp)
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size != int(np.prod(shape)):
            raise ValueError("Scenario array size mismatch.")
        return arr.reshape(shape)

    def _serialize_scenarios_for_session(self):
        out = []
        for name, arr in (self._scenario_data or {}).items():
            meta = dict(self._scenario_meta.get(name, self._parse_scenario_meta(name)))
            source = "imported" if name in (self._imported_scenarios or {}) else "queue_result"
            out.append(
                {
                    "name": str(name),
                    "source": source,
                    "meta": self._sanitize_for_json(meta),
                    "array": self._encode_array_for_json(arr),
                }
            )
        return out

    def _restore_scenarios_from_session(self, scenario_defs):
        if not isinstance(scenario_defs, list):
            return 0, 0
        restored = 0
        skipped = 0
        self._imported_scenarios = {}
        self._scenario_meta_overrides = {}
        for d in scenario_defs:
            try:
                name = str(d.get("name", "")).strip()
                if not name:
                    raise ValueError("Missing scenario name.")
                arr = self._decode_array_from_json(d.get("array", {}))
                self._imported_scenarios[name] = np.asarray(arr, dtype=np.float32)
                meta = d.get("meta") or {}
                self._scenario_meta_overrides[name] = {
                    "modality": str(meta.get("modality", "Unknown")),
                    "algo": str(meta.get("algo", "Unknown")),
                    "iter_number": int(meta.get("iter_number", 0)),
                }
                restored += 1
            except Exception:
                skipped += 1
        return restored, skipped

    def _restore_lesions_from_session(self, lesion_defs):
        if not isinstance(lesion_defs, list):
            return 0, 0
        restored = []
        skipped = 0
        for i, d in enumerate(lesion_defs):
            try:
                center_raw = d.get("center", [0, 0, 0])
                center = tuple(int(v) for v in center_raw[:3])
                while len(center) < 3:
                    center = center + (0,)
                radius = float(d.get("radius", 0.0))
                intensity = float(d.get("intensity", 0.0))
                size_scale = float(d.get("size_scale", 1.0))
                cs_raw = d.get("custom_shape")
                custom_shape = None
                if cs_raw is not None:
                    custom_shape = np.asarray(cs_raw, dtype=np.float32)
                    if custom_shape.ndim != 3:
                        raise ValueError("custom_shape must be 3D")
                src = d.get("source_spacing_mm")
                lesion = Lesion(
                    center,
                    radius,
                    intensity,
                    custom_shape=custom_shape,
                    size_scale=size_scale,
                    uniform_intensity=bool(d.get("uniform_intensity", False)),
                    use_recon_voxel_spacing=bool(d.get("use_recon_voxel_spacing", False)),
                    source_spacing_mm=src,
                )
                lesion.name = str(d.get("name", f"Lesion_{i+1}"))
                lesion.id = str(d.get("id", "")) or f"session_{i+1}"
                if custom_shape is not None:
                    lesion.custom_shape_is_source = bool(d.get("custom_shape_is_source", True))
                else:
                    pw = self.parent_window
                    if hasattr(pw, "_normalize_sphere_lesion_size"):
                        pw._normalize_sphere_lesion_size(lesion)
                restored.append(lesion)
            except Exception:
                skipped += 1

        setattr(self.parent_window, "lesion_objects", restored)
        try:
            if hasattr(self.parent_window, "_refresh_lesion_combo"):
                self.parent_window._refresh_lesion_combo()
            if hasattr(self.parent_window, "_rebuild_lesion_mask_display"):
                self.parent_window._rebuild_lesion_mask_display()
            if hasattr(self.parent_window, "_update_ui_state"):
                self.parent_window._update_ui_state()
            if hasattr(self.parent_window, "_update_lesion_ui_state"):
                self.parent_window._update_lesion_ui_state()
        except Exception:
            pass
        return len(restored), skipped

    def save_analysis_session_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save analysis session",
            "analysis_session.json",
            "JSON session (*.json);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        x_key = self.x_axis_combo.currentData()
        y_key = self.y_axis_combo.currentData()
        dsi_features = self._selected_radiomics_features()
        payload = {
            "format": self._SESSION_FORMAT,
            "version": self._SESSION_VERSION,
            "readme": (
                "After moving to another study: use Refresh Sources, import NIfTI volumes (Import Image) "
                "so scenario names match those in selection.scenario_names, then Load Session. "
                "Lesions are matched by lesion_objects.name. Click Run Analysis to recompute metrics on new images."
            ),
            "settings": {
                "analysis_mode": self.analysis_mode_combo.currentText(),
                "x_axis_key": x_key,
                "y_axis_key": y_key,
                "radiomics_feature": dsi_features[0] if dsi_features else None,
                "dsi_feature_panel": dsi_features,
                "filter_start_mm": self.filter_start_input.text(),
                "filter_stop_mm": self.filter_stop_input.text(),
                "filter_step_mm": self.filter_step_input.text(),
                "show_legend": self.show_legend_checkbox.isChecked(),
                "noise_bias_scale_mode": self.noise_bias_scale_combo.currentData(),
                "noise_bias_x_min": self.noise_bias_xmin_edit.text(),
                "noise_bias_x_max": self.noise_bias_xmax_edit.text(),
                "noise_bias_y_min": self.noise_bias_ymin_edit.text(),
                "noise_bias_y_max": self.noise_bias_ymax_edit.text(),
                "noise_reference_mode": self.noise_reference_mode_combo.currentData(),
                "noise_roi_center": list(self._noise_roi_center) if self._noise_roi_center else None,
                "noise_roi_center_mm": (
                    list(self._noise_roi_center_mm) if self._noise_roi_center_mm else None
                ),
                "noise_roi_radius_vox": self._noise_roi_radius_vox,
                "noise_roi_radius_mm": self._noise_roi_radius_mm,
                "noise_roi_template_key": self._noise_roi_template_key,
            },
            "selection": {
                "scenario_names": self._selected_scenario_names_ordered(),
                "lesion_names": self._selected_lesion_names_ordered(),
            },
            "scenarios": self._serialize_scenarios_for_session(),
            "lesion_definitions": self._serialize_lesions_for_session(),
            "legend_aliases": dict(self._legend_aliases),
            "summary_text": self.analysis_summary_label.text(),
            "table_mode": self.analysis_mode_combo.currentText(),
            "records": [self._sanitize_for_json(r) for r in (self._current_records or [])],
            "figure_tab_titles": [self.figure_tabs.tabText(i) for i in range(self.figure_tabs.count())],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            n = len(self._current_records or [])
            QMessageBox.information(
                self,
                "Session saved",
                f"Saved analysis session to:\n{path}\n\n"
                f"Cached result rows: {n}. "
                "Lesion definitions and scenario images (with metadata) are included.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save session.\n\n{e}")

    def _restore_figure_tab_titles(self, titles):
        titles = [str(t).strip() for t in (titles or []) if str(t).strip()] or ["Figure 1"]
        while self.figure_tabs.count() > 1:
            self._close_figure_tab(self.figure_tabs.count() - 1)
        self.figure_tabs.setTabText(0, titles[0])
        for t in titles[1:]:
            self._add_figure_tab(t)

    def load_analysis_session_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load analysis session",
            "",
            "JSON session (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not read JSON.\n\n{e}")
            return
        if data.get("format") != self._SESSION_FORMAT:
            QMessageBox.warning(
                self,
                "Wrong file",
                "This JSON is not a toolbox analysis session (missing or unknown format field).",
            )
            return
        ver = int(data.get("version", 0))
        if ver != self._SESSION_VERSION:
            QMessageBox.warning(
                self,
                "Version mismatch",
                f"This file is version {ver}; this app expects {self._SESSION_VERSION}. Loading may be incomplete.",
            )

        scenario_defs = data.get("scenarios")
        lesion_defs = data.get("lesion_definitions")
        restored_scenarios = skipped_scenarios = 0
        restored_lesions = skipped_lesions = 0
        if scenario_defs is not None:
            restored_scenarios, skipped_scenarios = self._restore_scenarios_from_session(scenario_defs)
        if lesion_defs is not None:
            restored_lesions, skipped_lesions = self._restore_lesions_from_session(lesion_defs)

        self.refresh_sources()

        settings = data.get("settings") or {}
        mode = settings.get("analysis_mode") or "Recovery Coefficients"

        self.analysis_mode_combo.blockSignals(True)
        mi = self.analysis_mode_combo.findText(mode)
        if mi >= 0:
            self.analysis_mode_combo.setCurrentIndex(mi)
        self.analysis_mode_combo.blockSignals(False)

        self._update_mode_visibility()

        self.x_axis_combo.blockSignals(True)
        self.y_axis_combo.blockSignals(True)
        self.radiomics_feature_combo.blockSignals(True)

        xk = settings.get("x_axis_key")
        yk = settings.get("y_axis_key")
        if xk is not None:
            xi = self.x_axis_combo.findData(xk)
            if xi >= 0:
                self.x_axis_combo.setCurrentIndex(xi)
        if yk is not None:
            yi = self.y_axis_combo.findData(yk)
            if yi >= 0:
                self.y_axis_combo.setCurrentIndex(yi)

        panel = settings.get("dsi_feature_panel")
        if isinstance(panel, list) and panel:
            self._set_selected_radiomics_features([str(k) for k in panel])
        else:
            rf = settings.get("radiomics_feature")
            if rf:
                # Legacy single-feature sessions: treat the saved feature as the only one in the DSI panel.
                self._set_selected_radiomics_features([str(rf)])

        if "filter_start_mm" in settings:
            self.filter_start_input.setText(str(settings["filter_start_mm"]))
        if "filter_stop_mm" in settings:
            self.filter_stop_input.setText(str(settings["filter_stop_mm"]))
        if "filter_step_mm" in settings:
            self.filter_step_input.setText(str(settings["filter_step_mm"]))
        self.show_legend_checkbox.setChecked(bool(settings.get("show_legend", True)))

        nbm = settings.get("noise_bias_scale_mode")
        if nbm is not None:
            ni = self.noise_bias_scale_combo.findData(nbm)
            if ni >= 0:
                self.noise_bias_scale_combo.setCurrentIndex(ni)
        if "noise_bias_x_min" in settings:
            self.noise_bias_xmin_edit.setText(str(settings["noise_bias_x_min"]))
        if "noise_bias_x_max" in settings:
            self.noise_bias_xmax_edit.setText(str(settings["noise_bias_x_max"]))
        if "noise_bias_y_min" in settings:
            self.noise_bias_ymin_edit.setText(str(settings["noise_bias_y_min"]))
        if "noise_bias_y_max" in settings:
            self.noise_bias_ymax_edit.setText(str(settings["noise_bias_y_max"]))
        self._sync_noise_bias_manual_visibility()
        nrm = settings.get("noise_reference_mode")
        if nrm is not None:
            ri = self.noise_reference_mode_combo.findData(nrm)
            if ri >= 0:
                self.noise_reference_mode_combo.setCurrentIndex(ri)
        roi_center_mm = settings.get("noise_roi_center_mm")
        if isinstance(roi_center_mm, (list, tuple)) and len(roi_center_mm) == 3:
            self._noise_roi_center_mm = tuple(float(v) for v in roi_center_mm)
        roi_center = settings.get("noise_roi_center")
        if isinstance(roi_center, (list, tuple)) and len(roi_center) == 3:
            self._noise_roi_center = tuple(float(v) for v in roi_center)
        if "noise_roi_radius_vox" in settings:
            self._noise_roi_radius_vox = float(settings["noise_roi_radius_vox"])
        if "noise_roi_radius_mm" in settings:
            self._noise_roi_radius_mm = float(settings["noise_roi_radius_mm"])
        if settings.get("noise_roi_template_key"):
            self._noise_roi_template_key = str(settings["noise_roi_template_key"])
        self._sync_noise_roi_voxel_cache()
        self._sync_noise_reference_lesion_visibility()
        self._update_noise_roi_status_label()

        self.x_axis_combo.blockSignals(False)
        self.y_axis_combo.blockSignals(False)
        self.radiomics_feature_combo.blockSignals(False)

        sel = data.get("selection") or {}
        sc_names = list(sel.get("scenario_names") or [])
        le_names = list(sel.get("lesion_names") or [])

        self.scenario_list.clearSelection()
        have_sc = {self.scenario_list.item(i).text() for i in range(self.scenario_list.count())}
        for i in range(self.scenario_list.count()):
            it = self.scenario_list.item(i)
            if it.text() in sc_names:
                it.setSelected(True)
        miss_sc = [n for n in sc_names if n not in have_sc]

        self.lesion_list.clearSelection()
        all_lesions = getattr(self.parent_window, "lesion_objects", [])
        cur_lesion_names = {str(getattr(L, "name", "")) for L in all_lesions}
        miss_le = [n for n in le_names if n not in cur_lesion_names]
        for i in range(self.lesion_list.count()):
            it = self.lesion_list.item(i)
            idx = it.data(Qt.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(all_lesions):
                nm = str(getattr(all_lesions[idx], "name", f"Lesion_{idx+1}"))
                if nm in le_names:
                    it.setSelected(True)

        self._legend_aliases = dict(data.get("legend_aliases") or {})

        raw_recs = data.get("records") or []
        self._current_records = []
        for r in raw_recs:
            if isinstance(r, dict):
                self._current_records.append(dict(r))
        self._records_mode = mode
        self._records_signature = self._current_analysis_signature()

        mode_table = data.get("table_mode") or mode
        self._populate_table(mode_table, self._current_records)

        st = data.get("summary_text")
        if isinstance(st, str) and st.strip():
            self.analysis_summary_label.setText(st)
        elif not self._current_records:
            self.analysis_summary_label.setText("Run summary will appear here.")

        self._restore_figure_tab_titles(data.get("figure_tab_titles"))
        self.plot_current_results()

        parts = [
            f"Loaded session from:\n{path}",
            f"Restored {len(self._current_records)} cached table row(s).",
        ]
        if scenario_defs is not None:
            parts.append(
                f"Restored scenario images: {restored_scenarios}."
                + (f" Skipped: {skipped_scenarios}." if skipped_scenarios else "")
            )
        if lesion_defs is not None:
            parts.append(f"Restored lesion definitions: {restored_lesions}." + (f" Skipped: {skipped_lesions}." if skipped_lesions else ""))
        if miss_sc:
            parts.append(f"Scenarios not in current list ({len(miss_sc)}): {', '.join(miss_sc[:5])}" + ("…" if len(miss_sc) > 5 else ""))
        if miss_le:
            parts.append(f"Lesion names not found ({len(miss_le)}): {', '.join(miss_le[:5])}" + ("…" if len(miss_le) > 5 else ""))
        if miss_sc or miss_le:
            parts.append("Adjust selections and click Run Analysis to compute on the new study.")
        else:
            parts.append("If volumes changed, click Run Analysis to recompute metrics.")
        QMessageBox.information(self, "Session loaded", "\n\n".join(parts))

    def _safe_filename(self, name):
        keep = []
        for ch in name:
            keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
        sanitized = "".join(keep).strip("._")
        return sanitized or "scenario"

    def export_selected_scenarios_nifti(self):
        selected = self._selected_scenarios()
        if not selected:
            QMessageBox.warning(self, "No Scenario", "Select at least one scenario to export.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for NIfTI export")
        if not out_dir:
            return

        spacing = self._voxel_spacing()  # x, y, z in mm
        exported = 0
        errors = []
        for scenario_name, arr_xyz in selected:
            try:
                arr_xyz = np.asarray(arr_xyz, dtype=np.float32)
                if arr_xyz.ndim != 3:
                    raise ValueError(f"Expected 3D array, got ndim={arr_xyz.ndim}")
                arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
                img = sitk.GetImageFromArray(arr_zyx)
                img.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
                out_name = self._safe_filename(scenario_name) + ".nii.gz"
                out_path = os.path.join(out_dir, out_name)
                sitk.WriteImage(img, out_path)
                exported += 1
            except Exception as e:
                errors.append(f"{scenario_name}: {e}")

        if errors:
            msg = "\n".join(errors[:8])
            if len(errors) > 8:
                msg += f"\n... and {len(errors) - 8} more"
            QMessageBox.warning(
                self,
                "Export completed with issues",
                f"Exported {exported}/{len(selected)} scenarios.\n\nIssues:\n{msg}",
            )
        else:
            QMessageBox.information(
                self,
                "Export complete",
                f"Exported {exported} scenario(s) to:\n{out_dir}",
            )

    def refresh_sources(self):
        self._scenario_data = self._collect_scenarios()
        self.scenario_list.clear()
        for name in self._scenario_data.keys():
            item = QListWidgetItem(name)
            item.setSelected(True)
            self.scenario_list.addItem(item)

        self.lesion_list.clear()
        for i, lesion in enumerate(getattr(self.parent_window, "lesion_objects", [])):
            lesion_name = getattr(lesion, "name", f"Lesion_{i+1}")
            item = QListWidgetItem(f"{i+1}: {lesion_name}")
            item.setData(Qt.UserRole, i)
            item.setSelected(True)
            self.lesion_list.addItem(item)
        self._apply_search_filter(self.scenario_list, self.scenario_search.text() if hasattr(self, "scenario_search") else "")
        self._apply_search_filter(self.lesion_list, self.lesion_search.text() if hasattr(self, "lesion_search") else "")

    def _noise_roi_spacing_mm(self):
        spacing = self._voxel_spacing()
        if spacing is None:
            return (1.0, 1.0, 1.0)
        parts = [float(s) for s in spacing[:3]]
        while len(parts) < 3:
            parts.append(parts[-1] if parts else 1.0)
        return tuple(max(p, 1e-6) for p in parts)

    @staticmethod
    def _vox_center_to_mm(center_vox, spacing_mm):
        cx, cy, cz = (float(center_vox[0]), float(center_vox[1]), float(center_vox[2]))
        sx, sy, sz = spacing_mm
        return (cx * sx, cy * sy, cz * sz)

    @staticmethod
    def _mm_center_to_vox(center_mm, spacing_mm):
        mx, my, mz = (float(center_mm[0]), float(center_mm[1]), float(center_mm[2]))
        sx, sy, sz = spacing_mm
        return (mx / sx, my / sy, mz / sz)

    def _sync_noise_roi_voxel_cache(self):
        """Derive voxel center/radius from mm (canonical) for display and legacy paths."""
        if self._noise_roi_center_mm is None:
            if self._noise_roi_center is not None:
                self._noise_roi_center_mm = self._vox_center_to_mm(
                    self._noise_roi_center, self._noise_roi_spacing_mm()
                )
            return
        spacing = self._noise_roi_spacing_mm()
        self._noise_roi_center = self._mm_center_to_vox(self._noise_roi_center_mm, spacing)
        bin_mm = float(min(spacing))
        if self._noise_roi_radius_mm > 0:
            self._noise_roi_radius_vox = max(float(self._noise_roi_radius_mm) / bin_mm, 0.5)

    def _noise_roi_template_choices(self):
        """
        Single placement preview: Tab 2 initial reconstruction with lesion mask overlay.
        ROI center/radius are stored in mm on the shared grid; this volume is never used
        for noise CV% — only for visually avoiding lesions when placing the sphere.
        """
        pw = self.parent_window
        img = getattr(pw, "image_array", None)
        lesion = getattr(pw, "lesion_array", None)
        if img is None or np.asarray(img).size == 0:
            return []
        img = np.asarray(img, dtype=np.float32)
        if lesion is None or np.asarray(lesion).size == 0 or not np.any(lesion > 0):
            return []
        lesion = np.asarray(lesion, dtype=np.float32)
        if lesion.shape != img.shape:
            return []
        combined = img.copy()
        hot = lesion > 0
        combined[hot] = np.maximum(combined[hot], lesion[hot])
        return [("initial_plus_lesions", "Initial recon + lesion mask", combined)]

    def _noise_roi_placement_template(self):
        """Default placement guide when opening the ROI dialog."""
        choices = self._noise_roi_template_choices()
        if not choices:
            return None, ""
        if self._noise_roi_template_key:
            for key, label, vol in choices:
                if key == self._noise_roi_template_key:
                    return vol, label
        return choices[0][2], choices[0][1]

    def _open_noise_roi_dialog(self):
        choices = self._noise_roi_template_choices()
        if not choices:
            QMessageBox.warning(
                self,
                "No placement preview",
                "Run Tab 2 initial reconstruction and insert lesions first.\n\n"
                "The ROI picker shows initial recon + lesion mask (placement only). "
                "Noise CV% is still computed from each selected reconstruction "
                "scenario during analysis.",
            )
            return
        self._sync_noise_roi_voxel_cache()
        dlg = NoiseRoiPickerDialog(
            template_choices=choices,
            initial_template_key="initial_plus_lesions",
            voxel_spacing=self._noise_roi_spacing_mm(),
            initial_center_mm=self._noise_roi_center_mm,
            initial_radius_mm=self._noise_roi_radius_mm,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._noise_roi_center_mm = tuple(float(v) for v in dlg.center_mm)
        self._noise_roi_radius_mm = float(dlg.radius_mm)
        self._noise_roi_template_key = "initial_plus_lesions"
        self._sync_noise_roi_voxel_cache()
        self._update_noise_roi_status_label()
        self._analysis_inputs_changed()

    def _update_noise_roi_status_label(self):
        pass

    def _noise_roi_sphere_mask(self, shape, spacing_mm=None):
        if self._noise_roi_center_mm is None or self._noise_roi_radius_mm <= 0:
            return None
        spacing = spacing_mm if spacing_mm is not None else self._noise_roi_spacing_mm()
        sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        mx, my, mz = self._noise_roi_center_mm
        r_mm = float(self._noise_roi_radius_mm)
        xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
        dist2_mm = ((xx * sx - mx) ** 2 + (yy * sy - my) ** 2 + (zz * sz - mz) ** 2)
        return dist2_mm <= r_mm ** 2

    def _collect_scenarios(self):
        scenarios = {}
        self._scenario_meta = {}

        for item in getattr(self.parent_window, "queue_results", []):
            if getattr(item, "status", "") == "Completed" and getattr(item, "result", None) is not None:
                scenarios[item.name] = np.asarray(item.result, dtype=np.float32)
                self._scenario_meta[item.name] = self._scenario_meta_overrides.get(
                    item.name, self._meta_from_queue_result(item)
                )
        for name, arr in self._imported_scenarios.items():
            scenarios[name] = np.asarray(arr, dtype=np.float32)
            self._scenario_meta[name] = self._scenario_meta_overrides.get(
                name, self._parse_scenario_meta(name)
            )
        return scenarios

    def _meta_from_queue_result(self, item):
        """Prefer recon metadata stored on the saved queue result."""
        name = str(getattr(item, "name", ""))
        modality = getattr(item, "modality", None)
        algo = getattr(item, "algo", None)
        iter_number = getattr(item, "iter_number", None)

        if not modality or not algo or iter_number is None:
            parsed = self._parse_scenario_meta(name)
            if not modality:
                modality = parsed.get("modality", "Unknown")
            if not algo:
                algo = parsed.get("algo", "Unknown")
            if iter_number is None:
                iter_number = iter_number_for_recon_result(
                    name,
                    n_iters=getattr(item, "n_iters", None),
                    save_all_iterations=bool(getattr(item, "save_all_iterations", False)),
                )

        return {
            "modality": str(modality or "Unknown"),
            "algo": str(algo or "Unknown"),
            "iter_number": int(iter_number),
        }

    def _parse_scenario_meta(self, scenario_name):
        low = scenario_name.lower()
        modality = "SPECT" if "spect" in low else ("PET" if "pet" in low else "Unknown")
        algo = "Unknown"
        for a in ["OSEM", "BSREM", "OSMAPOSL", "KEM"]:
            if a.lower() in low:
                algo = a
                break
        parsed_iter = parse_iteration_from_scenario_name(scenario_name)
        iter_number = parsed_iter if parsed_iter is not None else 0
        return {"modality": modality, "algo": algo, "iter_number": iter_number}

    def _selected_scenarios(self):
        names = [i.text() for i in self.scenario_list.selectedItems()]
        unique_names = []
        seen = set()
        for n in names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)
        return [(name, self._scenario_data[name]) for name in unique_names if name in self._scenario_data]

    def _selected_lesions(self):
        lesions = []
        all_lesions = getattr(self.parent_window, "lesion_objects", [])
        for item in self.lesion_list.selectedItems():
            idx = item.data(Qt.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(all_lesions):
                lesions.append((idx, all_lesions[idx]))
        return lesions

    def _current_analysis_signature(self):
        """Signature of inputs that define record content (not axis visualization choices)."""
        mode = self.analysis_mode_combo.currentText()
        sc_names = tuple(sorted([n for n, _ in self._selected_scenarios()]))
        ls_names = tuple(sorted([str(getattr(l, "name", f"Lesion_{i+1}")) for i, l in self._selected_lesions()]))
        if mode == "Post-Recon Filtering Effect":
            return (
                mode,
                sc_names,
                ls_names,
                str(self.filter_start_input.text()).strip(),
                str(self.filter_stop_input.text()).strip(),
                str(self.filter_step_input.text()).strip(),
            )
        if mode == "Noise-Bias Curve":
            return (
                mode,
                sc_names,
                ls_names,
                self.noise_reference_mode_combo.currentData() if hasattr(self, "noise_reference_mode_combo") else "per_lesion_ring",
                self._noise_roi_center,
                self._noise_roi_center_mm,
                float(self._noise_roi_radius_mm),
            )
        return (mode, sc_names, ls_names)

    def _voxel_spacing(self):
        """Use recon object-grid spacing from tab 2 (post-recon or Object Voxel Size field)."""
        fn = getattr(self.parent_window, "get_effective_voxel_spacing_mm", None)
        if callable(fn):
            spacing = fn()
            if spacing is not None:
                spacing = tuple(float(x) for x in spacing)
                if len(spacing) >= 3:
                    return spacing[:3]
                if len(spacing) == 2:
                    return (spacing[0], spacing[1], spacing[1])
                if len(spacing) == 1:
                    return (spacing[0], spacing[0], spacing[0])
        return (1.0, 1.0, 1.0)

    def _spacing_for_lesion(self, lesion):
        """Per-lesion spacing for volume and mm-based edge metrics."""
        return lesion_volume_spacing_mm(lesion, self._voxel_spacing())

    def _lesion_mask(self, lesion, shape):
        return build_lesion_mask_on_grid(shape, lesion)

    def _lesion_spill_exclusion_margin_mm(self, lesions=None):
        """
        Clearance from lesion truth mask when carving the global noise ROI.
        Library (non-sphere) lesions need extra margin because recon spill extends
        beyond the segmentation mask — a voxel shell is not enough.
        """
        fwhm = float(self._recon_psf_fwhm_mm())
        has_library = False
        if lesions:
            has_library = any(
                getattr(lesion, "custom_shape", None) is not None for _idx, lesion in lesions
            )
        margin = max(8.0, 1.5 * fwhm)
        if has_library:
            margin = max(margin, 10.0, 2.0 * fwhm)
        return float(margin)

    def _all_lesion_exclusion_mask(self, shape, lesions=None):
        """Union of lesion masks plus a physical spill margin (mm) excluded from global noise ROI."""
        if lesions is None:
            lesions = self._selected_lesions()
        if not lesions:
            all_objs = getattr(self.parent_window, "lesion_objects", [])
            lesions = [(i, l) for i, l in enumerate(all_objs)]
        union = np.zeros(shape, dtype=bool)
        for _idx, lesion in lesions:
            union |= self._lesion_mask(lesion, shape)
        if not np.any(union):
            return union
        spacing = np.array(self._noise_roi_spacing_mm(), dtype=np.float64)
        spacing = np.where(spacing > 0, spacing, 1.0)
        margin_mm = self._lesion_spill_exclusion_margin_mm(lesions)
        try:
            dist_outside_mm = scipy.ndimage.distance_transform_edt(~union, sampling=spacing)
        except TypeError:
            dist_outside_mm = scipy.ndimage.distance_transform_edt(
                ~np.asarray(union, dtype=np.uint8), sampling=spacing
            )
        return dist_outside_mm < margin_mm

    def _noise_roi_min_lesion_distance_mm(self, shape, lesions=None):
        """Shortest distance (mm) from liver ROI center to any lesion truth voxel."""
        if self._noise_roi_center_mm is None:
            return float("nan")
        if lesions is None:
            lesions = self._selected_lesions()
        if not lesions:
            all_objs = getattr(self.parent_window, "lesion_objects", [])
            lesions = [(i, l) for i, l in enumerate(all_objs)]
        union = np.zeros(shape, dtype=bool)
        for _idx, lesion in lesions:
            union |= self._lesion_mask(lesion, shape)
        if not np.any(union):
            return float("inf")
        spacing = np.array(self._noise_roi_spacing_mm(), dtype=np.float64)
        spacing = np.where(spacing > 0, spacing, 1.0)
        cx, cy, cz = self._mm_center_to_vox(self._noise_roi_center_mm, tuple(spacing))
        ix = int(np.clip(round(cx), 0, shape[0] - 1))
        iy = int(np.clip(round(cy), 0, shape[1] - 1))
        iz = int(np.clip(round(cz), 0, shape[2] - 1))
        try:
            dist_outside_mm = scipy.ndimage.distance_transform_edt(~union, sampling=spacing)
        except TypeError:
            dist_outside_mm = scipy.ndimage.distance_transform_edt(
                ~np.asarray(union, dtype=np.uint8), sampling=spacing
            )
        return float(dist_outside_mm[ix, iy, iz])

    def _noise_cv_from_background_voxels(self, image, bg_mask):
        """
        COV% = 100 × std / mean in background voxels.
        Drops air/non-tissue voxels so a liver ROI clipped by the body contour
        does not inflate CV with zeros.
        """
        if bg_mask is None or not np.any(bg_mask):
            return float("nan"), 0.0, 0.0, 0
        vox = np.asarray(image[bg_mask], dtype=np.float64)
        vox = vox[np.isfinite(vox)]
        if vox.size == 0:
            return float("nan"), 0.0, 0.0, 0
        pos = vox[vox > 0]
        if pos.size >= 20:
            tissue_floor = max(0.05 * float(np.median(pos)), 1e-9)
            vox = vox[vox >= tissue_floor]
        if vox.size < 5:
            return float("nan"), 0.0, 0.0, int(vox.size)
        bg_mean = float(np.mean(vox))
        bg_std = float(np.std(vox))
        noise_cv = 100.0 * bg_std / (abs(bg_mean) + 1e-9)
        return noise_cv, bg_mean, bg_std, int(vox.size)

    def _build_scenario_noise_cache(self, scenarios, lesions, mode):
        """
        Test 3: noise CV% is one value per reconstruction scenario (shared by all lesions).
        Computed in the global liver ROI on that scenario's own image.
        """
        cache = {}
        if mode != "Noise-Bias Curve" or not self._use_global_liver_noise_roi():
            return cache
        if self._noise_roi_center_mm is None:
            return cache
        min_lesion_dist = None
        for scenario_name, image in scenarios:
            image = np.asarray(image)
            if min_lesion_dist is None:
                min_lesion_dist = self._noise_roi_min_lesion_distance_mm(image.shape, lesions)
            bg_mask = self._global_liver_noise_mask(image.shape, lesions)
            noise_cv, bg_mean, bg_std, n_vox = self._noise_cv_from_background_voxels(image, bg_mask)
            cache[scenario_name] = {
                "noise_cv_percent": noise_cv,
                "noise_bg_mean": bg_mean,
                "noise_bg_std": bg_std,
                "noise_roi_voxel_count": n_vox,
                "noise_roi_min_lesion_distance_mm": min_lesion_dist,
                "noise_spill_exclusion_mm": self._lesion_spill_exclusion_margin_mm(lesions),
            }
        return cache

    def _background_ring_mask(self, lesion, shape):
        lesion_mask = self._lesion_mask(lesion, shape)
        dilated = scipy.ndimage.binary_dilation(lesion_mask, iterations=3)
        return np.logical_and(dilated, np.logical_not(lesion_mask))

    def _use_global_liver_noise_roi(self):
        if self.analysis_mode_combo.currentText() != "Noise-Bias Curve":
            return False
        return (
            hasattr(self, "noise_reference_mode_combo")
            and self.noise_reference_mode_combo.currentData() == "global_liver"
        )

    def _global_liver_noise_mask(self, shape, lesions=None):
        """Filled sphere in physical mm, excluding all lesions and a spill margin."""
        sphere = self._noise_roi_sphere_mask(shape, self._noise_roi_spacing_mm())
        if sphere is None or not np.any(sphere):
            return sphere
        exclude = self._all_lesion_exclusion_mask(shape, lesions)
        if np.any(exclude):
            sphere = np.logical_and(sphere, np.logical_not(exclude))
        return sphere

    def _noise_background_mask(self, lesion, shape, lesions=None):
        # Global liver mode: same ROI coordinates on every volume; voxel values come from
        # whichever reconstruction image is passed into _compute_record for that row.
        if self._use_global_liver_noise_roi():
            global_mask = self._global_liver_noise_mask(shape, lesions)
            if global_mask is not None and np.any(global_mask):
                return global_mask
        return self._background_ring_mask(lesion, shape)

    def _lesion_volume_ml(self, lesion, shape):
        return lesion_volume_ml_on_grid(shape, lesion, self._voxel_spacing())

    def _synthetic_phantom_lesion_volume(self, lesion, shape):
        """
        Same inserted-lesion model as the main window phantom (pre-reconstruction truth),
        built at the same grid as the analysis volume.
        """
        return build_lesion_truth_on_grid(shape, lesion)

    def _reference_lesion_voxels(self, lesion, image, lesion_mask):
        """Reference intensities from the Tab 3 lesion definition (mask + intensity model)."""
        truth = self._synthetic_phantom_lesion_volume(lesion, image.shape)
        return reference_lesion_voxels_on_grid(truth, lesion_mask)

    def _expected_lesion_mean(self, lesion, shape, lesion_mask):
        """Mean inserted truth from the Tab 3 lesion definition."""
        truth = self._synthetic_phantom_lesion_volume(lesion, shape)
        return expected_lesion_mean_on_grid(truth, lesion_mask, lesion)

    def _radiomics(self, voxels):
        if voxels.size == 0:
            return {}
        vals = voxels.astype(np.float64)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = float(100.0 * std / (abs(mean) + 1e-9))
        hist, _ = np.histogram(vals, bins=64)
        p = hist / (np.sum(hist) + 1e-12)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))
        energy = float(np.sum((vals - np.min(vals)) ** 2))
        centered = vals - mean
        m2 = np.mean(centered ** 2) + 1e-12
        skew = float(np.mean(centered ** 3) / (m2 ** 1.5))
        kurt = float(np.mean(centered ** 4) / (m2 ** 2) - 3.0)
        return {
            "peak": float(np.max(vals)),
            "mean": mean,
            "std": std,
            "cv_percent": cv,
            "entropy": entropy,
            "energy": energy,
            "p10": float(np.percentile(vals, 10)),
            "median": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "skewness": skew,
            "kurtosis": kurt,
        }

    def _gibbs_truth_level(self, lesion, lesion_mask, image_shape, reference_voxels):
        """
        Inserted-truth activity L for Tong overshoot (SUVmax / L − 1).

        Spheres and uniform library lesions: mean truth (flat activity).
        Heterogeneous library templates: template peak under the mask.
        """
        is_library = getattr(lesion, "custom_shape", None) is not None
        uniform = not is_library or bool(getattr(lesion, "uniform_intensity", False))
        if reference_voxels is not None and reference_voxels.size > 0:
            ref_vals = np.asarray(reference_voxels, dtype=np.float64)
            if uniform:
                return float(np.mean(ref_vals))
            return float(np.max(ref_vals))
        if lesion_mask is not None and np.any(lesion_mask):
            return float(self._expected_lesion_mean(lesion, image_shape, lesion_mask))
        return float(getattr(lesion, "intensity", 0.0))

    def _gibbs_edge_window_mm(self, fwhm_mm, bin_mm):
        """Near-edge half-width w ≈ one system-resolution FWHM (Rogasch/Hofheinz; Tsutsui)."""
        try:
            fwhm_val = float(fwhm_mm) if fwhm_mm is not None else 0.0
        except (TypeError, ValueError):
            fwhm_val = 0.0
        default_w = 4.0
        return max(fwhm_val if fwhm_val > 0 else default_w, 2.0 * float(bin_mm))

    def _recon_psf_fwhm_mm(self):
        """Reconstruction PSF FWHM (mm) from the main window's PSF input."""
        try:
            val = float(self.parent_window.psf_input.text())
            return val if val > 0 else 4.5
        except Exception:
            return 4.5

    def _ground_truth_volume(self, lesion, shape):
        """Inserted-truth activity map from the Tab 3 lesion definition."""
        return self._synthetic_phantom_lesion_volume(lesion, shape).astype(np.float64)

    def _lesion_signed_distance(self, lesion_mask, spacing):
        """
        Signed distance (mm) from the lesion boundary: negative inside, positive
        outside. Computed via Euclidean distance transform with anisotropic
        voxel spacing.
        """
        mask_bool = np.asarray(lesion_mask, dtype=bool)
        try:
            d_out = scipy.ndimage.distance_transform_edt(~mask_bool, sampling=spacing)
            d_in = scipy.ndimage.distance_transform_edt(mask_bool, sampling=spacing)
        except TypeError:
            mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
            d_out = scipy.ndimage.distance_transform_edt(~mask_u8, sampling=spacing)
            d_in = scipy.ndimage.distance_transform_edt(mask_u8, sampling=spacing)
        return d_out.astype(np.float64) - d_in.astype(np.float64)

    def _gibbs_shell_voxels(self, image, signed_d, psf_fwhm_mm, voxel_mm):
        """
        Voxel-level statistics in three boundary-aligned shells, defined by the
        signed distance d (mm) to the lesion boundary (d<0 inside, d>0 outside).

        Shells follow the physical Gibbs pattern: bright overshoot just INSIDE
        the boundary, dim undershoot just OUTSIDE, true background well past
        both. Shell thickness adapts to voxel spacing so each shell holds at
        least one voxel layer (the Euclidean distance transform reports
        center-to-center distances, so the smallest non-zero |d| is ≈ one
        voxel spacing):

          inside-rim   d ∈ [−rim, 0]                      — bright-rim overshoot
                                                              (Tong 2011 §III-A;
                                                              Snyder 1987).
          outside-rim  d ∈ (0, +rim]                       — outside undershoot.
          far-bg       d ∈ (max(rim, FWHM), max(2·rim, 2·FWHM)] — local background
                                                                  past the ringing.

        rim = max(0.5·FWHM, voxel) ensures each rim shell contains the boundary
        voxel layer regardless of voxel/FWHM ratio.

        Returns inside-rim max A+, outside-rim min A−, far-bg mean b. NaN if a
        shell is empty on this grid.
        """
        img = np.asarray(image, dtype=np.float64)
        rim_r = max(0.5 * float(psf_fwhm_mm), float(voxel_mm))
        far_lo = max(rim_r, float(psf_fwhm_mm))
        far_hi = max(2.0 * rim_r, 2.0 * float(psf_fwhm_mm))

        inside_rim = (signed_d <= 0.0) & (signed_d >= -rim_r)
        outside_rim = (signed_d > 0.0) & (signed_d <= rim_r)
        far_bg = (signed_d > far_lo) & (signed_d <= far_hi)

        a_plus = float(np.max(img[inside_rim])) if np.any(inside_rim) else float("nan")
        a_minus = float(np.min(img[outside_rim])) if np.any(outside_rim) else float("nan")
        b_bg = float(np.mean(img[far_bg])) if np.any(far_bg) else float("nan")
        return a_plus, a_minus, b_bg

    def _gibbs_metrics(
        self,
        image,
        lesion_mask,
        lesion_voxels,
        background_voxels,
        lesion,
        reference_voxels,
    ):
        """
        Test 2 (Gibbs Effect) — three literature-validated edge-artifact metrics:

          Overshoot %  = (SUVmax / L − 1) × 100
                         Peak inflation above inserted-truth activity L.
                         Tong A.M. et al., IEEE TNS 58(5):2264 (2011);
                         Bai B., Esser P., IEEE NSS/MIC (2010);
                         Rahmim A., Qi J., Sossi V., Med. Phys. 40(6):064301 (2013).

          Undershoot % = max(0, (b − A−) / b) × 100
                         Activity dip below local background just outside the
                         lesion boundary — the outside half of the Gibbs ring.
                         A− = min voxel in the outside-rim shell d ∈ (0, rim];
                         b  = mean voxel in the far-bg shell.
                         Tong et al. 2011 IEEE TNS §III-A; Snyder D.L. et al.,
                         IEEE TMI 6(3):228 (1987).

          GA %         = (A+ − A−) / (A+ + A−) × 100
                         Edge-shell amplitude index from voxel extrema (not
                         radial means) so the bright-rim signature is preserved
                         on real PET data. A+ = max voxel in the inside-rim
                         shell d ∈ [−rim, 0]; A− as above.
                         Rogasch J.M.M., Hofheinz F. et al., EJNMMI Phys. 1:12
                         (2014).

          Edge Window w (mm) — informational; ≈ system-resolution FWHM
                               (Tong 2011 IEEE TNS).

        For uniform spheres and uniform library lesions L = flat inserted activity
        (mean truth). For heterogeneous library lesions L is the template peak.
        d is the signed Euclidean distance from the lesion boundary (negative
        inside, positive outside), computed with reconstruction-grid spacing.

        Undershoot and GA require the rasterized lesion to be at least one
        system-resolution diameter across (2·r_eq ≥ 2·PSF_FWHM, with a floor of
        2 voxels per rim shell). The Overshoot does not depend on shells and is
        always reported. Sub-resolution lesions return NaN for Undershoot and
        GA in line with Hofheinz/Rogasch 2014 and Tong 2011, which restrict
        ringing quantification to resolved lesions.
        """
        nan_result = {
            "overshoot_percent": np.nan,
            "undershoot_percent": np.nan,
            "ga_percent": np.nan,
            "edge_window_mm": np.nan,
        }
        if lesion_voxels.size == 0 or lesion_mask is None or not np.any(lesion_mask):
            return nan_result

        lesion_vals = np.asarray(lesion_voxels, dtype=np.float64)
        lesion_peak = float(np.max(lesion_vals))
        is_library = getattr(lesion, "custom_shape", None) is not None

        image_shape = image.shape if image is not None else lesion_mask.shape
        L_truth = self._gibbs_truth_level(lesion, lesion_mask, image_shape, reference_voxels)
        L_safe = abs(L_truth) + 1e-9

        result = dict(nan_result)

        spacing = np.array(self._voxel_spacing(), dtype=np.float64)
        spacing = np.where(spacing > 0, spacing, 1.0)
        bin_mm = float(np.min(spacing))
        # Shell geometry always follows the reconstruction PSF (Tab 2), never the
        # post-recon Gaussian filter FWHM used in filter-sweep records.
        psf_fwhm = self._recon_psf_fwhm_mm()
        w_mm = self._gibbs_edge_window_mm(psf_fwhm, bin_mm)
        result["edge_window_mm"] = w_mm

        if image is None:
            result["overshoot_percent"] = 100.0 * (lesion_peak / L_safe - 1.0)
            return result

        mask_bool = np.asarray(lesion_mask, dtype=bool)
        voxel_count = int(np.count_nonzero(mask_bool))
        lesion_vol_mm3 = max(voxel_count * float(np.prod(spacing)), 1e-9)
        r_eq_mm = (3.0 * lesion_vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        min_diameter_mm = max(2.0 * psf_fwhm, 2.0 * bin_mm)
        if 2.0 * r_eq_mm < min_diameter_mm:
            result["overshoot_percent"] = 100.0 * (lesion_peak / L_safe - 1.0)
            return result

        signed_d = self._lesion_signed_distance(mask_bool, spacing)
        a_plus, a_minus, b_bg = self._gibbs_shell_voxels(image, signed_d, psf_fwhm, bin_mm)

        peak_for_overshoot = lesion_peak
        if is_library and np.isfinite(a_plus):
            # Rim-focused peak for irregular library morphology (Gibbs is a boundary effect).
            peak_for_overshoot = float(a_plus)
        result["overshoot_percent"] = 100.0 * (peak_for_overshoot / L_safe - 1.0)

        if np.isfinite(a_minus) and np.isfinite(b_bg) and abs(b_bg) > 1e-9:
            result["undershoot_percent"] = 100.0 * max(0.0, (b_bg - a_minus) / b_bg)

        if np.isfinite(a_plus) and np.isfinite(a_minus) and (a_plus + a_minus) > 1e-9:
            result["ga_percent"] = 100.0 * (a_plus - a_minus) / (a_plus + a_minus)

        return result

    def _apply_gaussian_fwhm(self, image, fwhm_mm):
        if fwhm_mm <= 0:
            return image
        spacing = np.array(self._voxel_spacing(), dtype=np.float64)
        sigma_mm = float(fwhm_mm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_vox = tuple(float(sigma_mm / max(s, 1e-6)) for s in spacing)
        out = scipy.ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma_vox, mode="nearest")
        return out.astype(np.float32)

    def _compute_record(
        self, scenario_name, image, lesion_idx, lesion, fwhm_mm=0.0, scenario_noise_cache=None, lesions=None
    ):
        lesion_mask = self._lesion_mask(lesion, image.shape)
        bg_mask = self._noise_background_mask(lesion, image.shape, lesions=lesions)
        lesion_vox = image[lesion_mask]
        reference_voxels = self._reference_lesion_voxels(lesion, image, lesion_mask)

        measured_mean = float(np.mean(lesion_vox)) if lesion_vox.size > 0 else 0.0
        expected_mean = float(self._expected_lesion_mean(lesion, image.shape, lesion_mask))
        rc = measured_mean / expected_mean if abs(expected_mean) > 1e-9 else 0.0
        bias = 100.0 * (rc - 1.0)
        abs_bias = abs(bias)
        noise_roi_voxel_count = 0
        noise_roi_min_lesion_distance_mm = float("nan")
        noise_spill_exclusion_mm = float("nan")
        if scenario_noise_cache and scenario_name in scenario_noise_cache:
            cached = scenario_noise_cache[scenario_name]
            noise_cv = float(cached.get("noise_cv_percent", float("nan")))
            bg_mean = float(cached.get("noise_bg_mean", 0.0))
            bg_std = float(cached.get("noise_bg_std", 0.0))
            noise_roi_voxel_count = int(cached.get("noise_roi_voxel_count", 0))
            noise_roi_min_lesion_distance_mm = float(
                cached.get("noise_roi_min_lesion_distance_mm", float("nan"))
            )
            noise_spill_exclusion_mm = float(cached.get("noise_spill_exclusion_mm", float("nan")))
            bg_vox = image[bg_mask] if bg_mask is not None and np.any(bg_mask) else np.array([], dtype=np.float32)
        else:
            noise_cv, bg_mean, bg_std, noise_roi_voxel_count = self._noise_cv_from_background_voxels(
                image, bg_mask
            )
            bg_vox = image[bg_mask] if bg_mask is not None and np.any(bg_mask) else np.array([], dtype=np.float32)
        noise_ref_name = ""
        if self._use_global_liver_noise_roi() and self._noise_roi_center_mm is not None:
            mx, my, mz = self._noise_roi_center_mm
            noise_ref_name = (
                f"roi@({mx:.1f},{my:.1f},{mz:.1f})mm r={self._noise_roi_radius_mm:.1f}mm"
            )
        cnr = (measured_mean - bg_mean) / (bg_std + 1e-9)
        scenario_meta = self._scenario_meta.get(scenario_name, {"modality": "Unknown", "algo": "Unknown", "iter_number": 0})

        rec = {
            "scenario": scenario_name,
            "modality": scenario_meta.get("modality", "Unknown"),
            "recon_algo": scenario_meta.get("algo", "Unknown"),
            "iter_number": scenario_meta.get("iter_number", -1),
            "lesion_idx": lesion_idx + 1,
            "lesion_name": getattr(lesion, "name", f"Lesion_{lesion_idx+1}"),
            # Patient-derived library lesions carry a non-None custom_shape; uniform
            # spheres do not. The flag is exposed in records so downstream code
            # (table, plot, summary) can distinguish lesion kinds; DSI itself
            # gates per-feature on the well-definedness of the ground truth,
            # not on the lesion kind, so SUV-class stability is available even
            # for spheres while entropy-class features auto-skip when degenerate.
            "is_library": bool(getattr(lesion, "custom_shape", None) is not None),
            "lesion_kind": "library" if getattr(lesion, "custom_shape", None) is not None else "sphere",
            "volume_ml": self._lesion_volume_ml(lesion, image.shape),
            "lesion_radius_px": float(getattr(lesion, "radius", 0.0)),
            "expected_mean": expected_mean,
            "measured_mean": measured_mean,
            "rc_mean": rc,
            "cnr": cnr,
            "bias_percent": bias,
            "abs_bias_percent": abs_bias,
            "noise_cv_percent": noise_cv,
            "noise_bg_mean": bg_mean,
            "noise_bg_std": bg_std,
            "noise_roi_voxel_count": noise_roi_voxel_count,
            "noise_roi_min_lesion_distance_mm": noise_roi_min_lesion_distance_mm,
            "noise_spill_exclusion_mm": noise_spill_exclusion_mm,
            "noise_reference_mode": (
                "global_liver" if self._use_global_liver_noise_roi() else "per_lesion_ring"
            ),
            "noise_reference_lesion": noise_ref_name,
            "noise_voxel_source_scenario": scenario_name,
            "filter_fwhm_mm": float(fwhm_mm),
        }
        lesion_radiomics = self._radiomics(lesion_vox)
        rec.update(lesion_radiomics)
        rec.update(
            self._gibbs_metrics(
                image,
                lesion_mask,
                lesion_vox,
                bg_vox,
                lesion,
                reference_voxels,
            )
        )

        # Reference radiomics: ground truth from the Tab 3 lesion definition (mask + intensity model).
        # Per-feature stability is the absolute relative deviation from ground truth, matching the
        # paper's first-order detection-stability formulation (|x_measured − x_GT| / x_GT × 100%).
        # Features whose reference value is degenerate (essentially zero, e.g. std/energy/entropy
        # for a uniform-intensity sphere) make the relative-error metric undefined; we mark those
        # as NaN so the DSI panel reduction skips them rather than blowing up.
        zero_threshold = 1e-9
        degenerate_spread_features = frozenset(
            {"std", "cv_percent", "entropy", "energy", "skewness", "kurtosis"}
        )
        if reference_voxels.size > 0:
            reference_radiomics = self._radiomics(reference_voxels)
            for key, base_val in reference_radiomics.items():
                rec[f"ground_truth_{key}"] = base_val
                cur_val = lesion_radiomics.get(key, 0.0)
                base_finite = np.isfinite(base_val)
                cur_finite = np.isfinite(cur_val)
                abs_base = abs(float(base_val)) if base_finite else 0.0
                abs_cur = abs(float(cur_val)) if cur_finite else 0.0
                if key in degenerate_spread_features and base_finite:
                    ref_std = float(np.std(reference_voxels.astype(np.float64)))
                    if ref_std < zero_threshold:
                        rec[f"ground_truth_{key}"] = float("nan")
                        rec[f"delta_{key}_percent"] = float("nan")
                        rec[f"stability_{key}"] = float("nan")
                        continue
                if not base_finite or not cur_finite:
                    rec[f"delta_{key}_percent"] = float("nan")
                    rec[f"stability_{key}"] = float("nan")
                elif abs_base < zero_threshold:
                    if abs_cur < zero_threshold:
                        rec[f"delta_{key}_percent"] = 0.0
                        rec[f"stability_{key}"] = 0.0
                    else:
                        rec[f"delta_{key}_percent"] = float("nan")
                        rec[f"stability_{key}"] = float("nan")
                else:
                    rec[f"delta_{key}_percent"] = 100.0 * (cur_val - base_val) / abs_base
                    rec[f"stability_{key}"] = 100.0 * abs(cur_val - base_val) / abs_base
        return rec

    def run_analysis(self):
        scenarios = self._selected_scenarios()
        lesions = self._selected_lesions()
        self._last_selected_scenario_count = len(scenarios)
        self._last_selected_lesion_count = len(lesions)
        if self._use_global_liver_noise_roi():
            if self._noise_roi_center_mm is None:
                QMessageBox.warning(
                    self,
                    "Noise ROI required",
                    "Select Global liver ROI and click Define liver noise ROI to place the sphere.",
                )
                return
        if not scenarios:
            QMessageBox.warning(self, "No Scenario", "Select at least one reconstruction scenario.")
            return
        if not lesions:
            QMessageBox.warning(self, "No Lesion", "Select at least one lesion.")
            return

        mode = self.analysis_mode_combo.currentText()
        records = []
        scenario_noise_cache = self._build_scenario_noise_cache(scenarios, lesions, mode)
        if mode == "Noise-Bias Curve" and self._use_global_liver_noise_roi():
            low_vox = [
                name
                for name, entry in scenario_noise_cache.items()
                if int(entry.get("noise_roi_voxel_count", 0)) < 50
            ]
            if low_vox:
                QMessageBox.warning(
                    self,
                    "Noise ROI may be invalid",
                    "Few tissue voxels in the global liver ROI for: "
                    f"{', '.join(low_vox[:4])}"
                    f"{'…' if len(low_vox) > 4 else ''}.\n\n"
                    "Re-place the sphere in uniform liver (away from lesions and air). "
                    "Including air or lesion spill inflates noise CV% and hides BSREM benefit.",
                )
            if scenario_noise_cache:
                sample = next(iter(scenario_noise_cache.values()))
                min_dist = float(sample.get("noise_roi_min_lesion_distance_mm", float("nan")))
                spill_mm = float(sample.get("noise_spill_exclusion_mm", 10.0))
                need_clear = spill_mm + float(self._noise_roi_radius_mm)
                if np.isfinite(min_dist) and min_dist < need_clear:
                    QMessageBox.warning(
                        self,
                        "ROI too close to library lesion",
                        f"Liver ROI center is only {min_dist:.1f} mm from the nearest lesion "
                        f"(need ≳ {need_clear:.1f} mm including spill margin for non-sphere lesions).\n\n"
                        "Reconstruction spill extends beyond the lesion mask; nearby ROIs "
                        "pick up edge signal and inflate noise CV%. Move the ROI farther from "
                        "all lesions (especially R-Liver if it sits in the liver).",
                    )

        if mode == "Post-Recon Filtering Effect":
            try:
                start = float(self.filter_start_input.text())
                stop = float(self.filter_stop_input.text())
                step = float(self.filter_step_input.text())
                if step <= 0:
                    raise ValueError("step")
            except Exception:
                QMessageBox.warning(self, "Invalid Sweep", "Use numeric filter start/stop/step and a positive step.")
                return
            if len(scenarios) > 1:
                QMessageBox.information(
                    self,
                    "Filter Sweep",
                    "Multiple scenarios selected. The sweep will run for all selected scenarios.",
                )
            sweep = np.arange(start, stop + 1e-9, step)
            for scenario_name, image in scenarios:
                for fwhm in sweep:
                    filt_image = self._apply_gaussian_fwhm(image, fwhm)
                    for lesion_idx, lesion in lesions:
                        records.append(
                            self._compute_record(
                                scenario_name,
                                filt_image,
                                lesion_idx,
                                lesion,
                                fwhm_mm=fwhm,
                                lesions=lesions,
                            )
                        )
        else:
            for scenario_name, image in scenarios:
                for lesion_idx, lesion in lesions:
                    records.append(
                        self._compute_record(
                            scenario_name,
                            image,
                            lesion_idx,
                            lesion,
                            fwhm_mm=0.0,
                            scenario_noise_cache=scenario_noise_cache,
                            lesions=lesions,
                        )
                    )

        self._current_records = records
        self._records_mode = mode
        self._records_signature = self._current_analysis_signature()
        self._force_blank_plot = False
        self._populate_table(mode, records)
        summary = (
            f"Selected scenarios: {self._last_selected_scenario_count} | "
            f"Selected lesions: {self._last_selected_lesion_count}"
        )
        if mode == "Radiomics":
            dsi_features = self._selected_radiomics_features()
            n_library = sum(1 for _, l in lesions if getattr(l, "custom_shape", None) is not None)
            n_sphere = len(lesions) - n_library
            paper_panel = {"peak", "p90", "entropy"}
            is_paper_default = set(dsi_features) == paper_panel
            panel_note = (
                "paper default: SUVmax, SUV90, first-order entropy"
                if is_paper_default
                else "user-selected panel"
            )
            summary += (
                " | Reference radiomics: ground truth from each lesion's Tab 3 definition "
                "(mask + uniform or template intensity on the recon grid). "
                "Detection Stability Index (DSI) = mean of |x_measured − x_GT| / |x_GT| × 100% "
                f"over the selected detection panel ({len(dsi_features)} feature(s) — {panel_note}: "
                f"{', '.join(dsi_features)}). "
                "Lower DSI is better — 0% means perfect agreement with ground truth; DSI is "
                "unbounded above (a 5×-overshoot in one feature contributes 400% to that term). "
                f"DSI is computed per lesion ({n_library} library + {n_sphere} sphere selected). "
                "Within each record, panel features whose ground-truth value is degenerate "
                "(near zero, e.g. entropy / energy / std on a uniform-intensity sphere) are "
                "skipped automatically; DSI averages only the remaining well-defined features, "
                "so on uniform spheres DSI naturally reduces to the SUV-class stability "
                "(SUVmax / SUV90 / SUVmean), while on library lesions all panel features contribute."
            )
        elif mode == "Gibbs Effect":
            summary += (
                " | Test 2 — Gibbs edge-artifact panel (voxel-level shell extrema). "
                "Overshoot % = (SUVmax / L − 1) × 100 — negative values indicate "
                "under-recovery (peak below truth), not Gibbs ringing. "
                "(Tong et al. 2011 IEEE TNS 58(5):2264; Bai & Esser 2010; Rahmim et al. 2013). "
                "Undershoot % = max(0, (b − A−)/b) × 100 — outside dip below local background "
                "(Tong 2011 §III-A; Snyder et al., IEEE TMI 6(3):228, 1987). "
                "GA % = (A+ − A−)/(A+ + A−) × 100 — rim-shell amplitude index "
                "(Rogasch & Hofheinz, EJNMMI Phys. 1:12, 2014). "
                "A+ = max voxel in inside-rim shell d ∈ [−rim,0]; A− = min voxel in outside-rim shell d ∈ (0,rim]; "
                "b = mean voxel in far-bg shell past d > max(rim, FWHM); rim = max(0.5·FWHM, voxel). "
                "Lesions below ~2·w equivalent diameter are excluded from GA / Undershoot."
            )
        self.analysis_summary_label.setText(summary)
        # Ensure axis combos are valid for the active mode before drawing on current tab.
        self._update_mode_visibility()
        self.plot_current_results()

    def _populate_table(self, mode, records):
        self.results_table.setSortingEnabled(False)
        if not records:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.results_table.setSortingEnabled(True)
            return

        # Deterministic ordering for easier QC: algorithm -> iteration -> lesion -> scenario
        records = sorted(
            records,
            key=lambda r: (
                str(r.get("recon_algo", "")),
                int(r.get("iter_number", 0)),
                str(r.get("lesion_name", "")),
                str(r.get("scenario", "")),
            ),
        )

        if mode == "Radiomics":
            cols = [
                "scenario",
                "modality",
                "recon_algo",
                "iter_number",
                "lesion_name",
                "volume_ml",
                "dsi_percent",
            ]
            for m in ["peak", "p90", "entropy", "mean", "std", "cv_percent", "energy", "p10", "median", "skewness", "kurtosis"]:
                cols.append(m)
                cols.append(f"ground_truth_{m}")
            # Compute DSI fresh from the current feature-panel selection so the table
            # tracks user toggles without re-running the analysis.
            dsi_features = self._selected_radiomics_features()
            for rec in records:
                rec["dsi_percent"] = self._dsi_for_record(rec, feature_keys=dsi_features)
        elif mode == "Gibbs Effect":
            cols = [
                "scenario",
                "modality",
                "recon_algo",
                "iter_number",
                "lesion_name",
                "volume_ml",
                "edge_window_mm",
                "expected_mean",
                "measured_mean",
                "overshoot_percent",
                "undershoot_percent",
                "ga_percent",
            ]
        elif mode == "Noise-Bias Curve":
            cols = [
                "scenario", "modality", "recon_algo", "iter_number", "lesion_name",
                "expected_mean", "measured_mean", "bias_percent", "abs_bias_percent",
                "noise_cv_percent", "noise_bg_mean", "noise_roi_voxel_count",
                "noise_roi_min_lesion_distance_mm", "noise_spill_exclusion_mm",
                "noise_reference_mode", "noise_voxel_source_scenario",
                "rc_mean",
            ]
        elif mode == "Post-Recon Filtering Effect":
            cols = [
                "scenario",
                "modality",
                "recon_algo",
                "iter_number",
                "lesion_name",
                "filter_fwhm_mm",
                "measured_mean",
                "rc_mean",
                "cnr",
                "bias_percent",
                "noise_cv_percent",
            ]
        else:
            cols = ["scenario", "modality", "recon_algo", "iter_number", "lesion_name", "volume_ml", "expected_mean", "measured_mean", "rc_mean", "bias_percent"]

        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if len(cols) > 0:
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setRowCount(len(records))

        for r, rec in enumerate(records):
            for c, key in enumerate(cols):
                val = rec.get(key, None)
                if isinstance(val, float):
                    if not np.isfinite(val):
                        txt = "N/A"
                    else:
                        txt = f"{val:.5g}"
                elif val is None:
                    txt = "N/A"
                else:
                    txt = str(val)
                self.results_table.setItem(r, c, QTableWidgetItem(txt))
        self.results_table.setSortingEnabled(True)

    def plot_current_results(self):
        current_tab = self.figure_tabs.currentWidget()
        if current_tab is None:
            return
        ctx = next((c for c in self._figure_tabs_context if c.get("tab") is current_tab), None)
        if ctx is None:
            return
        self._draw_plot(ctx["plot"], ctx.get("legend"))

    def _save_figure_widget(self, capture_widget):
        if not self._current_records:
            QMessageBox.information(self, "No Data", "Run analysis first to generate a figure.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            "analysis_figure.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;Bitmap (*.bmp)",
        )
        if not path:
            return
        try:
            ok = capture_widget.grab().save(path)
            if not ok:
                raise IOError("Qt failed to save the figure image.")
            QMessageBox.information(self, "Figure saved", f"Saved figure to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save figure:\n{e}")

    def _legend_marker_icon(self, color_rgb, symbol_key, w=26, h=20):
        """Icon matching pyqtgraph scatter markers: color = algorithm, shape = lesion."""
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        r, g, b = (int(max(0, min(255, c))) for c in color_rgb)
        c = QColor(r, g, b)
        painter.setPen(QPen(c, 2))
        painter.setBrush(QBrush(c))
        cx, cy = w // 2, h // 2
        rad = max(3, min(w, h) // 2 - 3)
        sym = str(symbol_key)
        if sym == "o":
            painter.drawEllipse(cx - rad, cy - rad, 2 * rad, 2 * rad)
        elif sym == "s":
            painter.drawRect(cx - rad, cy - rad, 2 * rad, 2 * rad)
        elif sym == "t":
            # Match pyqtgraph "t" orientation (triangle-down).
            poly = QPolygon(
                [
                    QPoint(cx - rad, cy - rad + 1),
                    QPoint(cx + rad, cy - rad + 1),
                    QPoint(cx, cy + rad),
                ]
            )
            painter.drawPolygon(poly)
        elif sym == "t1":
            # pyqtgraph "t1" = triangle pointing up.
            poly = QPolygon(
                [
                    QPoint(cx, cy - rad),
                    QPoint(cx - rad, cy + rad),
                    QPoint(cx + rad, cy + rad),
                ]
            )
            painter.drawPolygon(poly)
        elif sym == "t2":
            # pyqtgraph "t2" = triangle pointing right.
            poly = QPolygon(
                [
                    QPoint(cx + rad, cy),
                    QPoint(cx - rad, cy - rad),
                    QPoint(cx - rad, cy + rad),
                ]
            )
            painter.drawPolygon(poly)
        elif sym == "t3":
            # pyqtgraph "t3" = triangle pointing left.
            poly = QPolygon(
                [
                    QPoint(cx - rad, cy),
                    QPoint(cx + rad, cy - rad),
                    QPoint(cx + rad, cy + rad),
                ]
            )
            painter.drawPolygon(poly)
        elif sym == "d":
            poly = QPolygon(
                [
                    QPoint(cx, cy - rad),
                    QPoint(cx + rad - 1, cy),
                    QPoint(cx, cy + rad),
                    QPoint(cx - rad + 1, cy),
                ]
            )
            painter.drawPolygon(poly)
        elif sym == "+":
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(cx - rad, cy, cx + rad, cy)
            painter.drawLine(cx, cy - rad, cx, cy + rad)
        elif sym == "x":
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(cx - rad, cy - rad, cx + rad, cy + rad)
            painter.drawLine(cx - rad, cy + rad, cx + rad, cy - rad)
        elif sym == "p":
            pts = []
            for i in range(5):
                ang = -pi / 2.0 + i * 2.0 * pi / 5.0
                pts.append(QPoint(int(cx + rad * cos(ang)), int(cy + rad * sin(ang))))
            painter.drawPolygon(QPolygon(pts))
        elif sym == "h":
            pts = []
            for i in range(6):
                ang = -pi / 2.0 + i * 2.0 * pi / 6.0
                pts.append(QPoint(int(cx + rad * cos(ang)), int(cy + rad * sin(ang))))
            painter.drawPolygon(QPolygon(pts))
        elif sym == "star":
            pts = []
            for i in range(10):
                ang = pi / 2.0 + i * pi / 5.0
                rad_i = rad if i % 2 == 0 else max(2, rad // 2)
                pts.append(QPoint(int(cx + rad_i * cos(ang)), int(cy - rad_i * sin(ang))))
            painter.drawPolygon(QPolygon(pts))
        else:
            painter.drawEllipse(cx - rad, cy - rad, 2 * rad, 2 * rad)
        painter.end()
        return QIcon(pm)

    def _algo_color_map(self):
        return {
            "OSEM": (60, 170, 255),
            "BSREM": (255, 70, 170),
            "OSMAPOSL": (80, 210, 130),
            "KEM": (220, 120, 255),
        }

    # Per-symbol visual-size compensation. pyqtgraph's symbolSize is the bounding-box
    # size, so a triangle/diamond/plus of the same symbolSize as a circle looks smaller
    # because they cover less of their bbox. These multipliers normalize perceived size.
    _LESION_SYMBOL_SIZE_FACTOR = {
        "o": 1.00,
        "s": 1.00,
        "t": 1.30,
        "t1": 1.30,
        "t2": 1.30,
        "t3": 1.30,
        "d": 1.25,
        "p": 1.15,
        "h": 1.10,
        "star": 1.25,
        "+": 1.55,
        "x": 1.55,
    }

    def _lesion_symbol_and_size(self, rec):
        """Stable marker per lesion index; cycles the symbol pool then grows symbol size.

        Returns (symbol, symbolSize) where symbolSize is scaled by a per-shape factor
        so all shapes look perceptually similar in size on the plot.
        """
        pool = self._LESION_SYMBOL_POOL
        n = len(pool)
        li = max(1, int(rec.get("lesion_idx", 1)))
        sym = pool[(li - 1) % n]
        tier = max(0, (li - 1) // n)
        base_size = 7 + min(tier, 5)
        scale = self._LESION_SYMBOL_SIZE_FACTOR.get(sym, 1.0)
        return sym, float(base_size * scale)

    def _lesion_symbol_for_legend_name(self, lesion_name):
        for r in self._current_records or []:
            if str(r.get("lesion_name", "")).strip() == str(lesion_name).strip():
                return self._lesion_symbol_and_size(r)[0]
        h = int(hashlib.md5(str(lesion_name).encode("utf-8")).hexdigest(), 16)
        return self._LESION_SYMBOL_POOL[h % len(self._LESION_SYMBOL_POOL)]

    def _iteration_visual_maps(self, records, x_axis_metric):
        """When X is not iteration and multiple iterations exist: line dash + rank for marker outline."""
        if not records or x_axis_metric == "iter_number":
            return None, None
        iters = sorted({int(r.get("iter_number", 0)) for r in records})
        if len(iters) <= 1:
            return None, None
        styles = (Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine)
        pen_map = {it: styles[i % len(styles)] for i, it in enumerate(iters)}
        rank_map = {it: i for i, it in enumerate(iters)}
        return pen_map, rank_map

    def _series_plot_tooltip(self, rec):
        return (
            f"Scenario: {rec.get('scenario', '')}\n"
            f"Algorithm: {rec.get('recon_algo', '')}\n"
            f"Iteration: {rec.get('iter_number', '')}\n"
            f"Lesion: {rec.get('lesion_name', '')}"
        )

    def _legend_dash_line_icon(self, pen_style, w=44, h=16):
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(216, 222, 233), 2, pen_style))
        painter.drawLine(4, h // 2, w - 4, h // 2)
        painter.end()
        return QIcon(pm)

    def _compact_legend_default_label(self, key):
        s = str(key)
        if s.startswith("__algo__"):
            return s[len("__algo__") :]
        if s.startswith("__lesion__"):
            return s[len("__lesion__") :]
        if s == "__ref__phantom":
            return "Reference"
        return s

    def _build_compact_legend(self, legend_list, include_reference=False, x_axis_metric=None):
        """One row per algorithm (colored square) and one row per lesion (name + shape)."""
        if legend_list is None or not self.show_legend_checkbox.isChecked():
            return
        records = self._current_records or []

        def display_name(k):
            return self._legend_aliases.get(str(k), self._compact_legend_default_label(k))

        algo_color = self._algo_color_map()
        algos = set()
        lesions = set()
        for rec in records:
            a = str(rec.get("recon_algo", "Unknown")).strip() or "Unknown"
            algos.add(a.upper())
            ln = rec.get("lesion_name")
            if ln is not None and str(ln).strip():
                lesions.add(str(ln).strip())

        def algo_sort_key(a):
            order = {"OSEM": 0, "BSREM": 1, "OSMAPOSL": 2, "KEM": 3}
            return (order.get(a.upper(), 99), a)

        sorted_algos = sorted(algos, key=algo_sort_key)
        sorted_lesions = sorted(lesions)
        neutral = (216, 222, 233)

        self._legend_refresh_in_progress = True
        for algo in sorted_algos:
            k = f"__algo__{algo}"
            color = algo_color.get(algo.upper(), (200, 200, 200))
            item = QListWidgetItem(display_name(k))
            item.setData(Qt.UserRole, k)
            item.setIcon(self._legend_marker_icon(color, "s"))
            item.setForeground(QColor(216, 222, 233))
            legend_list.addItem(item)
        for ln in sorted_lesions:
            k = f"__lesion__{ln}"
            sym = self._lesion_symbol_for_legend_name(ln)
            item = QListWidgetItem(display_name(k))
            item.setData(Qt.UserRole, k)
            item.setIcon(self._legend_marker_icon(neutral, sym))
            item.setForeground(QColor(216, 222, 233))
            legend_list.addItem(item)
        legend_list.setToolTip(
            "Algorithms: colored squares. Lesions: marker shape. "
            "Hover any curve or markers for scenario, algorithm, iteration, and lesion. "
            "Double-click a row to rename (algorithms and lesions only)."
        )
        if include_reference:
            k = "__ref__phantom"
            ref_color = (140, 140, 150)
            item = QListWidgetItem(display_name(k))
            item.setData(Qt.UserRole, k)
            item.setIcon(self._legend_marker_icon(ref_color, "s"))
            item.setForeground(QColor(216, 222, 233))
            legend_list.addItem(item)
        self._legend_refresh_in_progress = False

    def _recenter_y_axis_label(self, plot):
        """Vertically center the rotated left-axis title (pyqtgraph AxisItem.resizeEvent)."""
        try:
            left = plot.getPlotItem().getAxis("left")
            if left is None or left.label is None:
                return
            text = str(getattr(left, "labelText", "") or "").strip()
            if not text:
                return
            # pyqtgraph rotates the label on the left axis; resizeEvent places it at
            # the vertical midpoint. setAngle() does not exist on QGraphicsTextItem —
            # calling it raised AttributeError and prevented all prior centering attempts.
            left.resizeEvent()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _schedule_y_label_recenter(self, plot):
        """Re-run label centering after Qt/pyqtgraph finish layout."""
        if plot is None:
            return
        self._recenter_y_axis_label(plot)
        QTimer.singleShot(0, lambda p=plot: self._recenter_y_axis_label(p))
        QTimer.singleShot(100, lambda p=plot: self._recenter_y_axis_label(p))
        if not getattr(plot, "_ylabel_recenter_hooked", False):
            plot._ylabel_recenter_hooked = True
            try:
                vb = plot.getPlotItem().getViewBox()
                vb.sigResized.connect(lambda: self._recenter_y_axis_label(plot))
            except (AttributeError, RuntimeError):
                pass

    def _apply_analysis_plot_axis_style(self, plot):
        """Readable axes; keep tick spacing and move Y-label farther from tick numbers."""
        axis_pen = pg.mkPen(236, 239, 244, width=1)
        text_pen = pg.mkPen(216, 222, 233)
        label_font = QFont()
        label_font.setPointSize(11)
        label_font.setWeight(QFont.Normal)
        tick_font = QFont()
        tick_font.setPointSize(10)
        tick_font.setWeight(QFont.Normal)
        left = plot.getAxis("left")
        bottom = plot.getAxis("bottom")
        for ax in (left, bottom):
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)
        try:
            left.setStyle(tickFont=tick_font, tickTextOffset=10)
        except (TypeError, AttributeError):
            try:
                left.setStyle(tickTextOffset=10)
            except (TypeError, AttributeError):
                pass
        try:
            bottom.setStyle(tickFont=tick_font, tickTextOffset=14)
        except (TypeError, AttributeError):
            try:
                bottom.setStyle(tickTextOffset=14)
            except (TypeError, AttributeError):
                pass
        try:
            y_lbl = str(getattr(left, "labelText", "") or "").strip()
            n = len(y_lbl.strip())
            tick_room = 40
            label_room = max(24, min(int(n * 5.5), 48))
            w = tick_room + label_room
            w = max(68, min(w, 100))
            left.setWidth(w)
        except (AttributeError, TypeError):
            pass
        for ax in (left, bottom):
            try:
                if getattr(ax, "label", None) is not None:
                    ax.label.setFont(label_font)
            except (AttributeError, RuntimeError):
                pass
        self._recenter_y_axis_label(plot)

    def _trajectory_group_key(self, rec):
        """One curve per reconstruction algorithm and lesion (iterations are points along the curve)."""
        algo = str(rec.get("recon_algo", "Unknown")).strip() or "Unknown"
        lesion = str(rec.get("lesion_name", "")).strip() or f"Lesion_{rec.get('lesion_idx', 0)}"
        return f"{algo} | {lesion}"

    def _group_records_for_trajectory(self, records):
        grouped = {}
        for rec in records:
            grouped.setdefault(self._trajectory_group_key(rec), []).append(rec)
        return grouped

    def _sort_records_for_plot(self, rows, x_key):
        if not rows:
            return rows
        iter_vals = {int(r.get("iter_number", 0)) for r in rows}
        if len(iter_vals) > 1:
            return sorted(rows, key=lambda r: int(r.get("iter_number", 0)))
        if x_key == "filter_fwhm_mm":
            return sorted(rows, key=lambda r: float(r.get("filter_fwhm_mm", 0.0)))
        if x_key == "iter_number":
            return sorted(rows, key=lambda r: int(r.get("iter_number", 0)))
        return sorted(rows, key=lambda r: float(r.get(x_key, 0.0)))

    @staticmethod
    def _smooth_curve_through_points(x, y, n_samples=120):
        """
        Single smooth curve through iteration-ordered points (PCHIP, no overshoot).
        Parameterizes by point index so iteration sequence is preserved.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = int(x.size)
        if n < 3:
            return x, y
        t = np.arange(n, dtype=np.float64)
        ts = np.linspace(0.0, n - 1, max(int(n_samples), n * 15))
        try:
            xs = scipy.interpolate.PchipInterpolator(t, x)(ts)
            ys = scipy.interpolate.PchipInterpolator(t, y)(ts)
            return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        except (ValueError, TypeError):
            return x, y

    def _metric_values_for_plot(self, rows, key, mode=None, x_key=None):
        vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        # Paper noise–bias curves use |bias| on the horizontal axis; signed bias on Y stays signed.
        if mode == "Noise-Bias Curve" and key == "bias_percent" and x_key == key:
            vals = np.abs(vals)
        return vals

    def _apply_iteration_axis_limits(self, target_plot, x_key, records):
        if x_key != "iter_number" or not records:
            return
        iters = sorted({int(r.get("iter_number", 0)) for r in records})
        if not iters:
            return
        lo, hi = float(iters[0]), float(iters[-1])
        pad = max(0.5, min(0.15 * max(hi - lo, 1.0), 2.0))
        try:
            target_plot.setLimits(xMin=max(0.5, lo - pad))
        except Exception:
            pass
        target_plot.setXRange(max(0.5, lo - pad), hi + pad, padding=0.0)

    def _draw_plot(self, target_plot, legend_list=None):
        target_plot.clear()
        # Reset any prior manual view range so each analysis mode starts from a fresh fit.
        try:
            target_plot.enableAutoRange(x=True, y=True)
        except Exception:
            pass
        mode = self.analysis_mode_combo.currentText()
        if self._force_blank_plot:
            if legend_list is not None:
                self._legend_refresh_in_progress = True
                legend_list.clear()
                self._legend_refresh_in_progress = False
            target_plot.setLabel("left", "")
            target_plot.setLabel("bottom", "")
            return
        if self._records_mode is not None and self._records_mode != mode:
            self._invalidate_analysis_results(
                "Settings changed. Click Run Analysis to refresh results."
            )
            if legend_list is not None:
                self._legend_refresh_in_progress = True
                legend_list.clear()
                self._legend_refresh_in_progress = False
            target_plot.setLabel("left", "")
            target_plot.setLabel("bottom", "")
            return
        # If selections/sweep changed since last run, do not render stale records.
        current_sig = self._current_analysis_signature()
        if self._records_signature is not None and self._records_signature != current_sig:
            self._invalidate_analysis_results(
                "Settings changed. Click Run Analysis to refresh results."
            )
            if legend_list is not None:
                self._legend_refresh_in_progress = True
                legend_list.clear()
                self._legend_refresh_in_progress = False
            target_plot.setLabel("left", "")
            target_plot.setLabel("bottom", "")
            return
        if not self._current_records:
            if legend_list is not None:
                self._legend_refresh_in_progress = True
                legend_list.clear()
                self._legend_refresh_in_progress = False
            return

        use_external_legend = legend_list is not None
        if use_external_legend:
            self._legend_refresh_in_progress = True
            legend_list.setVisible(self.show_legend_checkbox.isChecked())
            legend_list.clear()
            self._legend_refresh_in_progress = False
        elif self.show_legend_checkbox.isChecked():
            target_plot.addLegend(offset=(10, 10))

        def style_for_record(rec):
            algo = str(rec.get("recon_algo", "")).upper()
            algo_color_map = self._algo_color_map()
            color = algo_color_map.get(algo, (200, 200, 200))
            line_style = Qt.SolidLine
            symbol, sym_sz = self._lesion_symbol_and_size(rec)
            return color, symbol, line_style, sym_sz

        x_key = self.x_axis_combo.currentData()
        y_key = self.y_axis_combo.currentData()
        if not x_key or not y_key:
            return

        if mode == "Post-Recon Filtering Effect":
            iter_pen_map, iter_rank_map = None, None
        else:
            iter_pen_map, iter_rank_map = self._iteration_visual_maps(self._current_records, x_key)

        def line_pen_for_rec(rec, color, width=2):
            st = Qt.SolidLine
            if iter_pen_map is not None:
                st = iter_pen_map.get(int(rec.get("iter_number", 0)), Qt.SolidLine)
            return pg.mkPen(color, width=width, style=st)

        def symbol_pen_for_rec(rec, color, x_arr):
            # Keep marker outlines visually consistent across all points.
            return None

        def _values_for_metric(metric_key):
            vals = []
            for r in self._current_records:
                try:
                    v = float(r.get(metric_key, np.nan))
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    continue
            return vals

        def _apply_nonnegative_volume_axis_limits(x_metric, y_metric):
            # Lesion volume is physically non-negative; keep axes anchored at 0.
            for axis_name, metric in (("x", x_metric), ("y", y_metric)):
                if metric != "volume_ml":
                    continue
                vals = _values_for_metric(metric)
                max_v = max(vals) if vals else 0.0
                upper = max(1.0, max_v * 1.08)
                if axis_name == "x":
                    target_plot.setXRange(0.0, upper, padding=0.0)
                    try:
                        target_plot.setLimits(xMin=0.0)
                    except Exception:
                        pass
                else:
                    target_plot.setYRange(0.0, upper, padding=0.0)
                    try:
                        target_plot.setLimits(yMin=0.0)
                    except Exception:
                        pass

        if mode == "Noise-Bias Curve":
            def axis_label_for_noise_bias(k):
                if k == "bias_percent":
                    return "Bias %"
                return self._metric_labels.get(k, k)

            target_plot.setLabel("left", axis_label_for_noise_bias(y_key))
            target_plot.setLabel("bottom", axis_label_for_noise_bias(x_key))
            grouped = self._group_records_for_trajectory(self._current_records)
            xs_list, ys_list = [], []
            for _name, rows in grouped.items():
                rows = self._sort_records_for_plot(rows, x_key)
                x = self._metric_values_for_plot(rows, x_key, mode=mode, x_key=x_key)
                y = self._metric_values_for_plot(rows, y_key, mode=mode, x_key=x_key)
                if x.size:
                    xs_list.append(np.asarray(x, dtype=float).ravel())
                if y.size:
                    ys_list.append(np.asarray(y, dtype=float).ravel())
            xe = np.concatenate(xs_list) if xs_list else np.array([], dtype=float)
            ye = np.concatenate(ys_list) if ys_list else np.array([], dtype=float)
            for _name, rows in grouped.items():
                rows = self._sort_records_for_plot(rows, x_key)
                x = self._metric_values_for_plot(rows, x_key, mode=mode, x_key=x_key)
                y = self._metric_values_for_plot(rows, y_key, mode=mode, x_key=x_key)
                color, symbol, _line_style, sym_sz = style_for_record(rows[0])
                pen = line_pen_for_rec(rows[0], color)
                brush = pg.mkBrush(color)
                sp = symbol_pen_for_rec(rows[0], color, x)
                sc = target_plot.plot(
                    x, y, pen=None, symbol=symbol, symbolSize=sym_sz, symbolBrush=brush, symbolPen=sp
                )
                if sc is not None:
                    sc.setToolTip(self._series_plot_tooltip(rows[0]))
                if x.size > 1:
                    xs, ys = self._smooth_curve_through_points(x, y)
                    ln = target_plot.plot(xs, ys, pen=pen)
                    if ln is not None:
                        ln.setToolTip(self._series_plot_tooltip(rows[0]))
            rx0, rx1, ry0, ry1 = self._noise_bias_axis_ranges(xe, ye, x_key, y_key)
            try:
                target_plot.setLimits(xMin=rx0, xMax=rx1, yMin=ry0, yMax=ry1)
            except Exception:
                pass
            target_plot.setXRange(rx0, rx1, padding=0.0)
            target_plot.setYRange(ry0, ry1, padding=0.0)
            if use_external_legend:
                self._build_compact_legend(legend_list, include_reference=False, x_axis_metric=x_key)
            _apply_nonnegative_volume_axis_limits(x_key, y_key)
            self._apply_analysis_plot_axis_style(target_plot)
            self._schedule_y_label_recenter(target_plot)
            return

        if mode == "Post-Recon Filtering Effect":
            # Clear any hard view limits left over from the Noise-Bias mode,
            # otherwise post-recon sweep curves get clipped to that x/y range.
            try:
                target_plot.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            except Exception:
                pass
            target_plot.setLabel("left", self._metric_labels.get(y_key, y_key))
            target_plot.setLabel("bottom", self._metric_labels.get(x_key, x_key))
            grouped = {}
            for rec in self._current_records:
                key = f"{rec['scenario']} | {rec['lesion_name']} | {rec['recon_algo']}"
                grouped.setdefault(key, []).append(rec)
            for idx, (name, rows) in enumerate(grouped.items()):
                rows = sorted(rows, key=lambda r: r["filter_fwhm_mm"])
                x = np.array([float(r.get(x_key, 0.0)) for r in rows], dtype=float)
                y = np.array([float(r.get(y_key, 0.0)) for r in rows], dtype=float)
                color, symbol, line_style, sym_sz = style_for_record(rows[0])
                pen = pg.mkPen(color, width=2, style=line_style)
                brush = pg.mkBrush(color)
                di = target_plot.plot(
                    x, y, pen=pen, symbol=symbol, symbolSize=max(6.0, sym_sz - 1.0), symbolBrush=brush
                )
                if di is not None:
                    di.setToolTip(self._series_plot_tooltip(rows[0]))
            if use_external_legend:
                self._build_compact_legend(legend_list, include_reference=False, x_axis_metric=x_key)
            _apply_nonnegative_volume_axis_limits(x_key, y_key)
            self._apply_analysis_plot_axis_style(target_plot)
            self._schedule_y_label_recenter(target_plot)
            return

        if mode == "Radiomics":
            y_key = self.y_axis_combo.currentData()
            if not y_key:
                return
            dsi_features = self._selected_radiomics_features()
            single_feature = dsi_features[0] if dsi_features else "mean"
            x_is_none = x_key == "__none__"
            y_is_none = y_key == "__none__"
            x_metric = "__sample_index__" if x_is_none else x_key
            y_metric = single_feature if y_is_none else y_key

            def _y_value(rec, key):
                if key == "dsi_percent":
                    return self._dsi_for_record(rec, feature_keys=dsi_features)
                try:
                    return float(rec.get(key, 0.0))
                except (TypeError, ValueError):
                    return float("nan")

            target_plot.setLabel("left", self._metric_labels.get(y_metric, y_metric))
            target_plot.setLabel("bottom", self._metric_labels.get(x_metric, x_metric))
            if x_is_none:
                grouped = {}
                for rec in self._current_records:
                    key = self._trajectory_group_key(rec)
                    grouped.setdefault(key, []).append(rec)
            else:
                grouped = self._group_records_for_trajectory(self._current_records)
            plotted_ref = False
            for _name, rows in grouped.items():
                if x_is_none:
                    rows = self._sort_records_for_plot(rows, "iter_number")
                    x = np.arange(1, len(rows) + 1, dtype=float)
                else:
                    rows = self._sort_records_for_plot(rows, x_key)
                    x = np.array([float(r.get(x_key, 0.0)) for r in rows], dtype=float)
                y = np.array([_y_value(r, y_metric) for r in rows], dtype=float)
                color, symbol, _line_style, sym_sz = style_for_record(rows[0])
                pen = line_pen_for_rec(rows[0], color)
                brush = pg.mkBrush(color)
                sp = symbol_pen_for_rec(rows[0], color, x)
                di = target_plot.plot(
                    x, y, pen=None, symbol=symbol, symbolSize=sym_sz, symbolBrush=brush, symbolPen=sp
                )
                if di is not None:
                    di.setToolTip(self._series_plot_tooltip(rows[0]))
                if x.size > 1:
                    ln = target_plot.plot(x, y, pen=pen)
                    if ln is not None:
                        ln.setToolTip(self._series_plot_tooltip(rows[0]))
            # Ground-truth reference: horizontal segment per lesion when X is iteration.
            if not str(y_metric).startswith("ground_truth_"):
                gt_key = f"ground_truth_{y_metric}"
                ref_color = (140, 140, 150)
                ref_pen = pg.mkPen(ref_color, width=2, style=Qt.DotLine)
                for _name, rows in grouped.items():
                    rows = self._sort_records_for_plot(rows, x_key if not x_is_none else "iter_number")
                    gy = rows[0].get(gt_key)
                    if gy is None or not np.isfinite(float(gy)):
                        continue
                    gfy = float(gy)
                    if x_is_none:
                        gx = np.array([1.0, float(len(rows))], dtype=float)
                    else:
                        gx = np.array(
                            [float(r.get(x_key, 0.0)) for r in rows],
                            dtype=float,
                        )
                        if gx.size < 1:
                            continue
                        if gx.size == 1:
                            gx = np.array([gx[0], gx[0] + 0.5], dtype=float)
                    ref_di = target_plot.plot(
                        [float(np.min(gx)), float(np.max(gx))],
                        [gfy, gfy],
                        pen=ref_pen,
                    )
                    if ref_di is not None:
                        ref_di.setToolTip(
                            "Phantom reference (ground truth)\n"
                            f"Lesion: {rows[0].get('lesion_name', '')}\n"
                            f"Feature: {y_metric}"
                        )
                    plotted_ref = True
            if use_external_legend:
                self._build_compact_legend(legend_list, include_reference=plotted_ref, x_axis_metric=x_key)
            _apply_nonnegative_volume_axis_limits(x_metric, y_metric)
            self._apply_iteration_axis_limits(target_plot, x_metric if not x_is_none else "iter_number", self._current_records)
            self._apply_analysis_plot_axis_style(target_plot)
            self._schedule_y_label_recenter(target_plot)
            return

        target_plot.setLabel("left", self._metric_labels.get(y_key, y_key))
        target_plot.setLabel("bottom", self._metric_labels.get(x_key, x_key))
        grouped = self._group_records_for_trajectory(self._current_records)
        for _name, rows in grouped.items():
            rows = self._sort_records_for_plot(rows, x_key)
            x = np.array([float(r.get(x_key, np.nan)) for r in rows], dtype=float)
            y = np.array([float(r.get(y_key, np.nan)) for r in rows], dtype=float)
            color, symbol, _line_style, sym_sz = style_for_record(rows[0])
            pen = line_pen_for_rec(rows[0], color)
            brush = pg.mkBrush(color)
            sp = symbol_pen_for_rec(rows[0], color, x)
            di = target_plot.plot(
                x, y, pen=None, symbol=symbol, symbolSize=sym_sz, symbolBrush=brush, symbolPen=sp
            )
            if di is not None:
                di.setToolTip(self._series_plot_tooltip(rows[0]))
            if x.size > 1:
                ln = target_plot.plot(x, y, pen=pen)
                if ln is not None:
                    ln.setToolTip(self._series_plot_tooltip(rows[0]))
        if mode == "Gibbs Effect" and y_key == "overshoot_percent":
            ref_pen = pg.mkPen((120, 120, 130), width=1, style=Qt.DashLine)
            target_plot.addLine(y=0.0, pen=ref_pen)
        if mode == "Recovery Coefficients":
            ref_pen = pg.mkPen((170, 170, 180), width=1, style=Qt.DashLine)
            target_plot.addLine(y=1.0, pen=ref_pen)
            # Ensure reference line is visible in the current Y-range.
            try:
                y_min, y_max = target_plot.viewRange()[1]
                if y_max < 1.0 or y_min > 1.0:
                    low = min(y_min, 1.0)
                    high = max(y_max, 1.0)
                    if abs(high - low) < 1e-6:
                        high = low + 1.0
                    target_plot.setYRange(low, high, padding=0.05)
            except Exception:
                pass
        if use_external_legend:
            self._build_compact_legend(legend_list, include_reference=False, x_axis_metric=x_key)
        _apply_nonnegative_volume_axis_limits(x_key, y_key)
        self._apply_iteration_axis_limits(target_plot, x_key, self._current_records)
        self._apply_analysis_plot_axis_style(target_plot)
        self._schedule_y_label_recenter(target_plot)
