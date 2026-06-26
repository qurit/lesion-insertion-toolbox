from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QCheckBox, QDialogButtonBox, QMessageBox, QHBoxLayout, 
                             QPushButton, QLabel, QWidget, QGroupBox,
                             QGridLayout, QSlider, QSpinBox, QComboBox, QListWidget, 
                             QListWidgetItem, QSplitter, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSizePolicy, QDoubleSpinBox, QAbstractItemView, QFileDialog, QStyle)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
import pyqtgraph as pg
import SimpleITK as sitk
import numpy as np
import os
import json
import logging
import uuid
import scipy.stats
import base64

# Attempt to import PyRadiomics, set flag if missing
try:
    from radiomics import featureextractor
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False

from src.widgets import MyColorBarItem, MyImageItem
from src.utils import (
    apply_lesion_precorrection,
    DEFAULT_PSF_FWHM_MM,
    library_morphology_support_from_shape,
    normalize_psf_fwhm_mm,
    resize_volume_preserve_morphology,
    uniform_library_patch_from_resized,
    wiener_deconv_roundtrip_nmse,
)

class MonteCarloSettingsDialog(QDialog):
    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Monte Carlo Simulation Settings")
        self.setMinimumWidth(500)
        self._apply_stylesheet()
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.n_events_input = QLineEdit()
        self.n_parallel_input = QLineEdit()
        self.isotope_names_input = QLineEdit()
        self.isotope_ratios_input = QLineEdit()
        self.crystal_thickness_input = QLineEdit()
        self.cover_thickness_input = QLineEdit()
        self.backscatter_thickness_input = QLineEdit()
        self.energy_res_140_input = QLineEdit()
        self.adv_collimator_checkbox = QCheckBox()

        form_layout.addRow("Number of Events:", self.n_events_input)
        form_layout.addRow("Parallel Processes:", self.n_parallel_input)
        form_layout.addRow("Isotope Names (comma-sep):", self.isotope_names_input)
        form_layout.addRow("Isotope Ratios (comma-sep):", self.isotope_ratios_input)
        form_layout.addRow("Crystal Thickness (cm):", self.crystal_thickness_input)
        form_layout.addRow("Cover Thickness (cm):", self.cover_thickness_input)
        form_layout.addRow("Backscatter Thickness (cm):", self.backscatter_thickness_input)
        form_layout.addRow("Energy Res @ 140keV (%):", self.energy_res_140_input)
        form_layout.addRow("Advanced Collimator Model:", self.adv_collimator_checkbox)
        layout.addLayout(form_layout)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.populate_fields(current_settings)

    def _apply_stylesheet(self):
        stylesheet = """
            QDialog { background-color: #2E3440; color: #D8DEE9; font-family: Arial, sans-serif; }
            QLabel { font-size: 11pt; color: #D8DEE9; }
            QLineEdit { border: 1px solid #4C566A; border-radius: 4px; padding: 6px; background-color: #434C5E; color: #D8DEE9; }
            QPushButton { background-color: #5E81AC; color: #ECEFF4; font-weight: bold; border-radius: 4px; padding: 8px; border: none; }
            QPushButton:hover { background-color: #81A1C1; }
            QCheckBox { color: #D8DEE9; }
        """
        self.setStyleSheet(stylesheet)

    def populate_fields(self, settings):
        if not settings:
            settings = self.get_defaults()
        self.n_events_input.setText(str(settings.get('mc_n_events')))
        self.n_parallel_input.setText(str(settings.get('mc_n_parallel')))
        self.isotope_names_input.setText(', '.join(settings.get('mc_isotope_names')))
        self.isotope_ratios_input.setText(', '.join(map(str, settings.get('mc_isotope_ratios'))))
        self.crystal_thickness_input.setText(str(settings.get('mc_crystal_thickness')))
        self.cover_thickness_input.setText(str(settings.get('mc_cover_thickness')))
        self.backscatter_thickness_input.setText(str(settings.get('mc_backscatter_thickness')))
        self.energy_res_140_input.setText(str(settings.get('mc_energy_res_140')))
        self.adv_collimator_checkbox.setChecked(settings.get('mc_adv_collimator'))

    def get_settings(self):
        try:
            settings = {
                'mc_n_events': int(self.n_events_input.text()),
                'mc_n_parallel': int(self.n_parallel_input.text()),
                'mc_isotope_names': [name.strip().lower() for name in self.isotope_names_input.text().split(',')],
                'mc_isotope_ratios': [float(ratio) for ratio in self.isotope_ratios_input.text().split(',')],
                'mc_crystal_thickness': float(self.crystal_thickness_input.text()),
                'mc_cover_thickness': float(self.cover_thickness_input.text()),
                'mc_backscatter_thickness': float(self.backscatter_thickness_input.text()),
                'mc_energy_res_140': float(self.energy_res_140_input.text()),
                'mc_adv_collimator': self.adv_collimator_checkbox.isChecked()
            }
            return settings
        except (ValueError, IndexError) as e:
            QMessageBox.critical(self, "Input Error", f"Invalid format: {e}")
            return None

    def get_defaults(self):
        return {
            'mc_n_events': 10000000,
            'mc_n_parallel': 8,
            'mc_isotope_names': ['lu177'],
            'mc_isotope_ratios': [1.0],
            'mc_crystal_thickness': 0.9525,
            'mc_cover_thickness': 0.1,
            'mc_backscatter_thickness': 6.6,
            'mc_energy_res_140': 10,
            'mc_adv_collimator': True
        }

class OccurrenceViewerWidget(QWidget):
    """
    A helper widget that displays Axial, Sagittal, and Coronal views 
    for a single modality (CT or PET) with professional controls.
    """
    sigViewChanged = pyqtSignal(int, int, int)

    def __init__(self, title, colormap_name, is_ct=False, parent=None):
        super().__init__(parent)
        self.title = title
        self.colormap_name = colormap_name
        self.is_ct = is_ct
        self.image_array = None
        self.sitk_image = None
        
        # --- Cache for 3D stats (Prevents jumping colorbar) ---
        self.data_min = 0.0
        self.data_max = 1.0
        
        self.initUI()
        self._setup_colormap()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # 1. Header
        header_layout = QHBoxLayout()
        self.load_btn = QPushButton(f"Load {self.title}")
        self.register_btn = QPushButton("Register to Patient")
        self.register_btn.setEnabled(False)
        self.register_btn.setStyleSheet("background-color: #BF616A;")
        
        title_container = QVBoxLayout()
        title_lbl = QLabel(f"<b>{self.title} Viewer</b>")
        title_lbl.setStyleSheet("font-size: 12pt; color: #88C0D0;")
        self.value_label = QLabel("Value: N/A")
        self.value_label.setStyleSheet("color: #EBCB8B; font-weight: bold;")
        
        title_container.addWidget(title_lbl)
        title_container.addWidget(self.value_label)
        
        header_layout.addLayout(title_container)
        header_layout.addStretch()
        header_layout.addWidget(self.register_btn)
        header_layout.addWidget(self.load_btn)
        main_layout.addLayout(header_layout)

        # 2. Images Row + Color Bar
        images_row_layout = QHBoxLayout()
        images_row_layout.setSpacing(10)

        self.view_axial = self._create_plot_widget("Axial (XY)")
        self.view_coronal = self._create_plot_widget("Coronal (XZ)")
        self.view_sagittal = self._create_plot_widget("Sagittal (YZ)")

        self.img_item_ax = MyImageItem()
        self.img_item_cor = MyImageItem()
        self.img_item_sag = MyImageItem()

        self.view_axial.addItem(self.img_item_ax)
        self.view_coronal.addItem(self.img_item_cor)
        self.view_sagittal.addItem(self.img_item_sag)

        images_row_layout.addWidget(self.view_axial, 1)
        images_row_layout.addWidget(self.view_coronal, 1)
        images_row_layout.addWidget(self.view_sagittal, 1)
        
        # Color Bar
        self.gl_widget = pg.GraphicsLayoutWidget()
        self.gl_widget.setMaximumWidth(70) 
        self.color_bar = MyColorBarItem(values=(0, 1), width=15)
        self.gl_widget.addItem(self.color_bar)
        images_row_layout.addWidget(self.gl_widget, 0) 

        main_layout.addLayout(images_row_layout, stretch=3)

        # 3. Controls
        controls_container = QGroupBox("Controls")
        controls_container.setFixedHeight(150) 
        controls_layout = QVBoxLayout(controls_container)
        
        if self.is_ct:
            wl_layout = QHBoxLayout()
            self.spin_width = QSpinBox()
            self.spin_width.setRange(1, 4000)
            self.spin_width.setValue(400)
            self.spin_width.setPrefix("W: ")
            
            self.spin_level = QSpinBox()
            self.spin_level.setRange(-2000, 2000)
            self.spin_level.setValue(50)
            self.spin_level.setPrefix("L: ")
            
            wl_layout.addWidget(self.spin_level)
            wl_layout.addWidget(self.spin_width)
            
            apply_wl_btn = QPushButton("Set")
            apply_wl_btn.setFixedWidth(40)
            apply_wl_btn.clicked.connect(self._apply_manual_windowing)
            wl_layout.addWidget(apply_wl_btn)
            
            controls_layout.addLayout(wl_layout)

        sliders_form = QFormLayout()
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_z = QSlider(Qt.Horizontal)
        self.label_x = QLabel("X: 0")
        self.label_y = QLabel("Y: 0")
        self.label_z = QLabel("Z: 0")
        
        sliders_form.addRow(self.label_x, self.slider_x)
        sliders_form.addRow(self.label_y, self.slider_y)
        sliders_form.addRow(self.label_z, self.slider_z)
        
        controls_layout.addLayout(sliders_form)
        main_layout.addWidget(controls_container)

        # Connect Signals
        self.load_btn.clicked.connect(self.load_image)
        self.register_btn.clicked.connect(self.register_to_patient)
        self.slider_x.valueChanged.connect(self.update_views)
        self.slider_y.valueChanged.connect(self.update_views)
        self.slider_z.valueChanged.connect(self.update_views)
        
        self.slider_x.valueChanged.connect(self._emit_view_change)
        self.slider_y.valueChanged.connect(self._emit_view_change)
        self.slider_z.valueChanged.connect(self._emit_view_change)

        self.view_axial.scene().sigMouseMoved.connect(lambda pos: self._on_mouse_move(pos, 'axial'))
        self.view_coronal.scene().sigMouseMoved.connect(lambda pos: self._on_mouse_move(pos, 'coronal'))
        self.view_sagittal.scene().sigMouseMoved.connect(lambda pos: self._on_mouse_move(pos, 'sagittal'))

        self.color_bar.sigLevelsChanged.connect(self._update_levels)

    def _create_plot_widget(self, title):
        pw = pg.PlotWidget()
        pw.setTitle(title, color='#ECEFF4', size='10pt')
        pw.hideAxis('left')
        pw.hideAxis('bottom')
        pw.setAspectLocked(True)
        pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pw.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 0, 150), style=Qt.DotLine))
        pw.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 255, 0, 150), style=Qt.DotLine))
        pw.addItem(pw.vLine, ignoreBounds=True)
        pw.addItem(pw.hLine, ignoreBounds=True)
        return pw

    def _setup_colormap(self):
        if self.is_ct or self.colormap_name == 'gray':
            pos = [0.0, 1.0]
            colors = [(0, 0, 0, 255), (255, 255, 255, 255)]
            cmap = pg.ColorMap(pos=pos, color=colors)
        else:
            try:
                cmap = pg.colormap.get(self.colormap_name)
            except:
                pos = [0.0, 1.0]
                colors = [(0, 0, 0, 255), (255, 255, 255, 255)]
                cmap = pg.ColorMap(pos=pos, color=colors)
        
        self.lut = cmap.getLookupTable(nPts=512)
        self.color_bar.setColorMap(cmap)

    def _update_levels(self):
        levels = self.color_bar.levels()
        self.img_item_ax.setLevels(levels)
        self.img_item_cor.setLevels(levels)
        self.img_item_sag.setLevels(levels)

    def _apply_manual_windowing(self):
        if self.image_array is None: return
        
        w = self.spin_width.value()
        l = self.spin_level.value()
        win_min = l - w/2
        win_max = l + w/2
        
        d_min = self.data_min
        d_max = self.data_max
        
        rng = d_max - d_min
        if rng <= 0: rng = 1.0
        
        p1 = np.clip((win_min - d_min) / rng, 0.0, 1.0)
        p2 = np.clip((win_max - d_min) / rng, 0.0, 1.0)
        if p2 <= p1: p2 = p1 + 1e-5
        
        pos = [0.0, p1, p2, 1.0]
        colors = [
            (0, 0, 0, 255),       # Below window: Black
            (0, 0, 0, 255),       # Window Start: Black
            (255, 255, 255, 255), # Window End: White
            (255, 255, 255, 255)  # End: White
        ]
        
        cm = pg.ColorMap(pos=pos, color=colors)
        self.lut = cm.getLookupTable(nPts=512)
        
        self.color_bar.setColorMap(cm)
        self.color_bar.setLevels((d_min, d_max))
        self.update_views()

    def load_image(self):
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(self, f"Load {self.title}", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
            if path:
                filename = os.path.basename(path)
                if filename.startswith("._"):
                    QMessageBox.warning(self, "Invalid File", "MacOS metadata file detected.")
                    return

                try:
                    self.sitk_image = sitk.ReadImage(path)
                    
                    self.slider_x.blockSignals(True)
                    self.slider_y.blockSignals(True)
                    self.slider_z.blockSignals(True)
                    
                    self._update_display_from_sitk()
                    self.register_btn.setEnabled(True)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load map: {e}")
                finally:
                    self.slider_x.blockSignals(False)
                    self.slider_y.blockSignals(False)
                    self.slider_z.blockSignals(False)
                    self.update_views()

    def _update_display_from_sitk(self):
        if self.sitk_image is None: return
        arr = sitk.GetArrayFromImage(self.sitk_image)
        self.image_array = np.transpose(arr, (2, 1, 0)).astype(np.float32)
        
        self.data_min = np.min(self.image_array)
        self.data_max = np.max(self.image_array)
        if self.data_max <= self.data_min: self.data_max = self.data_min + 1.0
        
        dims = self.image_array.shape
        self.slider_x.setMaximum(dims[0]-1)
        self.slider_y.setMaximum(dims[1]-1)
        self.slider_z.setMaximum(dims[2]-1)
        self.slider_x.setValue(dims[0]//2)
        self.slider_y.setValue(dims[1]//2)
        self.slider_z.setValue(dims[2]//2)

        self.color_bar.blockSignals(True)
        if self.is_ct:
            self.spin_level.setValue(-777)
            self.spin_width.setValue(495)
            self._apply_manual_windowing()
        else:
            self.color_bar.setLevels((self.data_min, self.data_max))
            
        self.color_bar.blockSignals(False)
        self.update_views()

    def register_to_patient(self):
        main_win = self.window()
        while main_win and not hasattr(main_win, 'image_array'):
            parent = main_win.parent()
            if parent is None: break
            main_win = parent
        
        if not hasattr(main_win, 'image_array') and self.parent() and hasattr(self.parent(), 'image_array'):
             main_win = self.parent()

        if not main_win or main_win.image_array is None:
            QMessageBox.warning(self, "Registration Error", "No patient data loaded in Main Window.")
            return

        try:
            ref_np_z_y_x = np.transpose(main_win.image_array, (2, 1, 0)).astype(np.float32)
            ref_image = sitk.GetImageFromArray(ref_np_z_y_x)
            
            if main_win.voxel_size_mm:
                ref_image.SetSpacing(main_win.voxel_size_mm)
            
            sz = ref_image.GetSize()
            sp = ref_image.GetSpacing()
            new_origin = [-(sz[i] * sp[i]) / 2.0 for i in range(3)]
            ref_image.SetOrigin(new_origin)
            
            moving_image = sitk.Cast(self.sitk_image, sitk.sitkFloat32)

            init_transform = sitk.CenteredTransformInitializer(
                ref_image, 
                moving_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_image)
            resampler.SetTransform(init_transform)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            
            self.sitk_image = resampler.Execute(moving_image)
            self._update_display_from_sitk()
            QMessageBox.information(self, "Success", "Map aligned to Patient Center and resampled.")
            
        except Exception as e:
            QMessageBox.critical(self, "Registration Failed", f"Error during registration: {e}")

    def update_views(self):
        if self.image_array is None: return

        x = self.slider_x.value()
        y = self.slider_y.value()
        z = self.slider_z.value()
        
        self.label_x.setText(f"X: {x}")
        self.label_y.setText(f"Y: {y}")
        self.label_z.setText(f"Z: {z}")

        levels = self.color_bar.levels()

        ax_slice = self.image_array[:, :, z]
        self.img_item_ax.setImage(ax_slice, levels=levels, lut=self.lut, axisOrder='col-major', autoLevels=False)
        self.view_axial.vLine.setPos(x)
        self.view_axial.hLine.setPos(y)

        cor_slice = self.image_array[:, y, :]
        self.img_item_cor.setImage(cor_slice, levels=levels, lut=self.lut, axisOrder='col-major', autoLevels=False)
        self.view_coronal.vLine.setPos(x)
        self.view_coronal.hLine.setPos(z)

        sag_slice = self.image_array[x, :, :]
        self.img_item_sag.setImage(sag_slice, levels=levels, lut=self.lut, axisOrder='col-major', autoLevels=False)
        self.view_sagittal.vLine.setPos(y)
        self.view_sagittal.hLine.setPos(z)

    def _emit_view_change(self):
        if self.image_array is not None:
             self.sigViewChanged.emit(self.slider_x.value(), self.slider_y.value(), self.slider_z.value())

    def set_view(self, x, y, z):
        if self.image_array is None: return
        self.slider_x.blockSignals(True)
        self.slider_y.blockSignals(True)
        self.slider_z.blockSignals(True)
        dims = self.image_array.shape
        self.slider_x.setValue(min(x, dims[0]-1))
        self.slider_y.setValue(min(y, dims[1]-1))
        self.slider_z.setValue(min(z, dims[2]-1))
        self.slider_x.blockSignals(False)
        self.slider_y.blockSignals(False)
        self.slider_z.blockSignals(False)
        self.update_views()

    def _on_mouse_move(self, pos, view_type):
        if self.image_array is None: return
        
        if view_type == 'axial':
            vb = self.view_axial.getPlotItem().getViewBox()
            img_shape = (self.image_array.shape[0], self.image_array.shape[1])
        elif view_type == 'coronal':
            vb = self.view_coronal.getPlotItem().getViewBox()
            img_shape = (self.image_array.shape[0], self.image_array.shape[2])
        else:
            vb = self.view_sagittal.getPlotItem().getViewBox()
            img_shape = (self.image_array.shape[1], self.image_array.shape[2])

        mouse_point = vb.mapSceneToView(pos)
        mx, my = int(mouse_point.x()), int(mouse_point.y())
        
        if 0 <= mx < img_shape[0] and 0 <= my < img_shape[1]:
            val = 0.0
            x, y, z = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
            if view_type == 'axial':
                val = self.image_array[mx, my, z]
            elif view_type == 'coronal':
                val = self.image_array[mx, y, my] 
            elif view_type == 'sagittal':
                val = self.image_array[x, mx, my] 
            self.value_label.setText(f"Value: {val:.2f}")
        else:
            self.value_label.setText("Value: N/A")


class OccurrenceMapDialog(QDialog):
    """A popup window to view CT and PET Occurrence Maps."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Occurrence Map Viewer")
        self.resize(1500, 900)
        self._apply_stylesheet()
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        
        self.ct_widget = OccurrenceViewerWidget("CT Occurrence", "gray", is_ct=True, parent=self)
        self.pet_widget = OccurrenceViewerWidget("PET Occurrence", "viridis", is_ct=False, parent=self)
        
        self.ct_widget.sigViewChanged.connect(self.pet_widget.set_view)
        self.pet_widget.sigViewChanged.connect(self.ct_widget.set_view)

        separator = QWidget()
        separator.setFixedWidth(2)
        separator.setStyleSheet("background-color: #4C566A;")

        content_layout.addWidget(self.ct_widget, 1)
        content_layout.addWidget(separator)
        content_layout.addWidget(self.pet_widget, 1)
        
        main_layout.addLayout(content_layout)
        
        footer_layout = QHBoxLayout()
        self.close_btn = QPushButton("Close")
        self.close_btn.setFixedWidth(100)
        self.close_btn.clicked.connect(self.accept)
        footer_layout.addStretch()
        footer_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(footer_layout)

    def _apply_stylesheet(self):
        stylesheet = """
            QDialog, QWidget { background-color: #2E3440; color: #D8DEE9; font-family: Arial, sans-serif; }
            QGroupBox { border: 1px solid #4C566A; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QLabel { font-size: 10pt; color: #D8DEE9; }
            QPushButton { background-color: #5E81AC; color: #ECEFF4; font-weight: bold; border-radius: 4px; padding: 8px; border: none; font-size: 10pt; }
            QPushButton:hover { background-color: #81A1C1; }
            QPushButton:disabled { background-color: #4C566A; color: #6a758c; }
            QSlider::groove:horizontal { border: 1px solid #4C566A; background: #434C5E; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #D8DEE9; border: 1px solid #4C566A; width: 18px; margin: -5px 0; border-radius: 9px; }
            QSpinBox { border: 1px solid #4C566A; border-radius: 4px; padding: 4px; background-color: #434C5E; color: #D8DEE9; min-width: 60px; }
        """
        self.setStyleSheet(stylesheet)


class PartialVolumeCorrectionDialog(QDialog):
    """Optional rim edge-sharpening settings for library lesion import."""

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lesion Pre-correction (Edge Sharpening)")
        self.setMinimumWidth(440)
        current_settings = current_settings or {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Optional edge sharpening applied to each lesion when you "
            "click <b>Import to Library</b>. Boosts partial-volume rim voxels only; "
            "interior intensities stay as measured (no core peak amplification). "
            "PET + mask only; CT not required. Each import keeps its own setting; "
            "changing options here affects <b>only the next import</b>."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.enable_checkbox = QCheckBox("Apply edge sharpening on import")
        self.enable_checkbox.setChecked(bool(current_settings.get("enabled", False)))

        saved_fwhm = normalize_psf_fwhm_mm(current_settings.get("fwhm_mm"))
        fwhm_row = QWidget()
        fwhm_layout = QHBoxLayout(fwhm_row)
        fwhm_layout.setContentsMargins(0, 0, 0, 0)
        fwhm_layout.setSpacing(6)
        self.fwhm_spins = []
        for axis, val in zip(("X", "Y", "Z"), saved_fwhm):
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 20.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setSuffix(" mm")
            spin.setValue(float(val))
            spin.setToolTip(f"Reconstruction PSF FWHM along array axis {axis}.")
            fwhm_layout.addWidget(spin)
            self.fwhm_spins.append(spin)
        fwhm_layout.addStretch(1)

        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(5, 95)
        self.strength_spin.setSuffix(" %")
        self.strength_spin.setValue(int(current_settings.get("sharpening_strength_percent", 75)))
        self.strength_spin.setToolTip(
            "Boundary sharpening strength. Only the outermost 1–2 voxel layer is "
            "corrected; all interior voxels stay unchanged. If edges look too bright, "
            "lower this value."
        )

        form.addRow("", self.enable_checkbox)
        form.addRow("PSF FWHM (X, Y, Z):", fwhm_row)
        form.addRow("Sharpening strength:", self.strength_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet("""
            QDialog, QWidget { background-color: #2E3440; color: #D8DEE9; }
            QDoubleSpinBox, QSpinBox {
                background-color: #434C5E; color: #D8DEE9; border: 1px solid #4C566A; padding: 4px;
            }
            QPushButton { background-color: #5E81AC; color: #ECEFF4; border-radius: 4px; padding: 6px; }
        """)

    def get_settings(self):
        return {
            "enabled": self.enable_checkbox.isChecked(),
            "fwhm_mm": tuple(float(sp.value()) for sp in self.fwhm_spins),
            "sharpening_strength_percent": int(self.strength_spin.value()),
        }


class RealLesionLibraryDialog(QDialog):
    """
    A dialog to select real lesions from a library.
    Features: Batch Import, Individual Remove, Scaling, Dark Table.
    """
    # Emits (Array, RegionName, Scale, SizeScale, UniformIntensity, UseReconSpacing, SourceSpacing)
    lesion_selected_signal = pyqtSignal(object, object, float, float, bool, bool, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real Lesion Library & Collection")
        self.resize(1600, 900)
        self._apply_stylesheet()
        
        self.setWindowModality(Qt.NonModal)

        self.pet_image_sitk = None
        self.mask_image_sitk = None
        self.lesion_stats = None
        self.current_preview_array = None
        
        self.current_pet_path = None
        self.current_mask_path = None
        self.regions = ["Lung", "Liver", "Bone", "Soft Tissue"]
        
        # List to store collected lesions
        self.lesion_collection = []
        self._preview_source_array = None
        self._preview_record = None
        self.current_display_data = []
        self.pvc_settings = {
            "enabled": False,
            "fwhm_mm": DEFAULT_PSF_FWHM_MM,
            "sharpening_strength_percent": 75,
        }

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)

        # 0. Library Save/Load (Configuration)
        lib_config_group = QGroupBox("Configuration")
        lib_config_layout = QHBoxLayout(lib_config_group)
        self.btn_load_lib_conf = QPushButton("Load Library")
        self.btn_save_lib_conf = QPushButton("Save Library")
        lib_config_layout.addWidget(self.btn_load_lib_conf)
        lib_config_layout.addWidget(self.btn_save_lib_conf)
        left_layout.addWidget(lib_config_group)

        # 1. File Loading
        file_group = QGroupBox("1. Load Data")
        file_layout = QFormLayout(file_group)
        
        self.btn_load_pet = QPushButton("PET/SPECT")
        self.lbl_pet_status = QLabel("Not Loaded")
        
        self.btn_load_mask = QPushButton("Single Mask")
        self.lbl_mask_status = QLabel("Not Loaded")
        
        self.btn_batch_mask = QPushButton("Multiple Masks")
        self.lbl_batch_status = QLabel("Not Loaded")
        
        file_layout.addRow(self.lbl_pet_status, self.btn_load_pet)
        file_layout.addRow(self.lbl_mask_status, self.btn_load_mask)
        file_layout.addRow(self.lbl_batch_status, self.btn_batch_mask)

        self.lbl_pvc_status = QLabel("Off")
        self.btn_pvc = QPushButton("Lesion Pre-correction")
        self.btn_pvc.setToolTip(
            "Optional edge sharpening applied per lesion when importing to the library."
        )
        file_layout.addRow(self.lbl_pvc_status, self.btn_pvc)
        left_layout.addWidget(file_group)

        # 2. Region Assignment & Import
        import_group = QGroupBox("2. Anatomical Region")
        import_layout = QVBoxLayout(import_group)
        
        self.region_combo = QComboBox()
        self.region_combo.addItem("-- Select Anatomical Region --")
        self.region_combo.addItems(self.regions)
        self.region_combo.setEnabled(False)
        
        self.btn_import = QPushButton("Import to Library")
        self.btn_import.setStyleSheet("background-color: #88C0D0; color: #2E3440; font-weight: bold; padding: 6px;")
        self.btn_import.setEnabled(False)
        
        import_layout.addWidget(self.region_combo)
        import_layout.addWidget(self.btn_import)
        left_layout.addWidget(import_group)

        # 3. Collection (The main list)
        coll_group = QGroupBox("Lesion Library")
        coll_layout = QVBoxLayout(coll_group)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by Region:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Show All")
        self.filter_combo.addItems(self.regions)
        filter_layout.addWidget(self.filter_combo)
        coll_layout.addLayout(filter_layout)
        
        self.collection_list = QListWidget()
        coll_layout.addWidget(self.collection_list)
        
        # Removal Buttons
        btn_layout = QHBoxLayout()
        self.btn_remove_item = QPushButton("Remove Selected")
        self.btn_remove_item.clicked.connect(self.remove_selected_item)
        
        self.btn_clear_collection = QPushButton("Clear All")
        self.btn_clear_collection.setStyleSheet("background-color: #BF616A; color: white;")
        
        btn_layout.addWidget(self.btn_remove_item)
        btn_layout.addWidget(self.btn_clear_collection)
        coll_layout.addLayout(btn_layout)
        
        left_layout.addWidget(coll_group)

        insert_group = QGroupBox("Insertion Parameters")
        insert_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #4C566A; border-radius: 6px; margin-top: 10px; }")
        insert_layout = QGridLayout(insert_group)
        insert_layout.setSpacing(10)

        # Intensity Scaling
        lbl_intensity = QLabel("Intensity Scale:")
        self.lbl_intensity = lbl_intensity
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSuffix(" x")
        self.scale_spin.setDecimals(2)
        self.scale_spin.setStyleSheet("padding: 5px;")

        # Size Scaling
        lbl_size = QLabel("Size Scale:")
        self.size_scale_spin = QDoubleSpinBox()
        self.size_scale_spin.setRange(0.1, 10.0)
        self.size_scale_spin.setSingleStep(0.1)
        self.size_scale_spin.setValue(1.0)
        self.size_scale_spin.setSuffix(" x")
        self.size_scale_spin.setDecimals(2)
        self.size_scale_spin.setStyleSheet("padding: 5px;")

        self.uniform_intensity_checkbox = QCheckBox("Uniform intensity (shape only)")
        self.uniform_intensity_checkbox.setToolTip(
            "Keep the library lesion shape but fill the mask with a fixed intensity, "
            "like a sphere. Use this to isolate shape effects in recovery analysis."
        )
        self.uniform_intensity_checkbox.stateChanged.connect(self._update_uniform_intensity_label)
        self.uniform_intensity_checkbox.stateChanged.connect(self._refresh_library_preview)
        self.size_scale_spin.valueChanged.connect(self._refresh_library_preview)
        self.scale_spin.valueChanged.connect(self._refresh_library_preview)

        self.match_recon_voxel_checkbox = QCheckBox("Use recon voxel size (Tab 2)")
        self.match_recon_voxel_checkbox.setToolTip(
            "When checked, this lesion uses Tab 2 object-grid spacing for volume in Tab 3/4 "
            "and in the radiomics table below. The patch is not resampled."
        )
        self.match_recon_voxel_checkbox.stateChanged.connect(self._on_match_recon_voxel_changed)
        self._update_match_recon_voxel_checkbox_state()

        # Insert Button
        self.btn_insert_lesion = QPushButton("Insert Lesion")
        self.btn_insert_lesion.setStyleSheet("background-color: #A3BE8C; color: #2E3440; font-weight: bold; padding: 12px; font-size: 12pt; border-radius: 6px;")
        self.btn_insert_lesion.setEnabled(False)

        # Layout Arrangement
        insert_layout.addWidget(lbl_intensity, 0, 0)
        insert_layout.addWidget(self.scale_spin, 0, 1)
        insert_layout.addWidget(lbl_size, 1, 0)
        insert_layout.addWidget(self.size_scale_spin, 1, 1)
        insert_layout.addWidget(self.uniform_intensity_checkbox, 2, 0, 1, 2)
        insert_layout.addWidget(self.match_recon_voxel_checkbox, 3, 0, 1, 2)
        insert_layout.addWidget(self.btn_insert_lesion, 4, 0, 1, 2)
        insert_layout.setColumnStretch(1, 1)
        
        left_layout.addWidget(insert_group)
        main_layout.addWidget(left_panel)

        # --- Right Panel: Previewer & Radiomics ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.preview_widget = OccurrenceViewerWidget("Lesion Preview", "hot", is_ct=False)
        self.preview_widget.register_btn.hide()
        self.preview_widget.load_btn.hide()
        
        title = "Radiomic Features"
        self.stats_group = QGroupBox(title)
        self.stats_layout = QVBoxLayout(self.stats_group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Feature", "Value"])
        
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.stats_table.verticalHeader().setVisible(False)
        
        self.stats_table.setAlternatingRowColors(False)
        self.stats_table.setStyleSheet("QTableWidget { background-color: #3B4252; color: #ECEFF4; gridline-color: #4C566A; } QHeaderView::section { background-color: #4C566A; color: #ECEFF4; }")
        
        self.stats_layout.addWidget(self.stats_table)

        right_layout.addWidget(self.preview_widget, 6)
        right_layout.addWidget(self.stats_group, 4)
        main_layout.addWidget(right_panel)

        # Connections
        self.btn_load_pet.clicked.connect(self.load_pet)
        self.btn_load_mask.clicked.connect(self.load_mask)
        self.btn_batch_mask.clicked.connect(self.batch_load_masks)
        self.btn_pvc.clicked.connect(self._open_pvc_dialog)
        
        self.region_combo.currentTextChanged.connect(self._check_ready)
        self.btn_import.clicked.connect(self.import_lesions_from_mask)
        
        self.filter_combo.currentTextChanged.connect(self.refresh_collection_list)
        self.collection_list.itemClicked.connect(self.preview_collection_item)
        
        self.btn_insert_lesion.clicked.connect(self.insert_selected_lesion)
        self.btn_clear_collection.clicked.connect(self.clear_collection)
        
        self.btn_save_lib_conf.clicked.connect(self.save_library_config)
        self.btn_load_lib_conf.clicked.connect(self.load_library_config)

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QDialog, QWidget { background-color: #2E3440; color: #D8DEE9; }
            QGroupBox { border: 1px solid #4C566A; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QListWidget, QComboBox, QTableWidget { background-color: #434C5E; border: 1px solid #4C566A; border-radius: 4px; padding: 5px; color: #D8DEE9; font-size: 10pt; }
            QPushButton { background-color: #5E81AC; color: #ECEFF4; border-radius: 4px; padding: 6px; border: none; }
            QPushButton:hover { background-color: #81A1C1; }
            QPushButton:disabled { background-color: #4C566A; color: #6a758c; }
            QHeaderView::section { background-color: #4C566A; padding: 4px; border: 1px solid #2E3440; font-weight: bold; }
            QDoubleSpinBox { background-color: #434C5E; color: #D8DEE9; border: 1px solid #4C566A; padding: 4px; }
        """)

    def save_library_config(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Library", "lesion_library.json", "JSON Files (*.json)")
        if path:
            serialized_collection = []
            for record in self.lesion_collection:
                rec_copy = record.copy()
                for key in ("array", "array_raw"):
                    if key not in record:
                        continue
                    arr = np.asarray(record[key])
                    rec_copy[f"{key}_b64"] = base64.b64encode(arr.tobytes()).decode("utf-8")
                    rec_copy[f"{key}_dtype"] = str(arr.dtype)
                    rec_copy[f"{key}_shape"] = arr.shape
                    rec_copy.pop(key, None)
                serialized_collection.append(rec_copy)

            data = {
                "collection": serialized_collection,
                "note": "Complete lesion library with arrays and features."
            }
            try:
                with open(path, 'w') as f: json.dump(data, f, indent=4)
                QMessageBox.information(self, "Success", f"Library with {len(serialized_collection)} lesions saved.")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    @staticmethod
    def _decode_library_array(rec, prefix):
        """
        Decode a numpy volume from saved library JSON.

        Supports current keys ({prefix}_b64, {prefix}_dtype, {prefix}_shape) and
        legacy lesion files that used array_b64 + dtype + shape for the main array.
        """
        dtype_key = f"{prefix}_dtype"
        shape_key = f"{prefix}_shape"
        b64_key = f"{prefix}_b64"

        if b64_key in rec and dtype_key in rec and shape_key in rec:
            pass
        elif prefix == "array" and "array_b64" in rec and "dtype" in rec and "shape" in rec:
            b64_key = "array_b64"
            dtype_key = "dtype"
            shape_key = "shape"
        else:
            return None

        dtype = np.dtype(rec[dtype_key])
        shape = tuple(int(s) for s in rec[shape_key])
        arr_bytes = base64.b64decode(rec[b64_key])
        return np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)

    @staticmethod
    def _strip_library_array_keys(rec):
        for key in (
            "array_b64", "array_dtype", "array_shape",
            "array_raw_b64", "array_raw_dtype", "array_raw_shape",
            "dtype", "shape",
        ):
            rec.pop(key, None)

    def load_library_config(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Load Library", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f: data = json.load(f)
                
                if "collection" in data:
                    loaded_list = data["collection"]
                    loaded_count = 0
                    for rec in loaded_list:
                         new_rec = rec.copy()
                         arr = self._decode_library_array(rec, "array")
                         if arr is None:
                             continue
                         new_rec["array"] = arr
                         raw = self._decode_library_array(rec, "array_raw")
                         if raw is not None:
                             new_rec["array_raw"] = raw
                         self._strip_library_array_keys(new_rec)

                         if not any(r['id'] == new_rec['id'] for r in self.lesion_collection):
                             self.lesion_collection.append(new_rec)
                             loaded_count += 1
                    
                    self.refresh_collection_list()
                    QMessageBox.information(
                        self, "Success",
                        f"Loaded {loaded_count} lesion(s) from {len(loaded_list)} record(s).",
                    )
                else:
                    QMessageBox.warning(self, "Format Error", "JSON does not contain a 'collection' key.")

            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def load_pet(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select Library PET", "", "NIfTI Files (*.nii *.nii.gz)")
        if path:
            if os.path.basename(path).startswith("._"): QMessageBox.warning(self, "Invalid File", "MacOS metadata file detected."); return
            try:
                self.current_pet_path = path
                self.pet_image_sitk = sitk.ReadImage(path)
                self.lbl_pet_status.setText("Loaded")
                self.lbl_pet_status.setStyleSheet("color: #A3BE8C; font-weight: bold;")
                self._check_ready()
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load PET: {e}")

    def load_mask(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select Library Mask", "", "NIfTI Files (*.nii *.nii.gz)")
        if path:
            self._handle_mask_load(path, update_ui=True)

    def batch_load_masks(self):
        from PyQt5.QtWidgets import QFileDialog
        paths, _ = QFileDialog.getOpenFileNames(self, "Batch Select Masks", "", "NIfTI Files (*.nii *.nii.gz)")
        if not paths: return
        
        if self.pet_image_sitk is None:
            QMessageBox.warning(self, "No PET", "Please load a Reference PET file first.")
            return
        
        self.batch_masks_queue = paths
        self.lbl_batch_status.setText("Loaded")
        self.lbl_batch_status.setStyleSheet("color: #A3BE8C; font-weight: bold;")
        
        self.region_combo.setEnabled(True)
        self.btn_import.setEnabled(False) 

    def _handle_mask_load(self, path, update_ui=True):
        if os.path.basename(path).startswith("._"): return
        try:
            self.current_mask_path = path
            self.mask_image_sitk = sitk.ReadImage(path)
            
            if update_ui:
                self.lbl_mask_status.setText("Loaded")
                self.lbl_mask_status.setStyleSheet("color: #A3BE8C; font-weight: bold;")
                self.batch_masks_queue = None 
            
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(self.mask_image_sitk)
            self.lesion_stats = stats
            
            if update_ui:
                self._check_ready()
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load Mask: {e}")

    def _check_ready(self):
        ready_single = (self.pet_image_sitk is not None) and (self.mask_image_sitk is not None)
        # Ensure batch_masks_queue is evaluated as boolean to avoid NoneType error.
        ready_batch = (self.pet_image_sitk is not None) and hasattr(self, 'batch_masks_queue') and bool(self.batch_masks_queue)
        
        region_selected = self.region_combo.currentText() != "-- Select Anatomical Region --"
        self.region_combo.setEnabled(self.pet_image_sitk is not None)
        self.btn_import.setEnabled((ready_single or ready_batch) and region_selected)

    def import_lesions_from_mask(self):
        region = self.region_combo.currentText()
        if region == "-- Select Anatomical Region --": return

        if hasattr(self, 'batch_masks_queue') and self.batch_masks_queue:
            total_imported = 0
            for path in self.batch_masks_queue:
                try:
                    self._handle_mask_load(path, update_ui=False) 
                    count = self._process_current_mask_labels(region, path)
                    total_imported += count
                except: continue
            self.batch_masks_queue = None
            
            self.lbl_batch_status.setText("Not Loaded")
            self.lbl_batch_status.setStyleSheet("color: #D8DEE9;") 
            
            self.lbl_mask_status.setText("Not Loaded")
            self.lbl_mask_status.setStyleSheet("color: #D8DEE9;")
            self.mask_image_sitk = None
            self.lesion_stats = None
            
            QMessageBox.information(self, "Batch Import", f"Imported {total_imported} lesions into {region}.")
        else:
            if not self.lesion_stats: return
            count = self._process_current_mask_labels(region, self.current_mask_path)
            
            self.lbl_mask_status.setText("Not Loaded")
            self.lbl_mask_status.setStyleSheet("color: #D8DEE9;")
            self.mask_image_sitk = None
            self.lesion_stats = None
            
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} lesions as '{region}'.")
        
        self.refresh_collection_list()
        self.region_combo.setCurrentIndex(0)
        self._check_ready() # Ensure buttons disable

    def _process_current_mask_labels(self, region, source_name):
        labels = self.lesion_stats.GetLabels()
        if not labels: return 0
        
        count = 0
        for lid in labels:
            vol = self.lesion_stats.GetPhysicalSize(lid)
            
            bbox = self.lesion_stats.GetBoundingBox(lid)
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetIndex(bbox[:3])
            roi_filter.SetSize(bbox[3:])
            
            pet_crop = roi_filter.Execute(self.pet_image_sitk)
            mask_crop = roi_filter.Execute(self.mask_image_sitk)
            
            binary_mask = sitk.Equal(mask_crop, lid)
            pet_masked = sitk.Mask(pet_crop, binary_mask)
            
            arr = sitk.GetArrayFromImage(pet_masked)
            arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)

            mask_arr = sitk.GetArrayFromImage(binary_mask)
            mask_arr = np.transpose(mask_arr, (2, 1, 0)) > 0

            source_spacing_mm = None
            if self.pet_image_sitk is not None:
                try:
                    source_spacing_mm = tuple(float(s) for s in self.pet_image_sitk.GetSpacing())
                except (TypeError, ValueError, RuntimeError):
                    source_spacing_mm = None

            measured_crop = arr.copy()

            arr = self._apply_precorrect_to_array(arr, source_spacing_mm)
            fwhm_mm = normalize_psf_fwhm_mm(self.pvc_settings.get("fwhm_mm"))
            if self.pvc_settings.get("enabled"):
                roundtrip_nmse = wiener_deconv_roundtrip_nmse(
                    arr, measured_crop, mask_arr, source_spacing_mm, fwhm_mm,
                )
            else:
                roundtrip_nmse = None
            
            lesion_id = str(uuid.uuid4())[:8]
            
            existing_count = sum(1 for r in self.lesion_collection if r['region'] == region)
            unique_num = existing_count + 1
            name = f"Lesion {unique_num} ({region})"
            
            vals = arr[arr > 0]
            features = {}
            if self.pvc_settings.get("enabled"):
                features["precorrect_applied"] = True
                features["precorrect_method"] = "edge_rim"
                features["precorrect_fwhm_mm"] = list(fwhm_mm)
                features["precorrect_strength_percent"] = int(
                    self.pvc_settings.get("sharpening_strength_percent", 50)
                )
                if roundtrip_nmse is not None and np.isfinite(roundtrip_nmse):
                    features["precorrect_roundtrip_nmse"] = float(roundtrip_nmse)
            if vals.size > 0:
                features['volume'] = vol
                features['min'] = float(np.min(vals))
                features['max'] = float(np.max(vals))
                features['mean'] = float(np.mean(vals))
                features['std'] = float(np.std(vals))
                hist, _ = np.histogram(vals, bins=20, density=True)
                features['entropy_hist'] = scipy.stats.entropy(hist)
            
            record = {
                'id': lesion_id,
                'name': name,
                'array_raw': measured_crop.copy(),
                'array': np.asarray(arr, dtype=np.float32).copy(),
                'region': region,
                'vol': vol,
                'source_spacing_mm': source_spacing_mm,
                'lid': lid,
                'source_file': os.path.basename(source_name),
                'features': features
            }
            self.lesion_collection.append(record)
            count += 1
        return count

    def _open_pvc_dialog(self):
        dlg = PartialVolumeCorrectionDialog(self.pvc_settings, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.pvc_settings = dlg.get_settings()
            self._update_pvc_status_label()

    def _apply_precorrect_to_array(self, arr, spacing_mm):
        arr = np.asarray(arr, dtype=np.float32)
        mask = arr > 0
        if not self.pvc_settings.get("enabled") or not np.any(mask):
            return arr.copy()
        fwhm_mm = normalize_psf_fwhm_mm(self.pvc_settings.get("fwhm_mm"))
        strength = int(self.pvc_settings.get("sharpening_strength_percent", 50))
        out = apply_lesion_precorrection(
            arr, mask, spacing_mm, fwhm_mm, sharpening_strength_percent=strength,
        )
        return np.where(mask, out, 0.0).astype(np.float32)

    def _stored_lesion_array(self, record):
        """Return the lesion template frozen at import time (per-lesion correction)."""
        return np.asarray(record["array"], dtype=np.float32).copy()

    @staticmethod
    def _format_psf_fwhm(fwhm_mm):
        fx, fy, fz = normalize_psf_fwhm_mm(fwhm_mm)
        if abs(fx - fy) < 1e-3 and abs(fy - fz) < 1e-3:
            return f"{fx:.1f} mm"
        return f"{fx:.1f}×{fy:.1f}×{fz:.1f} mm"

    def _update_pvc_status_label(self):
        if not self.pvc_settings.get("enabled"):
            self.lbl_pvc_status.setText("Off")
            self.lbl_pvc_status.setStyleSheet("color: #D8DEE9;")
            return
        fwhm = self._format_psf_fwhm(self.pvc_settings.get("fwhm_mm"))
        strength = int(self.pvc_settings.get("sharpening_strength_percent", 50))
        self.lbl_pvc_status.setText(f"On (Edge, {fwhm}, {strength}%)")
        self.lbl_pvc_status.setStyleSheet("color: #A3BE8C; font-weight: bold;")

    @staticmethod
    def _format_voxel_spacing_mm(spacing):
        if spacing is None:
            return "—"
        try:
            parts = [float(s) for s in spacing[:3]]
        except (TypeError, ValueError, IndexError):
            return "—"
        if len(parts) < 3:
            return "—"
        return f"{parts[0]:.2f}×{parts[1]:.2f}×{parts[2]:.2f}"

    def _recon_spacing_mm(self):
        """Object-grid voxel size from Tab 2 (post-recon or Object Voxel Size field)."""
        main_win = self.parent()
        while main_win is not None and not hasattr(main_win, "get_effective_voxel_spacing_mm"):
            main_win = main_win.parent()
        if main_win is None:
            return None
        fn = getattr(main_win, "get_effective_voxel_spacing_mm", None)
        if not callable(fn):
            return None
        try:
            spacing = fn()
            if spacing is None:
                return None
            return tuple(float(s) for s in spacing[:3])
        except (TypeError, ValueError, IndexError):
            return None

    def _update_match_recon_voxel_checkbox_state(self):
        if not hasattr(self, "match_recon_voxel_checkbox"):
            return
        spacing = self._recon_spacing_mm()
        available = spacing is not None
        self.match_recon_voxel_checkbox.setEnabled(available)
        if not available:
            self.match_recon_voxel_checkbox.setChecked(False)
            self.match_recon_voxel_checkbox.setToolTip(
                "Set Object Voxel Size on Tab 2 or run initial reconstruction first."
            )
        else:
            self.match_recon_voxel_checkbox.setToolTip(
                f"Use Tab 2 spacing ({self._format_voxel_spacing_mm(spacing)}) for volume in "
                "Tab 3/4 and radiomics. The lesion patch is not resampled."
            )

    def _display_voxel_spacing_mm(self, record):
        """Spacing used for radiomics / volume rows in this dialog."""
        if (
            hasattr(self, "match_recon_voxel_checkbox")
            and self.match_recon_voxel_checkbox.isChecked()
        ):
            recon = self._recon_spacing_mm()
            if recon is not None:
                return recon
        if record is not None:
            return record.get("source_spacing_mm")
        return None

    def _display_volume_mm3(self, record, lesion_array):
        """Volume (mm³) shown in radiomics, consistent with selected spacing."""
        num_voxels = int(np.count_nonzero(np.asarray(lesion_array) > 0))
        if num_voxels <= 0:
            return 0.0
        spacing = self._display_voxel_spacing_mm(record)
        if spacing is None:
            return float(record.get("vol", 0.0)) if record else 0.0
        return float(num_voxels * spacing[0] * spacing[1] * spacing[2])

    def _on_match_recon_voxel_changed(self, _state=None):
        self._update_match_recon_voxel_checkbox_state()
        self._refresh_library_radiomics_if_needed()

    def _refresh_library_radiomics_if_needed(self):
        if self._preview_record is None or self.current_preview_array is None:
            return
        self._calculate_radiomics(
            self._preview_record.get("vol", 0.0),
            self.current_preview_array,
            self._preview_record,
        )

    def refresh_collection_list(self):
        self.collection_list.clear()
        filter_region = self.filter_combo.currentText()
        for record in self.lesion_collection:
            if filter_region == "Show All" or record['region'] == filter_region:
                disp_text = f"{record['name']}"
                item = QListWidgetItem(disp_text)
                item.setData(Qt.UserRole, record['id'])
                self.collection_list.addItem(item)

    def _refresh_library_preview(self, *_args):
        """Preview: cubic intensities clipped to NN morphology; uniform flattens."""
        if self._preview_source_array is None:
            return
        size_scale = float(self.size_scale_spin.value())
        source = np.asarray(self._preview_source_array, dtype=np.float32)
        support = library_morphology_support_from_shape(source, size_scale)
        if abs(size_scale - 1.0) > 1e-6:
            arr = resize_volume_preserve_morphology(source, size_scale)
        else:
            arr = source.copy()
        arr = np.where(support, arr, 0.0).astype(np.float32)
        if self.uniform_intensity_checkbox.isChecked():
            intensity = float(self.scale_spin.value())
            arr = uniform_library_patch_from_resized(arr, intensity, support=support)
        self.current_preview_array = arr
        self._update_preview_display(arr)

    def preview_collection_item(self, item):
        lesion_id = item.data(Qt.UserRole)
        record = next((r for r in self.lesion_collection if r['id'] == lesion_id), None)
        if not record:
            return

        self._update_match_recon_voxel_checkbox_state()
        self._preview_record = record
        self._preview_source_array = self._stored_lesion_array(record)
        self._refresh_library_preview()
        
        self._calculate_radiomics(record['vol'], self.current_preview_array, record)
        
        self.btn_insert_lesion.setEnabled(True)

    def showEvent(self, event):
        """Refresh recon spacing availability when the dialog is opened."""
        super().showEvent(event)
        self._update_match_recon_voxel_checkbox_state()

    def remove_selected_item(self):
        row = self.collection_list.currentRow()
        item = self.collection_list.currentItem()
        if not item:
            return
        
        lesion_id = item.data(Qt.UserRole)
        self.lesion_collection = [r for r in self.lesion_collection if r['id'] != lesion_id]
        self.collection_list.takeItem(row)
        
        if self.collection_list.count() == 0:
            self.stats_table.setRowCount(0)
            self.preview_widget.image_array = None
            self.current_preview_array = None
            self._preview_record = None
            self.preview_widget.update_views()
            self.btn_insert_lesion.setEnabled(False)

    def _update_preview_display(self, arr):
        # 1. Update data
        self.preview_widget.image_array = arr
        dims = arr.shape
        
        # 2. Block signals on the viewer sliders so they don't trigger 
        #    update_views() while we are resizing them.
        self.preview_widget.slider_x.blockSignals(True)
        self.preview_widget.slider_y.blockSignals(True)
        self.preview_widget.slider_z.blockSignals(True)
        
        try:
            # 3. Safe to resize now
            self.preview_widget.slider_x.setMaximum(dims[0]-1)
            self.preview_widget.slider_y.setMaximum(dims[1]-1)
            self.preview_widget.slider_z.setMaximum(dims[2]-1)
            
            self.preview_widget.slider_x.setValue(dims[0]//2)
            self.preview_widget.slider_y.setValue(dims[1]//2)
            self.preview_widget.slider_z.setValue(dims[2]//2)
            
            # Update range for colorbar
            vals = arr[arr > 0]
            if vals.size > 0:
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))
            else:
                min_val, max_val = 0.0, 1.0
            # Uniform-intensity lesions are flat; window from 0 so they are visible.
            if min_val == max_val:
                min_val = 0.0
                max_val = max(max_val * 1.1, max_val + 1e-3, 1.0)
            self.preview_widget.color_bar.setLevels((min_val, max_val))
            
        finally:
            # 4. Unblock and trigger single update
            self.preview_widget.slider_x.blockSignals(False)
            self.preview_widget.slider_y.blockSignals(False)
            self.preview_widget.slider_z.blockSignals(False)
            
        self.preview_widget.update_views()

    def _calculate_radiomics(self, volume, lesion_array, record=None):
        """
        Compute intensity and shape/texture radiomics features for a lesion array.

        Voxel spacing is reconstructed from the supplied physical volume, the mask
        is zero-padded to keep boundary voxels inside the analysis region, and
        Compactness 2 is computed manually when PyRadiomics does not provide it.
        """
        self.stats_table.setRowCount(0)
        display_data = []

        # Helper for 2-decimal formatting
        def fmt(val):
            if val is None: return "N/A"
            try:
                f_val = float(val)
                # Filter out pure zeros that imply missing features
                if f_val == 0.0 and val != 0: 
                     return "N/A"
                return f"{f_val:.2f}"
            except (ValueError, TypeError):
                return str(val)

        # 1. Basic Stats
        lesion_voxels = lesion_array[lesion_array > 0]
        num_voxels = lesion_voxels.size

        if num_voxels > 0:
            display_vol_mm3 = (
                self._display_volume_mm3(record, lesion_array)
                if record is not None
                else float(volume)
            )
            display_spacing = (
                self._display_voxel_spacing_mm(record) if record is not None else None
            )

            # --- Intensity Metrics ---
            display_data.append(("--- Intensity Metrics ---", "", True))
            display_data.append(("SUV Min", np.min(lesion_voxels), False))
            display_data.append(("SUV Mean", np.mean(lesion_voxels), False))
            display_data.append(("SUV Max", np.max(lesion_voxels), False))
            display_data.append(("Std Deviation", np.std(lesion_voxels), False))
            
            # 2. PyRadiomics Advanced Extraction
            if HAS_PYRADIOMICS:
                try:
                    # Physical spacing for shape features (explicit or estimated).
                    if display_spacing is not None:
                        spacing = tuple(float(s) for s in display_spacing[:3])
                    elif display_vol_mm3 > 0 and num_voxels > 0:
                        voxel_vol = display_vol_mm3 / num_voxels
                        est_spacing = voxel_vol ** (1.0 / 3.0)
                        spacing = (est_spacing, est_spacing, est_spacing)
                    elif volume > 0 and num_voxels > 0:
                        voxel_vol = volume / num_voxels
                        est_spacing = voxel_vol ** (1.0 / 3.0)
                        spacing = (est_spacing, est_spacing, est_spacing)
                    else:
                        spacing = (1.0, 1.0, 1.0)

                    # Pad with zeros so that lesions touching the volume boundary still
                    # have a full 3D neighbourhood for shape feature extraction.
                    padded_array = np.pad(lesion_array, pad_width=1, mode='constant', constant_values=0)
                    
                    im = sitk.GetImageFromArray(padded_array)
                    im.SetSpacing(spacing)
                    
                    ma = sitk.GetImageFromArray((padded_array > 0).astype(np.int32))
                    ma.SetSpacing(spacing)

                    settings = {
                        'binWidth': 0.25, 
                        'resampledPixelSpacing': None, 
                        'interpolator': sitk.sitkLinear,
                        'geometryTolerance': 1000,
                        'force2D': False 
                    }
                    
                    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
                    extractor.enableAllFeatures()
                    
                    features = extractor.execute(im, ma)
                    
                    # --- First Order ---
                    display_data.append(("--- First Order Statistics ---", "", True))
                    display_data.append(("Entropy (Hist)", features.get('original_firstorder_Entropy'), False))
                    display_data.append(("Uniformity", features.get('original_firstorder_Uniformity'), False))
                    display_data.append(("Skewness", features.get('original_firstorder_Skewness'), False))
                    display_data.append(("Kurtosis", features.get('original_firstorder_Kurtosis'), False))
                    
                    # --- Shape Features ---
                    display_data.append(("--- Shape Features ---", "", True))
                    display_data.append(("Volume (mL)", display_vol_mm3 / 1000.0, False)) 
                    display_data.append(("Voxel Count", num_voxels, False))
                    if record is not None:
                        display_data.append((
                            "Voxel size (mm)",
                            self._format_voxel_spacing_mm(self._display_voxel_spacing_mm(record)),
                            False,
                        ))
                    
                    # Retrieve Shape Metrics
                    mesh_vol = features.get('original_shape_MeshVolume')
                    surf_area = features.get('original_shape_SurfaceArea')
                    max_3d = features.get('original_shape_Maximum3DDiameter')
                    sphere = features.get('original_shape_Sphericity')
                    
                    # PyRadiomics sometimes returns a missing / zero Compactness 2;
                    # fall back to the analytical definition: 36 * pi * V^2 / A^3.
                    comp2 = features.get('original_shape_Compactness2')
                    if (comp2 is None or comp2 == 0) and (mesh_vol and surf_area and surf_area > 0):
                        try:
                            comp2 = (36.0 * np.pi * (mesh_vol ** 2)) / (surf_area ** 3)
                        except:
                            comp2 = None

                    display_data.append(("Compactness 2", comp2, False))
                    display_data.append(("Max 3D Diameter", max_3d, False))
                    display_data.append(("Sphericity", sphere, False))
                    display_data.append(("Surface Area", surf_area, False))
                    
                    # --- Texture ---
                    display_data.append(("--- GLCM (Texture) ---", "", True))
                    display_data.append(("GLCM Entropy", features.get('original_glcm_JointEntropy'), False))
                    display_data.append(("Contrast", features.get('original_glcm_Contrast'), False))
                    display_data.append(("Homogeneity (Idm)", features.get('original_glcm_Idm'), False))
                    display_data.append(("Correlation", features.get('original_glcm_Correlation'), False))
                    
                    display_data.append(("--- Heterogeneity ---", "", True))
                    display_data.append(("Run Entropy", features.get('original_glrlm_RunEntropy'), False))
                    display_data.append(("Zone Entropy", features.get('original_glszm_ZoneEntropy'), False))
                    display_data.append(("Zone Percentage", features.get('original_glszm_ZonePercentage'), False))
                    
                except Exception as e:
                    display_data.append(("PyRadiomics Error", str(e), True))
            else:
                display_data.append(("--- Shape Features ---", "", True))
                display_data.append(("Volume (mL)", display_vol_mm3 / 1000.0, False)) 
                display_data.append(("Voxel Count", num_voxels, False))
                if record is not None:
                    display_data.append((
                        "Voxel size (mm)",
                        self._format_voxel_spacing_mm(self._display_voxel_spacing_mm(record)),
                        False,
                    ))
                display_data.append(("Note", "PyRadiomics not installed.", True))
            
        else:
            display_data.append(("Status", "Lesion is empty", True))

        # Re-populate table
        self.stats_table.setRowCount(len(display_data))
        for row, (name, value, is_bold) in enumerate(display_data):
            val_str = fmt(value)
            name_item = QTableWidgetItem(name)
            val_item = QTableWidgetItem(val_str)
            
            if is_bold or name.startswith("---"):
                font = QFont(); font.setBold(True)
                name_item.setFont(font); name_item.setBackground(QColor("#4C566A"))
                val_item.setBackground(QColor("#4C566A"))
            
            self.stats_table.setItem(row, 0, name_item)
            self.stats_table.setItem(row, 1, val_item)

    def _update_uniform_intensity_label(self, _state=None):
        if self.uniform_intensity_checkbox.isChecked():
            self.lbl_intensity.setText("Intensity:")
        else:
            self.lbl_intensity.setText("Intensity Scale:")

    def insert_selected_lesion(self):
        if self._preview_source_array is None:
            return
        item = self.collection_list.currentItem()
        if not item:
            return
        lesion_id = item.data(Qt.UserRole)
        record = next((r for r in self.lesion_collection if r['id'] == lesion_id), None)
        region_name = record['region'] if record else "Unknown"
        
        scale_factor = self.scale_spin.value()
        size_scale = self.size_scale_spin.value()
        uniform_intensity = self.uniform_intensity_checkbox.isChecked()
        use_recon = self.match_recon_voxel_checkbox.isChecked()
        source_spacing = record.get("source_spacing_mm") if record else None
        
        # Emit unscaled import template; size_scale applied once in utils on save/recon.
        self.lesion_selected_signal.emit(
            np.asarray(self._preview_source_array, dtype=np.float32).copy(),
            region_name,
            scale_factor,
            size_scale,
            uniform_intensity,
            use_recon,
            source_spacing,
        )

    def clear_collection(self):
        self.lesion_collection = []
        self._preview_source_array = None
        self._preview_record = None
        self.collection_list.clear()
        self.preview_widget.image_array = None
        self.current_preview_array = None
        self.preview_widget.update_views()


class NoiseRoiPickerDialog(QDialog):
    """Place the global liver noise sphere; only mm center/radius are saved."""

    def __init__(
        self,
        template_volume=None,
        template_label="Placement template",
        template_choices=None,
        initial_template_key=None,
        voxel_spacing=(1.0, 1.0, 1.0),
        initial_center_mm=None,
        initial_radius_mm=14.0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Define Global Liver Noise ROI")
        self.setMinimumSize(820, 720)
        self.resize(900, 780)
        self._template_choices = []
        for item in template_choices or []:
            if len(item) != 3:
                continue
            key, label, vol = item
            arr = np.asarray(vol, dtype=np.float32)
            if arr.size == 0:
                continue
            self._template_choices.append((str(key), str(label), arr))
        if not self._template_choices and template_volume is not None:
            vol = np.asarray(template_volume, dtype=np.float32)
            if vol.size > 0:
                key = str(initial_template_key or "default")
                self._template_choices.append((key, str(template_label or "Placement template"), vol))
        if not self._template_choices:
            raise ValueError("NoiseRoiPickerDialog requires at least one template volume")

        self.selected_template_key = self._template_choices[0][0]
        pick_idx = 0
        if initial_template_key:
            for i, (key, _label, _vol) in enumerate(self._template_choices):
                if key == initial_template_key:
                    pick_idx = i
                    break
        self.selected_template_key, self._template_label, self._volume = self._template_choices[pick_idx]

        self._voxel_spacing = tuple(float(x) for x in voxel_spacing[:3]) if voxel_spacing else (1.0, 1.0, 1.0)
        while len(self._voxel_spacing) < 3:
            self._voxel_spacing = self._voxel_spacing + (self._voxel_spacing[-1],)
        self._updating = False
        self.center_mm = None
        self.center_vox = None
        self.radius_mm = float(initial_radius_mm)
        self.radius_vox = self._radius_vox_from_mm(self.radius_mm)

        if initial_center_mm is not None and len(initial_center_mm) == 3:
            self.center_mm = tuple(float(v) for v in initial_center_mm)
            self.center_vox = self._mm_to_vox(self.center_mm)
        elif self._volume.size > 0:
            shape = self._volume.shape
            self.center_vox = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
            self.center_mm = self._vox_to_mm(self.center_vox)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Place a sphere in uniform liver background (away from hot lesions). "
            "Drag the circle to move it; drag handles to resize. "
            "The preview is initial reconstruction + lesion mask for placement only. "
            "Saved center (mm) and radius (mm) are reused on every reconstruction; "
            "noise CV% during analysis reads voxel values from each selected scenario, "
            "not from this preview."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #A3BE8C; padding: 4px 0;")
        layout.addWidget(intro)

        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("Placement preview:"))
        if len(self._template_choices) == 1:
            _key, label, _vol = self._template_choices[0]
            template_label = QLabel(label)
            template_label.setStyleSheet("color: #D8DEE9; font-weight: bold;")
            template_row.addWidget(template_label, 1)
            self.template_combo = None
        else:
            self.template_combo = QComboBox()
            self.template_combo.setMinimumWidth(360)
            for key, label, _vol in self._template_choices:
                self.template_combo.addItem(label, key)
            self.template_combo.setCurrentIndex(pick_idx)
            self.template_combo.currentIndexChanged.connect(self._on_template_changed)
            template_row.addWidget(self.template_combo, 1)
        layout.addLayout(template_row)

        mid = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_label = QLabel("Axial Z: 1/1")
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 80.0)
        self.radius_spin.setDecimals(2)
        self.radius_spin.setSuffix(" mm")
        self.radius_spin.setValue(self.radius_mm)
        mid.addWidget(QLabel("Axial Z:"))
        mid.addWidget(self.slice_slider, 1)
        mid.addWidget(self.slice_label)
        mid.addSpacing(12)
        mid.addWidget(QLabel("Radius:"))
        mid.addWidget(self.radius_spin)
        layout.addLayout(mid)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(520)
        self.plot.setMenuEnabled(False)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.circle = pg.CircleROI(
            [0, 0], [10, 10], pen=pg.mkPen("#A3BE8C", width=2), movable=True, resizable=True
        )
        self.plot.addItem(self.circle)
        layout.addWidget(self.plot, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.radius_spin.valueChanged.connect(self._on_radius_spin_changed)
        self.circle.sigRegionChanged.connect(self._on_circle_changed)
        self._refresh_view()

    def _on_template_changed(self, index):
        if self.template_combo is None:
            return
        if index < 0 or index >= len(self._template_choices):
            return
        key, label, vol = self._template_choices[index]
        if vol.shape != self._volume.shape:
            QMessageBox.warning(
                self,
                "Grid mismatch",
                f"“{label}” has shape {vol.shape}, but the current template uses "
                f"{self._volume.shape}. Choose a volume on the same reconstruction grid.",
            )
            self.template_combo.blockSignals(True)
            for i, (k, _l, _v) in enumerate(self._template_choices):
                if k == self.selected_template_key:
                    self.template_combo.setCurrentIndex(i)
                    break
            self.template_combo.blockSignals(False)
            return
        self.selected_template_key = key
        self._template_label = label
        self._volume = vol
        self._refresh_view()

    def _bin_mm(self):
        spacing = self._voxel_spacing
        return float(min(s for s in spacing if s > 0) if any(s > 0 for s in spacing) else spacing[0])

    def _vox_to_mm(self, center_vox):
        cx, cy, cz = (float(center_vox[0]), float(center_vox[1]), float(center_vox[2]))
        sx, sy, sz = self._voxel_spacing
        return (cx * sx, cy * sy, cz * sz)

    def _mm_to_vox(self, center_mm):
        mx, my, mz = (float(center_mm[0]), float(center_mm[1]), float(center_mm[2]))
        sx, sy, sz = self._voxel_spacing
        return (mx / sx, my / sy, mz / sz)

    def _radius_vox_from_mm(self, radius_mm=None):
        if radius_mm is None:
            radius_mm = float(self.radius_spin.value())
        return max(float(radius_mm) / max(self._bin_mm(), 1e-6), 0.5)

    def _refresh_view(self):
        vol = self._volume
        if vol is None or vol.size == 0:
            self.image_item.clear()
            self.status_label.setText("No volume available.")
            return
        mx, my, mz = vol.shape
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max(0, mz - 1))
        if self.center_mm is None:
            self.center_vox = (mx / 2.0, my / 2.0, mz / 2.0)
            self.center_mm = self._vox_to_mm(self.center_vox)
        else:
            self.center_vox = self._mm_to_vox(self.center_mm)
        z_idx = int(np.clip(round(self.center_vox[2]), 0, mz - 1))
        self.slice_slider.setValue(z_idx)
        self.slice_slider.blockSignals(False)
        self.center_vox = (float(self.center_vox[0]), float(self.center_vox[1]), float(z_idx))
        self.center_mm = self._vox_to_mm(self.center_vox)
        self.radius_vox = self._radius_vox_from_mm(self.radius_mm)
        self.image_item.setImage(vol[:, :, z_idx], autoLevels=True)
        self._sync_circle()
        self.plot.autoRange()
        self.slice_label.setText(f"Axial Z: {z_idx + 1}/{mz}")
        self._update_status()

    def _sync_circle(self):
        if self.center_vox is None:
            return
        cx, cy, _cz = self.center_vox
        r = float(self.radius_vox)
        self._updating = True
        self.circle.setPos([cx - r, cy - r])
        self.circle.setSize([2.0 * r, 2.0 * r])
        self._updating = False

    def _on_slice_changed(self, z_idx):
        vol = self._volume
        if vol is None:
            return
        mx, my, mz = vol.shape
        z_idx = int(np.clip(z_idx, 0, mz - 1))
        if self.center_mm is None:
            self.center_vox = (mx / 2.0, my / 2.0, float(z_idx))
            self.center_mm = self._vox_to_mm(self.center_vox)
        else:
            cx, cy, _ = self.center_vox or self._mm_to_vox(self.center_mm)
            self.center_vox = (float(cx), float(cy), float(z_idx))
            self.center_mm = self._vox_to_mm(self.center_vox)
        self.image_item.setImage(vol[:, :, z_idx], autoLevels=True)
        self._sync_circle()
        self.slice_label.setText(f"Axial Z: {z_idx + 1}/{mz}")
        self._update_status()

    def _on_radius_spin_changed(self, value):
        self.radius_mm = float(value)
        self.radius_vox = self._radius_vox_from_mm(self.radius_mm)
        self._sync_circle()
        self._update_status()

    def _on_circle_changed(self):
        if self._updating:
            return
        pos = self.circle.pos()
        size = self.circle.size()
        cx = float(pos.x() + size.x() * 0.5)
        cy = float(pos.y() + size.y() * 0.5)
        r = float(max(size.x(), size.y()) * 0.5)
        if r < 0.5:
            r = 0.5
        z_idx = int(self.slice_slider.value())
        self.center_vox = (cx, cy, float(z_idx))
        self.center_mm = self._vox_to_mm(self.center_vox)
        self.radius_vox = r
        self.radius_mm = r * self._bin_mm()
        self.radius_spin.blockSignals(True)
        self.radius_spin.setValue(self.radius_mm)
        self.radius_spin.blockSignals(False)
        self._update_status()

    def _update_status(self):
        if self.center_mm is None:
            return
        mx, my, mz = self.center_mm
        cx, cy, cz = self.center_vox or (0.0, 0.0, 0.0)
        self.status_label.setText(
            f"Center: ({mx:.1f}, {my:.1f}, {mz:.1f}) mm "
            f"[({cx:.1f}, {cy:.1f}, {cz:.1f}) vox] | "
            f"radius: {self.radius_mm:.1f} mm ({self.radius_vox:.1f} vox)"
        )

    def _on_accept(self):
        if self.center_mm is None or self.radius_mm <= 0:
            QMessageBox.warning(self, "ROI incomplete", "Place the noise ROI circle on the image before confirming.")
            return
        # Final values from the circle (handles drag-resize without spin desync).
        self._on_circle_changed()
        self.accept()