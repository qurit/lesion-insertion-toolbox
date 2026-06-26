import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QEvent
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                             QPushButton, QLabel, QSlider, QDialog, QFormLayout,
                             QDoubleSpinBox, QDialogButtonBox, QSizePolicy, QStackedLayout, QMessageBox)
from PyQt5.QtGui import QPainter, QColor

class MyImageItem(pg.ImageItem):
    """A pg.ImageItem subclass that emits a signal when the LUT is changed."""
    sigLutChanged = pyqtSignal()

    def setLookupTable(self, lut, update=True):
        super().setLookupTable(lut, update)
        self.sigLutChanged.emit()
    
    def getHistogram(self, bins='auto', step='auto', targetHistogramSize=500, **kwargs):
        """
        Overridden to force step=1. 
        This ensures sparse data (like small lesions) are accurately counted in the histogram.
        """
        return super().getHistogram(bins=bins, step=1, targetHistogramSize=targetHistogramSize, **kwargs)


class MyColorBarItem(pg.ColorBarItem):
    """
    Restored for compatibility with dialogs.py.
    Includes the double-click feature for manual entry.
    """
    sigColorMapChanged = pyqtSignal()
    sigManualLevelsSet = pyqtSignal(float, float)

    def __init__(self, values=(0, 1), width=25, **kwargs):
        super().__init__(values=values, width=width, interactive=True, **kwargs)
        self.default_min = 0.0
        self.default_max = 1.0

    def setDefaultLevels(self, min_val, max_val):
        self.default_min = float(min_val)
        self.default_max = float(max_val)

    def setColorMap(self, cm):
        super().setColorMap(cm)
        self.sigColorMapChanged.emit()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            current_levels = self.levels()
            
            dialog = QDialog()
            dialog.setWindowTitle("Set Colorbar Range")
            dialog.setFixedWidth(300)
            layout = QFormLayout(dialog)
            
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1000000.0, 1000000.0)
            min_spin.setDecimals(4)
            min_spin.setValue(current_levels[0])
            
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1000000.0, 1000000.0)
            max_spin.setDecimals(4)
            max_spin.setValue(current_levels[1])
            
            layout.addRow("Min Value:", min_spin)
            layout.addRow("Max Value:", max_spin)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addRow(button_box)
            
            if dialog.exec_() == QDialog.Accepted:
                new_min = min_spin.value()
                new_max = max_spin.value()
                if new_min < new_max:
                    self.setLevels((new_min, new_max))
                    self.sigManualLevelsSet.emit(new_min, new_max)
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)


class MyHistogramLUTWidget(pg.HistogramLUTWidget):
    """
    An Advanced Histogram-based Colorbar for the Main Window.
    """
    sigManualLevelsSet = pyqtSignal(float, float)

    def sync_axis_range_to_levels(self, lo, hi, padding_ratio=0.05):
        """Fit the LUT axis ViewBox to (lo, hi) so ticks match after modality switches (e.g. CT → PET)."""
        lo, hi = float(lo), float(hi)
        span = hi - lo
        if span <= 0:
            span = 1.0
        pad = max(span * padding_ratio, 1e-6)
        vb = self.item.vb
        vb.setRange(yRange=(lo - pad, hi + pad), padding=0)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.default_min = 0.0
        self.default_max = 1.0
        self.item.axis.setPen(color=(200, 200, 200))
        # Perceptually-uniform pyqtgraph colormap presets, cycled on left-click.
        self._cmap_presets = [
            "viridis", "cividis", "plasma", "inferno", "magma",
            "turbo", "grey", "grey_inv", "hot", "cool", "jet"
        ]
        self._cmap_idx = 0
        # Mouse events are delivered to the GraphicsView viewport, not to the GradientEditorItem.
        try:
            self.setMouseTracking(True)
            self.viewport().installEventFilter(self)
        except Exception:
            pass

    def setDefaultLevels(self, min_val, max_val):
        self.default_min = float(min_val)
        self.default_max = float(max_val)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            current_levels = self.item.getLevels()
            
            dialog = QDialog()
            dialog.setWindowTitle("Set Histogram Range")
            dialog.setFixedWidth(350)
            layout = QFormLayout(dialog)
            
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1000000.0, 1000000.0)
            min_spin.setDecimals(4)
            min_spin.setValue(current_levels[0])
            
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1000000.0, 1000000.0)
            max_spin.setDecimals(4)
            max_spin.setValue(current_levels[1])
            
            layout.addRow("Min Value:", min_spin)
            layout.addRow("Max Value:", max_spin)

            restore_btn = QPushButton("Restore Defaults")
            restore_btn.setToolTip(f"Reset to: {self.default_min:.2f} / {self.default_max:.2f}")
            restore_btn.setStyleSheet("background-color: #5E81AC; color: white;")
            
            def restore_defaults():
                min_spin.setValue(self.default_min)
                max_spin.setValue(self.default_max)
            
            restore_btn.clicked.connect(restore_defaults)
            layout.addRow("", restore_btn)

            units_btn = QPushButton("Units / Quantification")
            units_btn.setStyleSheet("background-color: #A3BE8C; color: #2E3440;")
            units_btn.clicked.connect(self._open_unit_dialog)
            layout.addRow("", units_btn)

            cmap_btn = QPushButton("Change Colormap")
            cmap_btn.setToolTip("Select a professional colormap for recon/mask displays.")
            cmap_btn.setStyleSheet("background-color: #88C0D0; color: #2E3440;")
            cmap_btn.clicked.connect(self._open_colormap_dialog)
            layout.addRow("", cmap_btn)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addRow(button_box)
            
            if dialog.exec_() == QDialog.Accepted:
                new_min = min_spin.value()
                new_max = max_spin.value()
                if new_min < new_max:
                    self.item.setLevels(new_min, new_max)
                    self.sigManualLevelsSet.emit(new_min, new_max)
            
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        # Left-click cycles the colormap preset for recon/mask-style images.
        if ev.button() == Qt.LeftButton:
            parent_viewer = self.parent()
            if parent_viewer is not None and hasattr(parent_viewer, "_cycle_user_colormap_preset"):
                parent_viewer._cycle_user_colormap_preset()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def eventFilter(self, obj, event):
        # Catch left-clicks anywhere on the colorbar widget and cycle colormap.
        if obj == getattr(self, "viewport", lambda: None)() and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                parent_viewer = self.parent()
                if parent_viewer is not None and hasattr(parent_viewer, "_cycle_user_colormap_preset"):
                    parent_viewer._cycle_user_colormap_preset()
                    return True
        return super().eventFilter(obj, event)

    def _open_unit_dialog(self):
        parent_window = self._find_main_window()
        if parent_window and hasattr(parent_window, "open_unit_conversion_dialog"):
            parent_window.open_unit_conversion_dialog()
            return
        QMessageBox.information(self, "Units Not Available", "Unit conversion dialog is not available in the current context.")

    def _open_colormap_dialog(self):
        parent_viewer = self._find_parent_viewer()
        if parent_viewer is None or not hasattr(parent_viewer, "set_user_colormap_preset"):
            QMessageBox.information(self, "Colormap Not Available", "Colormap change is not available in the current context.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Choose Colormap")
        dialog.setMinimumWidth(320)
        layout = QFormLayout(dialog)

        combo = QComboBox()
        presets = getattr(self, "_cmap_presets", ["viridis", "magma", "inferno", "plasma", "grey"])
        combo.addItems(presets)
        current = getattr(parent_viewer, "user_colormap_preset", "viridis")
        if current in presets:
            combo.setCurrentText(current)

        layout.addRow("Colormap:", combo)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return
        parent_viewer.set_user_colormap_preset(combo.currentText())

    def _find_parent_viewer(self):
        # Walk QWidget parents until we find the AdvancedImageViewer
        w = self.parentWidget()
        while w is not None:
            if hasattr(w, "parent_window") and hasattr(w, "update_view"):
                return w
            w = w.parentWidget()
        # Fallback: QObject parent chain
        w = self.parent()
        while w is not None:
            if hasattr(w, "parent_window") and hasattr(w, "update_view"):
                return w
            w = w.parent()
        return None

    def _find_main_window(self):
        # Prefer getting it from the viewer, but fall back to walking parents.
        viewer = self._find_parent_viewer()
        if viewer is not None:
            mw = getattr(viewer, "parent_window", None)
            if mw is not None:
                return mw
        w = self.parentWidget()
        while w is not None:
            if hasattr(w, "open_unit_conversion_dialog"):
                return w
            w = w.parentWidget()
        return None


class AdvancedImageViewer(QWidget):
    """A widget for displaying a single 2D slice with the Advanced Histogram Control."""
    mouse_clicked_signal = pyqtSignal(object, int, int)
    view_changed_signal = pyqtSignal()
    manual_levels_signal = pyqtSignal(float, float)

    def __init__(self, parent_window, viewer_id):
        super().__init__()
        self.parent_window = parent_window
        self.viewer_id = viewer_id
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.bg_data = None
        self.fg_data = None
        
        self.fg_data_ref = None
        self.fg_stats = (0, 1)
        self.bg_data_ref = None
        self.bg_stats = (0, 1)

        self.slice_index = 0
        self.orientation = "Axial"
        self.content_type = "None"
        
        self.last_applied_content_type = None 
        self.last_fg_stats = None
        
        self.crosshair_h = None
        self.crosshair_v = None
        self.slice_info_text = "No Image Loaded"
        self.last_view_params = (None, None)
        self.needs_initial_fit = True
        self.user_colormap_preset = "viridis"
        self.force_level_reset = False

        self._refit_after_resize_timer = QTimer(self)
        self._refit_after_resize_timer.setSingleShot(True)
        self._refit_after_resize_timer.timeout.connect(self._reset_view)

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.content_combo = QComboBox()
        self.content_combo.addItems(["Initial Recon", "Registered CT", "Lesion Mask", "Final Recon", "Occurrence Map", "Reference CT"])
        
        self.bg_label = QLabel("BG:")
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["None", "Registered CT", "Initial Recon", "Reference CT", "Occurrence Map", "Final Recon", "Lesion Mask"])
        self.bg_combo.setCurrentText("None") 

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Axial", "Sagittal", "Coronal"])

        self.reset_view_btn = QPushButton("\u21BA")
        self.reset_view_btn.setToolTip("Reset View")
        self.reset_view_btn.setFixedWidth(30)

        controls_layout.addWidget(QLabel("FG:"))
        controls_layout.addWidget(self.content_combo, 2)
        controls_layout.addWidget(self.bg_label)
        controls_layout.addWidget(self.bg_combo, 2)
        controls_layout.addWidget(self.view_combo, 1)
        controls_layout.addWidget(self.reset_view_btn)

        # Plot Area (wrapped so Qt reliably gives this row vertical space from the parent)
        plot_area_container = QWidget()
        plot_area_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_area_layout = QHBoxLayout(plot_area_container)
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        # Keep plot roughly square-ish vs typical 1/3-pane width to reduce pyqtgraph letterboxing
        self.plot_widget.setMinimumHeight(320)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setAspectLocked(True)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.hideButtons()
        self.view_box = self.plot_item.getViewBox()

        # Overlay container so slice/status text appears inside the viewer (bottom-center).
        plot_overlay_container = QWidget()
        plot_overlay_layout = QStackedLayout(plot_overlay_container)
        plot_overlay_layout.setStackingMode(QStackedLayout.StackAll)
        plot_overlay_layout.setContentsMargins(0, 0, 0, 0)
        plot_overlay_layout.addWidget(self.plot_widget)
        
        # Histogram
        self.hist_widget = MyHistogramLUTWidget(self)
        self.hist_widget.setMinimumWidth(56)
        self.hist_widget.setMaximumWidth(66)
        # Show a cleaner colorbar look: hide histogram bars, keep gradient + numeric axis.
        self.hist_widget.item.plot.hide()
        self.hist_widget.item.axis.setStyle(showValues=True)
        self.hist_widget.item.axis.setWidth(40)
        # We purposely do NOT connect the drag signal to the Main Window to keep viewers independent.
        
        self.opacity_slider = QSlider(Qt.Vertical)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip("Fusion Opacity")

        self.info_label = QLabel(self.slice_info_text)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "color: #ECEFF4; font-size: 9pt; font-weight: 600; "
            "background-color: rgba(46, 52, 64, 150); padding: 2px 8px; border-radius: 4px;"
        )
        self.info_label.setContentsMargins(0, 0, 0, 0)
        self.info_label.setFixedHeight(24)
        self.info_label.setMinimumWidth(240)
        self.info_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        info_overlay_widget = QWidget()
        info_overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        info_overlay_layout = QVBoxLayout(info_overlay_widget)
        info_overlay_layout.setContentsMargins(0, 0, 0, 0)
        info_overlay_layout.addStretch(1)
        info_overlay_layout.addWidget(self.info_label, 0, Qt.AlignHCenter | Qt.AlignBottom)
        plot_overlay_layout.addWidget(info_overlay_widget)

        plot_area_layout.addWidget(plot_overlay_container, 1)
        plot_area_layout.addWidget(self.opacity_slider)
        plot_area_layout.addWidget(self.hist_widget)

        # Images
        self.image_item_bg = pg.ImageItem()
        self.image_item_bg.setZValue(0)
        self.plot_item.addItem(self.image_item_bg)
        
        self.image_item_fg = MyImageItem()
        self.image_item_fg.setZValue(1)
        self.image_item_fg.setCompositionMode(QPainter.CompositionMode_SourceOver)
        self.plot_item.addItem(self.image_item_fg)
        
        self.hist_widget.setImageItem(self.image_item_fg)

        # ROI
        self.roi = pg.CircleROI([0, 0], [1, 1], pen=pg.mkPen('r', width=2), movable=False, resizable=False, rotatable=False)
        self.roi.setVisible(False)
        self.roi.setZValue(10)
        self.plot_item.addItem(self.roi)

        layout.addLayout(controls_layout, 0)
        layout.addWidget(plot_area_container, 1)

        self.reset_view_btn.clicked.connect(self._reset_view)
        self.view_box.scene().sigMouseMoved.connect(self._mouse_moved)
        self.view_box.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.content_combo.currentTextChanged.connect(self._on_fg_content_type_changed)
        self.bg_combo.currentTextChanged.connect(self._emit_view_change)
        self.view_combo.currentTextChanged.connect(self._emit_view_change)
        self.opacity_slider.valueChanged.connect(self._emit_view_change)
        self.hist_widget.sigManualLevelsSet.connect(self.manual_levels_signal.emit)

    def _cycle_user_colormap_preset(self):
        presets = getattr(self.hist_widget, "_cmap_presets", ["viridis", "cividis", "plasma", "inferno", "magma", "turbo", "grey", "grey_inv", "hot", "cool", "jet"])
        current = getattr(self, "user_colormap_preset", "viridis")
        try:
            idx = presets.index(current)
        except ValueError:
            idx = 0
        idx = (idx + 1) % len(presets)
        self.user_colormap_preset = presets[idx]
        # Apply immediately for the current view if we're in the default (non-CT/non-occurrence) mode.
        try:
            if self.content_type not in ["Registered CT", "Reference CT", "Occurrence Map"]:
                self._apply_colormap_preset(self.user_colormap_preset)
        except Exception:
            pass
        self.view_changed_signal.emit()

    def set_user_colormap_preset(self, preset: str):
        self.user_colormap_preset = str(preset)
        try:
            if self.content_type not in ["Registered CT", "Reference CT", "Occurrence Map"]:
                self._apply_colormap_preset(self.user_colormap_preset)
        except Exception:
            pass
        self.view_changed_signal.emit()

    def _apply_colormap_preset(self, preset: str):
        if preset == "grey_inv":
            # Inverted grayscale: black->white becomes white->black
            cmap = pg.ColorMap(
                pos=[0.0, 1.0],
                color=[(255, 255, 255, 255), (0, 0, 0, 255)]
            )
            self.hist_widget.gradient.setColorMap(cmap)
        else:
            self.hist_widget.gradient.loadPreset(preset)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_item_fg.image is not None or self.image_item_bg.image is not None:
            self._refit_after_resize_timer.start(80)

    def _reset_view(self):
        items = []
        if self.image_item_bg.image is not None: items.append(self.image_item_bg)
        if self.image_item_fg.image is not None: items.append(self.image_item_fg)
        if items:
            # Fit to the union image bounds to keep slices centered in the view.
            bounds = items[0].boundingRect()
            for item in items[1:]:
                bounds = bounds.united(item.boundingRect())
            # Small padding prevents edge clipping so first/last pixels remain visible.
            self.view_box.setRange(rect=bounds, padding=0.03)

    def _on_fg_content_type_changed(self, _text):
        # Histogram/colormap state must not carry over (e.g. CT window vs PET recon) when FG source changes.
        self.force_level_reset = True
        self.last_applied_content_type = None
        self.last_fg_stats = None
        self.fg_data_ref = None
        self.view_changed_signal.emit()

    def _emit_view_change(self):
        self.view_changed_signal.emit()

    def _mouse_moved(self, pos):
        active_item = self.image_item_fg if self.image_item_fg.image is not None else self.image_item_bg
        if active_item.image is not None and self.view_box.sceneBoundingRect().contains(pos):
            mouse_point = self.view_box.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            w, h = active_item.image.shape
            if 0 <= x < w and 0 <= y < h:
                intensity = active_item.image[x, y]
                global_coords = self.parent_window._calculate_coords_from_view(self, x, y)
                gx, gy, gz = global_coords
                display_val, display_unit = self.parent_window.format_intensity_value_for_display(float(intensity), self.content_type)
                pointer_text = f"Position: ({gx+1}, {gy+1}, {gz+1}) | Value: {display_val:.2f} {display_unit}"
                if hasattr(self.parent_window, "slice_hover_label"):
                    self.parent_window.slice_hover_label.setText(pointer_text)
        else:
            if hasattr(self.parent_window, "slice_hover_label"):
                self.parent_window.slice_hover_label.setText("Hover: --")

    def _mouse_clicked(self, event):
        has_image = (self.image_item_fg.image is not None) or (self.image_item_bg.image is not None)
        if has_image and event.button() == Qt.LeftButton:
            pos = event.scenePos()
            mouse_point = self.view_box.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            w, h = 0, 0
            if self.image_item_fg.image is not None:
                w, h = self.image_item_fg.image.shape
            elif self.image_item_bg.image is not None:
                w, h = self.image_item_bg.image.shape
            if 0 <= x < w and 0 <= y < h:
                self.mouse_clicked_signal.emit(self, x, y)

    def _apply_style_to_image_item(self, image_item, data, content_type, shared_levels, shared_lut, global_min_max):
        """
        Applies styles (colormap, levels) to the image item.
        Handles independent viewer logic by ignoring shared_levels for FG.
        Prevents resetting levels on slice change.
        """
        if data is None:
            image_item.clear()
            if image_item == self.image_item_fg:
                self.last_applied_content_type = None
                self.last_fg_stats = None
            return

        d_min, d_max = global_min_max
        if d_max <= d_min: d_max = d_min + 1.0

        restore_min, restore_max = d_min, d_max

        # --- Change Detection ---
        is_new_content = False
        range_changed = False
        
        if image_item == self.image_item_fg:
            if content_type != self.last_applied_content_type:
                is_new_content = True
                self.last_applied_content_type = content_type
            
            if self.last_fg_stats != (d_min, d_max):
                range_changed = True
                self.last_fg_stats = (d_min, d_max)

        # --- Case A: CT Maps ---
        if content_type in ["Registered CT", "Reference CT"]:
            if content_type == "Registered CT":
                win_min, win_max = -160, 240
            else:
                win_min, win_max = -1025, -530 
            
            restore_min, restore_max = win_min, win_max
            
            pos = [0.0, 1.0]
            colors = [(0, 0, 0, 255), (255, 255, 255, 255)]
            cmap = pg.ColorMap(pos=pos, color=colors)
            lut = cmap.getLookupTable(nPts=512)
            
            image_item.setImage(data, autoLevels=False)

            if image_item == self.image_item_fg:
                # For CT, we enforce standard windows, but allow user deviation afterward.
                # Only reset if content changed.
                if is_new_content:
                    self.hist_widget.gradient.setColorMap(cmap)
                    self.hist_widget.setLevels(win_min, win_max)
                    self.hist_widget.setDefaultLevels(restore_min, restore_max)
                    self.hist_widget.sync_axis_range_to_levels(win_min, win_max, padding_ratio=0.08)
            else:
                image_item.setLookupTable(lut)
                image_item.setLevels((win_min, win_max))

        # --- Case B: Occurrence Map ---
        elif content_type == "Occurrence Map":
            pos = [0.0, 0.4, 1.0]
            colors = [(0, 0, 0, 255), (255, 0, 0, 255), (255, 255, 0, 255)]
            yr_cmap = pg.ColorMap(pos=pos, color=colors)
            lut = yr_cmap.getLookupTable()
            
            image_item.setImage(data, autoLevels=False)
            
            if image_item == self.image_item_fg:
                if is_new_content or range_changed:
                    self.hist_widget.gradient.setColorMap(yr_cmap)
                    self.hist_widget.setLevels(d_min, d_max)
                    self.hist_widget.setDefaultLevels(d_min, d_max)
                    self.hist_widget.sync_axis_range_to_levels(d_min, d_max)
            else:
                image_item.setLookupTable(lut)
                image_item.setLevels((d_min, d_max))

        # --- Case C: Default / Shared Logic (Recon, Mask) ---
        else:
            image_item.setImage(data, autoLevels=False)
            
            # Use local stats (independent levels) instead of shared_levels
            target_levels = (d_min, d_max)

            if image_item == self.image_item_fg:
                 # 1. Reset on New Content or Significant Data Change
                 if self.force_level_reset or (is_new_content and content_type in ["Initial Recon", "Final Recon", "Lesion Mask"]) or range_changed:
                      self._apply_colormap_preset(getattr(self, "user_colormap_preset", "viridis"))
                      self.hist_widget.setLevels(target_levels[0], target_levels[1])
                      self.hist_widget.setDefaultLevels(d_min, d_max)
                      self.force_level_reset = False
                      
                      image_item.setLevels(target_levels)
                      self.hist_widget.sync_axis_range_to_levels(d_min, d_max)

                 # 2. Slice scroll: keep levels; no axis resync.
                 # Pyqtgraph's internal state handles the persistence of levels.
                 else:
                      pass
            else:
                image_item.setLevels(target_levels)
                if shared_lut is None:
                    image_item.setLookupTable(pg.colormap.get('viridis').getLookupTable())
                else:
                    image_item.setLookupTable(shared_lut)

    def update_view(self, coords, images, shared_levels, shared_lut):
        current_view_params = (self.content_combo.currentText(), self.bg_combo.currentText(), self.view_combo.currentText())
        if self.last_view_params != current_view_params:
            self.last_view_params = current_view_params
            self.needs_initial_fit = True

        self.content_type = self.content_combo.currentText()
        bg_type = self.bg_combo.currentText()
        self.orientation = self.view_combo.currentText()

        content_key_map = {
            "Initial Recon": "initial", "Registered CT": "ct", "Lesion Mask": "lesion",
            "Final Recon": "final", "Occurrence Map": "occurrence_pet", "Reference CT": "occurrence_ct",
            "None": None
        }
        
        self.fg_data = images.get(content_key_map.get(self.content_type))
        self.bg_data = images.get(content_key_map.get(bg_type))

        if self.fg_data is not None:
            # If the content type is Lesion Mask, do not rely on object ID caching
            # because the array is often modified in-place (same ID, different data).
            is_dynamic = (self.content_type == "Lesion Mask")
            
            if (self.fg_data_ref is not self.fg_data) or is_dynamic:
                self.fg_data_ref = self.fg_data
                self.fg_stats = (np.min(self.fg_data), np.max(self.fg_data))
        else:
            self.fg_data_ref = None; self.fg_stats = (0, 1)

        if self.bg_data is not None:
            if self.bg_data_ref is not self.bg_data:
                self.bg_data_ref = self.bg_data
                self.bg_stats = (np.min(self.bg_data), np.max(self.bg_data))
        else:
            self.bg_data_ref = None; self.bg_stats = (0, 1)

        if self.fg_data is None and self.bg_data is None:
            self.image_item_bg.clear(); self.image_item_fg.clear()
            self.slice_info_text = "No Image Loaded"
            self.info_label.setText(self.slice_info_text)
            self._update_crosshairs(coords)
            self.needs_initial_fit = True
            return

        x, y, z = coords
        
        def get_slice_data(data, ori):
            if data is None: return None, 0
            mx, my, mz = data.shape
            if ori == "Axial":
                self.slice_index = np.clip(z, 0, mz-1)
                return data[:, :, self.slice_index], mz
            elif ori == "Sagittal":
                self.slice_index = np.clip(x, 0, mx-1)
                return data[self.slice_index, :, :], mx
            elif ori == "Coronal":
                self.slice_index = np.clip(y, 0, my-1)
                return data[:, self.slice_index, :], my
            return None, 0

        bg_slice, _ = get_slice_data(self.bg_data, self.orientation)
        fg_slice, max_dim = get_slice_data(self.fg_data, self.orientation)
        if fg_slice is None and bg_slice is not None: _, max_dim = get_slice_data(self.bg_data, self.orientation)

        self.slice_info_text = f"Slice: {self.slice_index + 1}/{max_dim}"
        self.info_label.setText(self.slice_info_text)

        # Note: shared_levels passed here is ignored by _apply_style... for FG item
        self._apply_style_to_image_item(self.image_item_bg, bg_slice, bg_type, None, None, self.bg_stats)
        self._apply_style_to_image_item(self.image_item_fg, fg_slice, self.content_type, shared_levels, None, self.fg_stats)
        
        opacity = self.opacity_slider.value() / 100.0
        self.image_item_fg.setOpacity(opacity)

        if self.needs_initial_fit:
            self._reset_view()
            self.needs_initial_fit = False

        self._update_crosshairs(coords)
        self._update_roi(self.parent_window.sphere_center, self.parent_window._get_current_radius_for_display_px(), coords)

    def _update_crosshairs(self, coords):
        x, y, z = coords
        if self.crosshair_h: self.plot_item.removeItem(self.crosshair_h)
        if self.crosshair_v: self.plot_item.removeItem(self.crosshair_v)

        if self.orientation == "Axial":
            self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pos=x)
            self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pos=y)
        elif self.orientation == "Sagittal":
            self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pos=y)
            self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pos=z)
        elif self.orientation == "Coronal":
            self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pos=x)
            self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pos=z)

        pen = pg.mkPen(color=(255, 255, 0, 150), style=Qt.DotLine, width=1.5)
        self.crosshair_h.setPen(pen)
        self.crosshair_v.setPen(pen)
        self.crosshair_h.setZValue(1000)
        self.crosshair_v.setZValue(1000)
        
        self.plot_item.addItem(self.crosshair_h, ignoreBounds=True)
        self.plot_item.addItem(self.crosshair_v, ignoreBounds=True)

    def _update_roi(self, center, radius, coords):
        if center is None or radius <= 0.1:
            self.roi.setVisible(False)
            return

        cx, cy, cz = center
        show_roi = False
        if self.orientation == 'Axial' and cz == self.slice_index:
            self.roi.setPos([cx - radius, cy - radius])
            show_roi = True
        elif self.orientation == 'Sagittal' and cx == self.slice_index:
            self.roi.setPos([cy - radius, cz - radius])
            show_roi = True
        elif self.orientation == 'Coronal' and cy == self.slice_index:
            self.roi.setPos([cx - radius, cz - radius])
            show_roi = True

        if show_roi:
            safe_size = max(radius * 2, 0.1)
            self.roi.setSize([safe_size, safe_size])
            self.roi.setVisible(True)
        else:
            self.roi.setVisible(False)