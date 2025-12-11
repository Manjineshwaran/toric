import sys
import os
import queue
import multiprocessing
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTabWidget, QCheckBox, QGroupBox, QFormLayout,
                             QScrollArea, QSizePolicy, QRadioButton, QButtonGroup,
                             QFileDialog, QMessageBox, QDialog, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage

# Import handler to connect UI with main pipeline
from handler import get_config_handler

# Import preprocessing functions
from preprocess_robust import (
    load_yolo_model,
    preprocess_single,
    ImageType,
    Config,
    detect_limbus,
    create_ring_mask,
    apply_mask,
    trim_eyelids_from_mask
)

# Import SuperPoint feature detection functions
from superpoint_6_log_saved import (
    initialize_models,
    extract_features,
    save_keypoints_image,
    AccuracyConfig
)
import torch

class IntraopDisplayThread(QThread):
    """Background display thread to overlay angles and emit pixmaps."""
    frame_ready = pyqtSignal(QPixmap)
    rotation_updated = pyqtSignal(float, float)

    def __init__(self, frame_queue, angles, parent=None):
        super().__init__(parent)
        self.frame_queue = frame_queue
        self.angles = angles
        self._running = True

    def stop(self):
        self._running = False
        try:
            self.frame_queue.put_nowait(None)
        except Exception:
            pass

    def run(self):
        while self._running:
            try:
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                frame, rotation_angle, confidence = item
                overlay = frame.copy()

                ref_angle = self.angles.get("reference", 0.0)
                toric_angle = self.angles.get("toric", 0.0)
                incision_angle = self.angles.get("incision", 0.0)

                cv2.putText(
                    overlay,
                    f"Detected: {rotation_angle:.2f} deg  (conf {confidence*100:.1f}%)",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"Ref {ref_angle:.1f} | Toric {toric_angle:.1f} | Incision {incision_angle:.1f}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

                rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                self.frame_ready.emit(pixmap)
                self.rotation_updated.emit(rotation_angle, confidence)
            except Exception:
                continue

        self._running = False

class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setAlignment(Qt.AlignCenter)
        self.show_ref_line = True
        self.show_toric_axis = True
        self.show_incision_axis = True
        self.show_iris_box = True
        self.pixmap = None
        
        # Store limbus info and angles for drawing
        self.limbus_center = None  # (x, y) in image coordinates
        self.limbus_radius = None
        self.reference_angle = 0.0
        self.toric_angle = 0.0
        self.incision_angle = 0.0
        self.image_size = None  # (width, height) of original image
        
    def set_image(self, pixmap):
        self.pixmap = pixmap
        if pixmap:
            self.image_size = (pixmap.width(), pixmap.height())
        self.update()
    
    def set_limbus_info(self, center, radius):
        """Set limbus center and radius (in image coordinates)"""
        self.limbus_center = center
        self.limbus_radius = radius
        self.update()
    
    def set_angles(self, reference_angle, toric_angle, incision_angle):
        """Set angles for drawing axes"""
        self.reference_angle = reference_angle
        self.toric_angle = toric_angle
        self.incision_angle = incision_angle
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the image if loaded
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            # No image loaded, don't draw any overlays
            return
        
        # Only draw overlays if any are enabled
        if not (self.show_ref_line or self.show_toric_axis or 
                self.show_incision_axis or self.show_iris_box):
            return
        
        # Calculate scaling and offset for image-to-canvas coordinate mapping
        if not self.pixmap or not self.image_size:
            # Fallback to center if no image
            center_x = self.width() // 2
            center_y = self.height() // 2
            radius = min(self.width(), self.height()) // 3
        else:
            # Calculate how the image is scaled and positioned
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            pixmap_x = (self.width() - scaled_pixmap.width()) // 2
            pixmap_y = (self.height() - scaled_pixmap.height()) // 2
            
            # Calculate scale factor
            scale_x = scaled_pixmap.width() / self.image_size[0]
            scale_y = scaled_pixmap.height() / self.image_size[1]
            
            # Map limbus center from image coordinates to canvas coordinates
            if self.limbus_center and self.limbus_radius:
                center_x = pixmap_x + int(self.limbus_center[0] * scale_x)
                center_y = pixmap_y + int(self.limbus_center[1] * scale_y)
                radius = int(self.limbus_radius * min(scale_x, scale_y))
            else:
                # Fallback to center if no limbus info
                center_x = self.width() // 2
                center_y = self.height() // 2
                radius = min(self.width(), self.height()) // 3
        
        import math
        
        # Draw iris bounding box (broken green line)
        if self.show_iris_box:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DotLine))  # Green dotted for limbus circle
            painter.drawEllipse(center_x - radius, center_y - radius, 
                              radius * 2, radius * 2)
            # Draw center point proportional to limbus radius (2.5% of radius, minimum 1 pixel)
            center_point_radius = max(1, int(radius * 0.025))
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(center_x - center_point_radius, center_y - center_point_radius,
                              center_point_radius * 2, center_point_radius * 2)
        
        # Draw reference line at reference_angle (broken yellow line)
        if self.show_ref_line:
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DotLine))  # Yellow dotted for reference
            ref_angle_rad = math.radians(self.reference_angle)
            length = int(radius * 1.5)
            x1 = center_x + int(length * math.cos(ref_angle_rad))
            y1 = center_y - int(length * math.sin(ref_angle_rad))  # Negative because y increases downward
            x2 = center_x - int(length * math.cos(ref_angle_rad))
            y2 = center_y + int(length * math.sin(ref_angle_rad))
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw toric axis at toric_angle (solid blue line with parallel offset lines)
        if self.show_toric_axis:
            # Blue: RGB(0, 0, 255) = QColor(0, 0, 255)
            pen = QPen(QColor(0, 0, 255), 2, Qt.SolidLine)  # Blue solid for toric
            painter.setPen(pen)
            toric_angle_rad = math.radians(self.toric_angle)
            length = int(radius * 1.5)
            x1 = center_x + int(length * math.cos(toric_angle_rad))
            y1 = center_y - int(length * math.sin(toric_angle_rad))
            x2 = center_x - int(length * math.cos(toric_angle_rad))
            y2 = center_y + int(length * math.sin(toric_angle_rad))
            
            # Draw main toric line
            painter.drawLine(x1, y1, x2, y2)
            
            # Draw two parallel offset lines on both sides
            # Offset distance is 5% of limbus radius (e.g., 10 pixels for 200 pixel radius)
            offset_distance = max(1, int(radius * 0.05))
            
            # Calculate perpendicular direction (rotate by 90 degrees)
            # Perpendicular vector: (-sin(angle), cos(angle))
            perp_x = -math.sin(toric_angle_rad)
            perp_y = math.cos(toric_angle_rad)
            
            # Offset line 1 (one side)
            x1_offset1 = int(x1 + offset_distance * perp_x)
            y1_offset1 = int(y1 - offset_distance * perp_y)  # Negative because y increases downward
            x2_offset1 = int(x2 + offset_distance * perp_x)
            y2_offset1 = int(y2 - offset_distance * perp_y)
            painter.drawLine(x1_offset1, y1_offset1, x2_offset1, y2_offset1)
            
            # Offset line 2 (other side)
            x1_offset2 = int(x1 - offset_distance * perp_x)
            y1_offset2 = int(y1 + offset_distance * perp_y)  # Positive because y increases downward
            x2_offset2 = int(x2 - offset_distance * perp_x)
            y2_offset2 = int(y2 + offset_distance * perp_y)
            painter.drawLine(x1_offset2, y1_offset2, x2_offset2, y2_offset2)
        
        # Draw incision axis at incision_angle
        if self.show_incision_axis:
            painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red for incision (RGB format)
            incision_angle_rad = math.radians(self.incision_angle)
            length = int(radius * 1.5)
            x1 = center_x + int(length * math.cos(incision_angle_rad))
            y1 = center_y - int(length * math.sin(incision_angle_rad))
            x2 = center_x - int(length * math.cos(incision_angle_rad))
            y2 = center_y + int(length * math.sin(incision_angle_rad))
            painter.drawLine(x1, y1, x2, y2)

class ToricTrackerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Toric Marker Tracker")
        self.setGeometry(100, 100, 1400, 800)
        
        # Get configuration handler instance
        self.config_handler = get_config_handler()
        
        # Initialize UI with current handler values
        self._initialize_ui_from_handler()
        
        # Store current image and rotation
        self.current_preop_image = None
        self.current_preop_image_path = None
        self.preprocessed_preop_image = None
        self.rotation_angle = 0
        
        # Preop YOLO model (loaded when needed)
        self.preop_yolo_model = None
        
        # Mask live update state
        self.mask_live_mode = False
        self.cropped_image_for_mask = None
        self.limbus_info_for_mask = None
        
        # SuperPoint feature detection models (initialized when needed)
        self.feature_extractor = None
        self.feature_matcher = None
        self.feature_device = None
        self.feature_config = None
        self.preop_features = None
        
        # Tracking state
        self.is_tracking = False
        self.tracking_video_path = None
        self.tracking_quit_flag = [False]
        self.tracking_pause_flag = [False]
        self.tracking_thread = None
        self.tracking_stats_logger = None
        self.intraop_display_thread = None
        self.display_frame_queue = None
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top reset button
        reset_btn = QPushButton("Reset All Application State")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        reset_btn.clicked.connect(self.reset_all)
        main_layout.addWidget(reset_btn)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #2196F3;
            }
        """)
        
        # Create tabs
        tabs.addTab(self.create_config_tab(), "1. Configuration")
        tabs.addTab(self.create_preop_image_tab(), "2. Pre-op Image")
        tabs.addTab(self.create_axis_setup_tab(), "3. Pre-op Axis Setup")
        tabs.addTab(self.create_live_tracking_tab(), "4. Live Tracking")
        # tabs.addTab(QWidget(), "6. Pre-op vs Intra-op")
        # tabs.addTab(QWidget(), "5. Export Data")
        
        # Connect tab change signal to load limbus detection image when tab 3 is selected
        tabs.currentChanged.connect(self.on_tab_changed)
        self.tabs_widget = tabs  # Store reference for tab index checking
        
        main_layout.addWidget(tabs)
    
    def _initialize_ui_from_handler(self):
        """Initialize UI elements with values from handler"""
        # Load model paths from handler
        preop_path = self.config_handler.get_preop_model_path()
        intraop_path = self.config_handler.get_intraop_model_path()
        
        # Update status labels if models are set (show default if set)
        if hasattr(self, 'label_preop_status'):
            if preop_path:
                file_name = os.path.basename(preop_path)
                self.label_preop_status.setText(f"Pre-op Model: {file_name}")
            else:
                self.label_preop_status.setText("Pre-op Model: None loaded.")
        
        if hasattr(self, 'label_intraop_status'):
            if intraop_path:
                file_name = os.path.basename(intraop_path)
                self.label_intraop_status.setText(f"Intra-op Model: {file_name}")
            else:
                self.label_intraop_status.setText("Intra-op Model: None loaded.")
        
        # Load confidence values from handler
        if hasattr(self, 'confidence_input'):
            self.confidence_input.setText(str(self.config_handler.get_yolo_confidence()))
        
        if hasattr(self, 'matching_confidence_input'):
            self.matching_confidence_input.setText(str(self.config_handler.get_matching_confidence_threshold()))
        
        if hasattr(self, 'freeze_confidence_input'):
            self.freeze_confidence_input.setText(str(self.config_handler.get_freeze_confidence_threshold()))
        
        # Load app mode from handler
        if hasattr(self, 'radio_normal') and hasattr(self, 'radio_demo'):
            if self.config_handler.get_app_mode() == 'normal':
                self.radio_normal.setChecked(True)
            else:
                self.radio_demo.setChecked(True)
        
    def create_config_tab(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #f5f5f5;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Select Application Mode")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #2196F3; background: transparent;")
        layout.addWidget(title_label)
        
        # Mode selection group
        mode_group = QGroupBox()
        mode_group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 20px;
            }
        """)
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(50)
        
        # Radio buttons
        self.radio_normal = QRadioButton("Normal Mode (Default Models, Not Editable)")
        self.radio_normal.setChecked(True)
        self.radio_normal.setStyleSheet("""
            QRadioButton {
                font-size: 11px;
                spacing: 10px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        self.radio_demo = QRadioButton("Demo Mode (User-Selected Models)")
        self.radio_demo.setStyleSheet(self.radio_normal.styleSheet())
        
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.radio_normal)
        self.mode_button_group.addButton(self.radio_demo)
        
        mode_layout.addWidget(self.radio_normal)
        mode_layout.addWidget(self.radio_demo)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # YOLO Models section
        yolo_label = QLabel("YOLO Models")
        yolo_label.setFont(QFont("Arial", 14, QFont.Bold))
        yolo_label.setStyleSheet("color: #2196F3; background: transparent; margin-top: 20px;")
        layout.addWidget(yolo_label)
        
        yolo_group = QGroupBox()
        yolo_group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 20px;
            }
        """)
        yolo_layout = QVBoxLayout()
        yolo_layout.setSpacing(15)
        
        # Pre-op Model (initialize from handler)
        preop_layout = QHBoxLayout()
        self.btn_load_preop = QPushButton("Load Pre-op Model (.pt)")
        self.btn_load_preop.setFixedWidth(200)
        self.btn_load_preop.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_load_preop.clicked.connect(self.load_preop_model)
        
        handler = get_config_handler()
        preop_path = handler.get_preop_model_path()
        # Show default model path if it exists, otherwise show "None loaded"
        if preop_path:
            file_name = os.path.basename(preop_path)
            self.label_preop_status = QLabel(f"Pre-op Model: {file_name}")
        else:
            self.label_preop_status = QLabel("Pre-op Model: None loaded.")
        self.label_preop_status.setStyleSheet("color: #666; font-size: 11px;")
        
        preop_layout.addWidget(self.btn_load_preop)
        preop_layout.addWidget(self.label_preop_status)
        preop_layout.addStretch()
        
        # Intra-op Model (initialize from handler)
        intraop_layout = QHBoxLayout()
        self.btn_load_intraop = QPushButton("Load Intra-op Model (.pt)")
        self.btn_load_intraop.setFixedWidth(200)
        self.btn_load_intraop.setStyleSheet(self.btn_load_preop.styleSheet())
        self.btn_load_intraop.clicked.connect(self.load_intraop_model)
        
        handler = get_config_handler()
        intraop_path = handler.get_intraop_model_path()
        # Show default model path if it exists, otherwise show "None loaded"
        if intraop_path:
            file_name = os.path.basename(intraop_path)
            self.label_intraop_status = QLabel(f"Intra-op Model: {file_name}")
        else:
            self.label_intraop_status = QLabel("Intra-op Model: None loaded.")
        self.label_intraop_status.setStyleSheet("color: #666; font-size: 11px;")
        
        intraop_layout.addWidget(self.btn_load_intraop)
        intraop_layout.addWidget(self.label_intraop_status)
        intraop_layout.addStretch()
        
        # Confidence score (initialize from handler)
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("YOLO Confidence Score (0.0 – 1.0):")
        confidence_label.setStyleSheet("font-size: 11px;")
        handler = get_config_handler()
        self.confidence_input = QLineEdit(str(handler.get_yolo_confidence()))
        self.confidence_input.setFixedWidth(100)
        self.confidence_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        confidence_arrow_buttons = self.create_arrow_buttons(self.confidence_input, 0.0, 1.0)
        
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_input)
        confidence_layout.addWidget(confidence_arrow_buttons)
        confidence_layout.addStretch()
        
        # Matching confidence threshold (initialize from handler)
        matching_confidence_layout = QHBoxLayout()
        matching_confidence_label = QLabel("Matching Confidence Threshold (0.0 – 1.0):")
        matching_confidence_label.setStyleSheet("font-size: 11px;")
        handler = get_config_handler()
        self.matching_confidence_input = QLineEdit(str(handler.get_matching_confidence_threshold()))
        self.matching_confidence_input.setFixedWidth(100)
        self.matching_confidence_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        matching_confidence_arrow_buttons = self.create_arrow_buttons(self.matching_confidence_input, 0.0, 1.0)
        
        matching_confidence_layout.addWidget(matching_confidence_label)
        matching_confidence_layout.addWidget(self.matching_confidence_input)
        matching_confidence_layout.addWidget(matching_confidence_arrow_buttons)
        matching_confidence_layout.addStretch()
        
        # Freeze confidence threshold (initialize from handler)
        freeze_confidence_layout = QHBoxLayout()
        freeze_confidence_label = QLabel("Freeze Confidence Threshold (0.0 – 1.0):")
        freeze_confidence_label.setStyleSheet("font-size: 11px;")
        handler = get_config_handler()
        self.freeze_confidence_input = QLineEdit(str(handler.get_freeze_confidence_threshold()))
        self.freeze_confidence_input.setFixedWidth(100)
        self.freeze_confidence_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        freeze_confidence_arrow_buttons = self.create_arrow_buttons(self.freeze_confidence_input, 0.0, 1.0)
        
        freeze_confidence_layout.addWidget(freeze_confidence_label)
        freeze_confidence_layout.addWidget(self.freeze_confidence_input)
        freeze_confidence_layout.addWidget(freeze_confidence_arrow_buttons)
        freeze_confidence_layout.addStretch()
        
        yolo_layout.addLayout(preop_layout)
        yolo_layout.addLayout(intraop_layout)
        yolo_layout.addLayout(confidence_layout)
        yolo_layout.addLayout(matching_confidence_layout)
        yolo_layout.addLayout(freeze_confidence_layout)
        
        yolo_group.setLayout(yolo_layout)
        layout.addWidget(yolo_group)
        
        # Submit button
        self.btn_submit_config = QPushButton("Submit / Start Registration")
        self.btn_submit_config.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_submit_config.clicked.connect(self.submit_configuration)
        layout.addWidget(self.btn_submit_config)
        
        layout.addStretch()
        
        return widget
    
    def create_preop_image_tab(self):
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setSpacing(0)
        
        # Left panel with controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(380)
        left_panel.setStyleSheet("background-color: #f5f5f5;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(20)
        
        # Step 1: Select Pre-op Image Source
        step1_label = QLabel("Step 1: Select Pre-op Image Source")
        step1_label.setFont(QFont("Arial", 12, QFont.Bold))
        step1_label.setStyleSheet("color: #2196F3;")
        left_layout.addWidget(step1_label)
        
        step1_group = QWidget()
        step1_group.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
        """)
        step1_group_layout = QVBoxLayout(step1_group)
        step1_group_layout.setContentsMargins(15, 15, 15, 15)
        step1_group_layout.setSpacing(10)
        
        # Radio buttons for source selection
        self.radio_camera = QRadioButton("Camera Input")
        self.radio_file = QRadioButton("File Source")
        self.radio_file.setChecked(True)
        
        self.source_button_group = QButtonGroup()
        self.source_button_group.addButton(self.radio_camera)
        self.source_button_group.addButton(self.radio_file)
        
        step1_group_layout.addWidget(self.radio_camera)
        step1_group_layout.addWidget(self.radio_file)
        
        # Browse button
        self.btn_browse_image = QPushButton("Browse for Image")
        self.btn_browse_image.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_browse_image.clicked.connect(self.load_preop_image)
        step1_group_layout.addWidget(self.btn_browse_image)
        
        # Preprocess Image button
        self.btn_preprocess_image = QPushButton("Preprocess Image")
        self.btn_preprocess_image.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_preprocess_image.clicked.connect(self.preprocess_preop_image)
        step1_group_layout.addWidget(self.btn_preprocess_image)
        
        left_layout.addWidget(step1_group)
        
        # Step 2: Register Pre-op Image
        step2_label = QLabel("Step 2: Register Pre-op Image")
        step2_label.setFont(QFont("Arial", 12, QFont.Bold))
        step2_label.setStyleSheet("color: #2196F3;")
        left_layout.addWidget(step2_label)
        
        step2_group = QWidget()
        step2_group.setStyleSheet(step1_group.styleSheet())
        step2_group_layout = QVBoxLayout(step2_group)
        step2_group_layout.setContentsMargins(15, 15, 15, 15)
        step2_group_layout.setSpacing(10)
        
        # Status label
        self.preop_status_label = QLabel("Pre-op image loaded: IMG_4139.jpeg. Detecting iris and features...")
        self.preop_status_label.setWordWrap(True)
        self.preop_status_label.setStyleSheet("color: #666; font-size: 10px;")
        step2_group_layout.addWidget(self.preop_status_label)
        
        # Rotate button
        self.btn_rotate = QPushButton("Rotate 90°")
        self.btn_rotate.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_rotate.clicked.connect(self.rotate_image)
        step2_group_layout.addWidget(self.btn_rotate)
        
        # Add Mask button
        self.btn_add_mask = QPushButton("Add Mask")
        self.btn_add_mask.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_add_mask.clicked.connect(self.add_mask)
        step2_group_layout.addWidget(self.btn_add_mask)
        
        # Configuration parameters section
        config_label = QLabel("Configuration Parameters:")
        config_label.setFont(QFont("Arial", 10, QFont.Bold))
        config_label.setStyleSheet("color: #2196F3; margin-top: 10px;")
        step2_group_layout.addWidget(config_label)
        
        # Inner/Outer Circle Ratios (initialize from handler)
        handler = get_config_handler()
        inner_exclude_layout = QHBoxLayout()
        inner_exclude_label = QLabel("Inner Exclude Ratio:")
        inner_exclude_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.inner_exclude_ratio_input = QLineEdit(str(handler.get_inner_exclude_ratio()))
        self.inner_exclude_ratio_input.setFixedWidth(80)
        self.inner_exclude_ratio_input.setStyleSheet("""
            QLineEdit {
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 10px;
            }
        """)
        inner_arrow_buttons = self.create_arrow_buttons(self.inner_exclude_ratio_input, 0.0, 2.0)
        inner_exclude_layout.addWidget(inner_exclude_label)
        inner_exclude_layout.addWidget(self.inner_exclude_ratio_input)
        inner_exclude_layout.addWidget(inner_arrow_buttons)
        inner_exclude_layout.addStretch()
        step2_group_layout.addLayout(inner_exclude_layout)
        
        outer_include_layout = QHBoxLayout()
        outer_include_label = QLabel("Outer Include Ratio:")
        outer_include_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.outer_include_ratio_input = QLineEdit(str(handler.get_outer_include_ratio()))
        self.outer_include_ratio_input.setFixedWidth(80)
        self.outer_include_ratio_input.setStyleSheet(self.inner_exclude_ratio_input.styleSheet())
        outer_arrow_buttons = self.create_arrow_buttons(self.outer_include_ratio_input, 0.0, 3.0)
        outer_include_layout.addWidget(outer_include_label)
        outer_include_layout.addWidget(self.outer_include_ratio_input)
        outer_include_layout.addWidget(outer_arrow_buttons)
        outer_include_layout.addStretch()
        step2_group_layout.addLayout(outer_include_layout)
        
        # Crop Ratios (initialize from handler)
        crop_width_layout = QHBoxLayout()
        crop_width_label = QLabel("Crop Width Ratio:")
        crop_width_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.crop_width_ratio_input = QLineEdit(str(handler.get_crop_width_ratio()))
        self.crop_width_ratio_input.setFixedWidth(80)
        self.crop_width_ratio_input.setStyleSheet(self.inner_exclude_ratio_input.styleSheet())
        crop_width_arrow_buttons = self.create_arrow_buttons(self.crop_width_ratio_input, 0.0, 10.0)
        crop_width_layout.addWidget(crop_width_label)
        crop_width_layout.addWidget(self.crop_width_ratio_input)
        crop_width_layout.addWidget(crop_width_arrow_buttons)
        crop_width_layout.addStretch()
        step2_group_layout.addLayout(crop_width_layout)
        
        crop_height_layout = QHBoxLayout()
        crop_height_label = QLabel("Crop Height Ratio:")
        crop_height_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.crop_height_ratio_input = QLineEdit(str(handler.get_crop_height_ratio()))
        self.crop_height_ratio_input.setFixedWidth(80)
        self.crop_height_ratio_input.setStyleSheet(self.inner_exclude_ratio_input.styleSheet())
        crop_height_arrow_buttons = self.create_arrow_buttons(self.crop_height_ratio_input, 0.0, 10.0)
        crop_height_layout.addWidget(crop_height_label)
        crop_height_layout.addWidget(self.crop_height_ratio_input)
        crop_height_layout.addWidget(crop_height_arrow_buttons)
        crop_height_layout.addStretch()
        step2_group_layout.addLayout(crop_height_layout)
        
        # Eyelid Trim Ratios (initialize from handler)
        eyelid_upper_layout = QHBoxLayout()
        eyelid_upper_label = QLabel("Eyelid Trim Upper Ratio:")
        eyelid_upper_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.eyelid_trim_upper_ratio_input = QLineEdit(str(handler.get_eyelid_trim_upper_ratio()))
        self.eyelid_trim_upper_ratio_input.setFixedWidth(80)
        self.eyelid_trim_upper_ratio_input.setStyleSheet(self.inner_exclude_ratio_input.styleSheet())
        eyelid_upper_arrow_buttons = self.create_arrow_buttons(self.eyelid_trim_upper_ratio_input, 0.0, 2.0)
        eyelid_upper_layout.addWidget(eyelid_upper_label)
        eyelid_upper_layout.addWidget(self.eyelid_trim_upper_ratio_input)
        eyelid_upper_layout.addWidget(eyelid_upper_arrow_buttons)
        eyelid_upper_layout.addStretch()
        step2_group_layout.addLayout(eyelid_upper_layout)
        
        eyelid_lower_layout = QHBoxLayout()
        eyelid_lower_label = QLabel("Eyelid Trim Lower Ratio:")
        eyelid_lower_label.setStyleSheet("font-size: 10px; min-width: 140px;")
        self.eyelid_trim_lower_ratio_input = QLineEdit(str(handler.get_eyelid_trim_lower_ratio()))
        self.eyelid_trim_lower_ratio_input.setFixedWidth(80)
        self.eyelid_trim_lower_ratio_input.setStyleSheet(self.inner_exclude_ratio_input.styleSheet())
        eyelid_lower_arrow_buttons = self.create_arrow_buttons(self.eyelid_trim_lower_ratio_input, 0.0, 2.0)
        eyelid_lower_layout.addWidget(eyelid_lower_label)
        eyelid_lower_layout.addWidget(self.eyelid_trim_lower_ratio_input)
        eyelid_lower_layout.addWidget(eyelid_lower_arrow_buttons)
        eyelid_lower_layout.addStretch()
        step2_group_layout.addLayout(eyelid_lower_layout)
        
        # Connect parameter inputs to save to handler and live mask updates
        self.inner_exclude_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        self.outer_include_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        self.eyelid_trim_upper_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        self.eyelid_trim_lower_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        self.crop_width_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        self.crop_height_ratio_input.editingFinished.connect(self._save_preprocessing_params_to_handler)
        
        # Connect to live mask updates
        self.inner_exclude_ratio_input.editingFinished.connect(self.update_mask_live)
        self.outer_include_ratio_input.editingFinished.connect(self.update_mask_live)
        self.eyelid_trim_upper_ratio_input.editingFinished.connect(self.update_mask_live)
        self.eyelid_trim_lower_ratio_input.editingFinished.connect(self.update_mask_live)
        # Also connect textChanged for real-time updates (with debouncing handled in update_mask_live)
        self.inner_exclude_ratio_input.textChanged.connect(self.update_mask_live)
        self.outer_include_ratio_input.textChanged.connect(self.update_mask_live)
        self.eyelid_trim_upper_ratio_input.textChanged.connect(self.update_mask_live)
        self.eyelid_trim_lower_ratio_input.textChanged.connect(self.update_mask_live)
        
        # Mask Eye button
        self.btn_maskeye = QPushButton("Mask Eye")
        self.btn_maskeye.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_maskeye.clicked.connect(self.mask_eye)
        step2_group_layout.addWidget(self.btn_maskeye)
        
        # Detect Feature button
        self.btn_start_analysis = QPushButton("Detect Feature")
        self.btn_start_analysis.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_start_analysis.clicked.connect(self.detect_feature)
        step2_group_layout.addWidget(self.btn_start_analysis)
        
        left_layout.addWidget(step2_group)
        left_layout.addStretch()
        
        main_layout.addWidget(left_panel)
        
        # Right panel with image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel("Pre-Operative Image")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(title_label)
        
        # Create scrollable area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        
        self.preop_canvas = ImageCanvas()
        # Disable all overlays for Tab 2 (Pre-op Image) - no circles or lines
        self.preop_canvas.show_ref_line = False
        self.preop_canvas.show_toric_axis = False
        self.preop_canvas.show_incision_axis = False
        self.preop_canvas.show_iris_box = False
        scroll.setWidget(self.preop_canvas)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        return widget
    
    def create_live_tracking_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top section with controls and video
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setSpacing(0)
        
        # Left control panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(380)
        left_panel.setStyleSheet("background-color: #f5f5f5;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)  # Reduced spacing for better control
        
        # Step 0: Select Input Source
        step0_label = QLabel("Step 0: Select input source for tracking")
        step0_label.setFont(QFont("Arial", 11, QFont.Bold))
        step0_label.setStyleSheet("color: #2196F3;")
        left_layout.addWidget(step0_label)
        
        self.btn_select_video = QPushButton("Select Video/Camera Input")
        self.btn_select_video.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_select_video.clicked.connect(self.select_video_input)
        left_layout.addWidget(self.btn_select_video)
        
        # Step 1: Select Live Tracking Camera
        step1_label = QLabel("Step 1: Select Live Tracking Camera")
        step1_label.setFont(QFont("Arial", 11, QFont.Bold))
        step1_label.setStyleSheet("color: #2196F3; margin-top: 10px;")
        left_layout.addWidget(step1_label)
        
        # Detect available cameras and create dropdown
        from main import detect_available_cameras
        available_cameras = detect_available_cameras()
        
        if not available_cameras:
            # No cameras detected, show warning
            camera_warning = QLabel("No cameras detected. Please connect a camera.")
            camera_warning.setStyleSheet("color: #f44336; font-size: 10px;")
            left_layout.addWidget(camera_warning)
            available_cameras = [0]  # Default to camera 0 even if not detected
        
        self.camera_input = QComboBox()
        self.camera_input.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
        """)
        
        # Populate dropdown with available cameras
        for cam_idx in available_cameras:
            self.camera_input.addItem(f"Camera {cam_idx}", cam_idx)
        
        # Set default to first available camera
        if available_cameras:
            self.camera_input.setCurrentIndex(0)
        
        left_layout.addWidget(self.camera_input)
        
        self.video_source_label = QLabel("Input Source: File - 251010-1708.mp4")
        self.video_source_label.setStyleSheet("color: #666; font-size: 10px;")
        self.video_source_label.setWordWrap(True)
        left_layout.addWidget(self.video_source_label)
        
        # Optional: Testing Live Tracking Camera
        test_label = QLabel("Optional: Testing Live Tracking Camera")
        test_label.setFont(QFont("Arial", 10, QFont.Bold))
        test_label.setStyleSheet("color: #2196F3; margin-top: 10px;")
        left_layout.addWidget(test_label)
        
        self.btn_test_camera = QPushButton("Test Tracking Camera (3 min)")
        self.btn_test_camera.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        left_layout.addWidget(self.btn_test_camera)
        
        # Step 2: Tracking Control
        step2_label = QLabel("Step 2: Tracking Control")
        step2_label.setFont(QFont("Arial", 11, QFont.Bold))
        step2_label.setStyleSheet("color: #2196F3; margin-top: 15px;")
        left_layout.addWidget(step2_label)
        
        # Create a container widget for tracking buttons with proper spacing
        tracking_container = QWidget()
        tracking_buttons_layout = QVBoxLayout(tracking_container)
        tracking_buttons_layout.setContentsMargins(0, 5, 0, 5)
        tracking_buttons_layout.setSpacing(8)  # Clear spacing between buttons
        
        self.btn_start_tracking = QPushButton("Start Tracking")
        self.btn_start_tracking.setFixedHeight(35)  # Set fixed height
        self.btn_start_tracking.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_start_tracking.clicked.connect(self.start_tracking)
        
        self.btn_pause_tracking = QPushButton("Pause Tracking")
        self.btn_pause_tracking.setFixedHeight(35)  # Set fixed height
        self.btn_pause_tracking.setStyleSheet(self.btn_start_tracking.styleSheet())
        self.btn_pause_tracking.clicked.connect(self.pause_tracking)
        
        self.btn_stop_tracking = QPushButton("Stop Tracking")
        self.btn_stop_tracking.setFixedHeight(35)  # Set fixed height
        self.btn_stop_tracking.setStyleSheet(self.btn_start_tracking.styleSheet())
        self.btn_stop_tracking.clicked.connect(self.stop_tracking)
        
        tracking_buttons_layout.addWidget(self.btn_start_tracking)
        tracking_buttons_layout.addWidget(self.btn_pause_tracking)
        tracking_buttons_layout.addWidget(self.btn_stop_tracking)
        
        left_layout.addWidget(tracking_container)
        
        self.tracking_status_label = QLabel("Video tracking finished.")
        self.tracking_status_label.setStyleSheet("color: #666; font-size: 10px; margin-top: 5px;")
        self.tracking_status_label.setWordWrap(True)
        left_layout.addWidget(self.tracking_status_label)
        
        # Step 3: Visualization axis Help
        step3_label = QLabel("Step 3: Visualization axis Help")
        step3_label.setFont(QFont("Arial", 11, QFont.Bold))
        step3_label.setStyleSheet("color: #2196F3; margin-top: 15px;")
        left_layout.addWidget(step3_label)
        
        # Create container for checkboxes
        checkbox_container = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_container)
        checkbox_layout.setContentsMargins(0, 5, 0, 5)
        checkbox_layout.setSpacing(6)
        
        self.check_ref_lines_tracking = QCheckBox("Show Reference Line")
        self.check_ref_lines_tracking.setChecked(True)
        
        self.check_toric_tracking = QCheckBox("Show Toric Axis")
        self.check_toric_tracking.setChecked(True)
        
        self.check_iris_ring = QCheckBox("Show Limbus Circle")
        self.check_iris_ring.setChecked(True)
        
        self.check_incision_tracking = QCheckBox("Show Incision Axis")
        self.check_incision_tracking.setChecked(True)
        
        # Create checkbox state references for dynamic updates during tracking
        # Initialize with current checkbox states
        self.tracking_show_reference_ref = [self.check_ref_lines_tracking.isChecked()]
        self.tracking_show_toric_ref = [self.check_toric_tracking.isChecked()]
        self.tracking_show_incision_ref = [self.check_incision_tracking.isChecked()]
        self.tracking_show_limbus_ref = [self.check_iris_ring.isChecked()]
        
        # Connect checkboxes to update state references
        self.check_ref_lines_tracking.stateChanged.connect(
            lambda state: self._update_tracking_checkbox('reference', state)
        )
        self.check_toric_tracking.stateChanged.connect(
            lambda state: self._update_tracking_checkbox('toric', state)
        )
        self.check_incision_tracking.stateChanged.connect(
            lambda state: self._update_tracking_checkbox('incision', state)
        )
        self.check_iris_ring.stateChanged.connect(
            lambda state: self._update_tracking_checkbox('limbus', state)
        )
        
        checkbox_layout.addWidget(self.check_ref_lines_tracking)
        checkbox_layout.addWidget(self.check_toric_tracking)
        checkbox_layout.addWidget(self.check_iris_ring)
        checkbox_layout.addWidget(self.check_incision_tracking)
        
        left_layout.addWidget(checkbox_container)
        
        self.rotation_label = QLabel("Current Angle of rotation: +0.0 deg")
        self.rotation_label.setStyleSheet("color: #666; font-size: 10px; margin-top: 8px;")
        left_layout.addWidget(self.rotation_label)
        
        # Manual Iris Radius
        radius_label = QLabel("Manual Iris Radius (Tracking)")
        radius_label.setFont(QFont("Arial", 10, QFont.Bold))
        radius_label.setStyleSheet("color: #2196F3; margin-top: 12px;")
        left_layout.addWidget(radius_label)
        
        radius_layout = QHBoxLayout()
        radius_layout.setSpacing(8)
        
        self.radius_input = QLineEdit()
        self.radius_input.setPlaceholderText("Enter radius in pixels")
        self.radius_input.setFixedHeight(32)
        self.radius_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background: white;
            }
        """)
        
        self.btn_apply_radius = QPushButton("Apply Tracking Radius")
        self.btn_apply_radius.setFixedHeight(32)
        self.btn_apply_radius.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        radius_layout.addWidget(self.radius_input, stretch=2)
        radius_layout.addWidget(self.btn_apply_radius, stretch=1)
        left_layout.addLayout(radius_layout)
        
        left_layout.addStretch()
        
        top_layout.addWidget(left_panel)
        
        # Right video display area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Video display (with overlays)
        self.video_display = QLabel()
        self.video_display.setStyleSheet("background-color: black; border: 1px solid #ccc;")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(800, 450)
        # Enable keyboard focus for video control (q to quit, p to pause)
        self.video_display.setFocusPolicy(Qt.StrongFocus)
        right_layout.addWidget(self.video_display, stretch=1)
        
        top_layout.addWidget(right_panel, stretch=1)
        
        main_layout.addWidget(top_section, stretch=2)
        
        return widget

    def create_axis_setup_tab(self):
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setSpacing(0)
        
        # Left panel with controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(450)
        left_panel.setStyleSheet("background-color: #f5f5f5;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Step 1: Set Reference Axis
        step1_group = QGroupBox("Step 1: Set Reference Axis")
        step1_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2196F3;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        step1_layout = QFormLayout()
        
        self.ref_angle_input = QLineEdit("0")
        step1_layout.addRow("Ref Axis Angle (Calibrate):", self.ref_angle_input)
        # Connect to update display in real-time
        self.ref_angle_input.textChanged.connect(self.update_axis_display_live)
        
        self.manual_y_offset = QLineEdit("0")
        step1_layout.addRow("Manual Y-axis Offset (Pre-op):", self.manual_y_offset)
        # Connect to update display in real-time
        self.manual_y_offset.textChanged.connect(self.update_axis_display_live)
        
        self.manual_x_offset = QLineEdit("0")
        step1_layout.addRow("Manual X-axis Offset (Pre-op):", self.manual_x_offset)
        # Connect to update display in real-time
        self.manual_x_offset.textChanged.connect(self.update_axis_display_live)
        
        set_ref_btn = QPushButton("Set Reference Axis from Pre-op Image")
        set_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #9e9e9e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 3px;
            }
        """)
        step1_layout.addRow(set_ref_btn)
        
        self.ref_status_label = QLabel("Reference Axis Set (angle for display: 0.0 deg). Ready for tracking or comparison.")
        self.ref_status_label.setWordWrap(True)
        self.ref_status_label.setStyleSheet("color: #666; font-size: 10px;")
        step1_layout.addRow(self.ref_status_label)
        
        step1_group.setLayout(step1_layout)
        left_layout.addWidget(step1_group)
        
        # Step 2: Set Toric Axis
        step2_group = QGroupBox("Step 2: Set Toric Axis")
        step2_group.setStyleSheet(step1_group.styleSheet())
        step2_layout = QFormLayout()
        
        self.toric_angle_input = QLineEdit("0")
        # Connect to update display in real-time
        self.toric_angle_input.textChanged.connect(self.update_axis_display_live)
        toric_hbox = QHBoxLayout()
        toric_hbox.addWidget(self.toric_angle_input)
        set_toric_btn = QPushButton("Set Toric Axis")
        set_toric_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        set_toric_btn.clicked.connect(self.set_toric_axis)
        toric_hbox.addWidget(set_toric_btn)
        
        step2_layout.addRow("Toric Axis Angle (degrees):", toric_hbox)
        
        self.toric_status = QLabel("Toric Axis Set to 0.0 degrees.")
        self.toric_status.setWordWrap(True)
        self.toric_status.setStyleSheet("color: #666; font-size: 10px;")
        step2_layout.addRow(self.toric_status)
        
        step2_group.setLayout(step2_layout)
        left_layout.addWidget(step2_group)
        
        # Step 3: Set Incision Axis
        step3_group = QGroupBox("Step 3: Set Incision Axis")
        step3_group.setStyleSheet(step1_group.styleSheet())
        step3_layout = QFormLayout()
        
        self.incision_angle_input = QLineEdit("0")
        # Connect to update display in real-time
        self.incision_angle_input.textChanged.connect(self.update_axis_display_live)
        incision_hbox = QHBoxLayout()
        incision_hbox.addWidget(self.incision_angle_input)
        set_incision_btn = QPushButton("Set Incision Axis")
        set_incision_btn.setStyleSheet(set_toric_btn.styleSheet())
        set_incision_btn.clicked.connect(self.set_incision_axis)
        incision_hbox.addWidget(set_incision_btn)
        
        step3_layout.addRow("Incision Axis Angle (degrees):", incision_hbox)
        
        self.incision_status = QLabel("Incision Axis Set to 0.0°")
        self.incision_status.setWordWrap(True)
        self.incision_status.setStyleSheet("color: #666; font-size: 10px;")
        step3_layout.addRow(self.incision_status)
        
        step3_group.setLayout(step3_layout)
        left_layout.addWidget(step3_group)
        
        # Step 4: Visualization Overlays
        step4_group = QGroupBox("Step 4: Visualization Overlays")
        step4_group.setStyleSheet(step1_group.styleSheet())
        step4_layout = QVBoxLayout()
        
        self.check_ref_line = QCheckBox("Show Reference Line")
        self.check_ref_line.setChecked(True)
        self.check_ref_line.stateChanged.connect(self.update_canvas)
        step4_layout.addWidget(self.check_ref_line)
        
        self.check_toric = QCheckBox("Show Toric Axis")
        self.check_toric.setChecked(True)
        self.check_toric.stateChanged.connect(self.update_canvas)
        step4_layout.addWidget(self.check_toric)
        
        self.check_incision = QCheckBox("Show Incision Axis")
        self.check_incision.setChecked(True)
        self.check_incision.stateChanged.connect(self.update_canvas)
        step4_layout.addWidget(self.check_incision)
        
        self.check_iris = QCheckBox("Show Iris Bounding Box (Green)")
        self.check_iris.setChecked(True)
        self.check_iris.stateChanged.connect(self.update_canvas)
        step4_layout.addWidget(self.check_iris)
        
        step4_group.setLayout(step4_layout)
        left_layout.addWidget(step4_group)
        
        # Submit button
        submit_btn = QPushButton("Submit All Axes")
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        submit_btn.clicked.connect(self.submit_all_axes)
        left_layout.addWidget(submit_btn)
        left_layout.addStretch()
        
        main_layout.addWidget(left_panel)
        
        # Right panel with image
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel("Pre-Operative Image")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(title_label)
        
        # Create scrollable area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        self.canvas = ImageCanvas()
        scroll.setWidget(self.canvas)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        return widget
    
    def load_preop_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-op Model", "", "PyTorch Models (*.pt)"
        )
        if file_path:
            # Update handler with new path
            self.config_handler.set_preop_model_path(file_path)
            
            # Update UI
            file_name = os.path.basename(file_path)
            self.label_preop_status.setText(f"Pre-op Model: {file_name}")
            
            print(f"[UI] Pre-op model path updated: {file_path}")
            print(f"[UI] Handler now has: {self.config_handler.get_preop_model_path()}")
    
    def load_intraop_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Intra-op Model", "", "PyTorch Models (*.pt)"
        )
        if file_path:
            # Update handler with new path
            self.config_handler.set_intraop_model_path(file_path)
            
            # Update UI
            file_name = os.path.basename(file_path)
            self.label_intraop_status.setText(f"Intra-op Model: {file_name}")
            
            print(f"[UI] Intra-op model path updated: {file_path}")
            print(f"[UI] Handler now has: {self.config_handler.get_intraop_model_path()}")
    
    def submit_configuration(self):
        """Submit configuration - update handler with all UI values"""
        try:
            # Delete output folder before submitting
            self.delete_output_folder()
            
            # Update app mode
            if self.radio_normal.isChecked():
                self.config_handler.set_app_mode('normal')
            else:
                self.config_handler.set_app_mode('demo')
            
            # Update YOLO confidence
            try:
                yolo_conf = float(self.confidence_input.text())
                self.config_handler.set_yolo_confidence(yolo_conf)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "YOLO Confidence must be a number between 0.0 and 1.0")
                return
            
            # Update matching confidence threshold
            try:
                match_conf = float(self.matching_confidence_input.text())
                self.config_handler.set_matching_confidence_threshold(match_conf)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Matching Confidence Threshold must be a number between 0.0 and 1.0")
                return
            
            # Update freeze confidence threshold
            try:
                freeze_conf = float(self.freeze_confidence_input.text())
                self.config_handler.set_freeze_confidence_threshold(freeze_conf)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Freeze Confidence Threshold must be a number between 0.0 and 1.0")
                return
            
            # Mark configuration as submitted
            self.config_handler.set_config_submitted(True)
            
            # Print confirmation
            print("\n" + "="*50)
            print("Configuration Submitted from UI:")
            print("="*50)
            print(f"Mode: {'Normal' if self.radio_normal.isChecked() else 'Demo'}")
            print(f"YOLO Confidence: {self.config_handler.get_yolo_confidence()}")
            print(f"Matching Confidence Threshold: {self.config_handler.get_matching_confidence_threshold()}")
            print(f"Freeze Confidence Threshold: {self.config_handler.get_freeze_confidence_threshold()}")
            print(f"Freeze Confidence Threshold: {self.config_handler.get_freeze_confidence_threshold()}")
            print(f"Pre-op model: {self.config_handler.get_preop_model_path()}")
            print(f"Intra-op model: {self.config_handler.get_intraop_model_path()}")
            print("="*50 + "\n")
            
            # Show success message
            QMessageBox.information(self, "Configuration Saved", 
                                  "Configuration has been saved successfully!\n\n"
                                  "The main pipeline will use these settings when running.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit configuration: {str(e)}")
            print(f"[ERROR] Failed to submit configuration: {e}")
    
    def load_preop_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-operative Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            # Save path to handler
            self.config_handler.set_preop_image_path(file_path)
            self.current_preop_image_path = file_path
            
            # Load and display original image
            self.current_preop_image = QPixmap(file_path)
            self.rotation_angle = 0
            self.preop_canvas.set_image(self.current_preop_image)
            self.canvas.set_image(self.current_preop_image)
            
            # Update status label
            file_name = os.path.basename(file_path)
            self.preop_status_label.setText(
                f"Pre-op image loaded: {file_name}. Ready to preprocess..."
            )
            print(f"[UI] Loaded pre-op image: {file_path}")
            print(f"[UI] Image path saved to handler")
    
    def preprocess_preop_image(self):
        """Preprocess the pre-op image using current parameters"""
        if not self.current_preop_image_path:
            QMessageBox.warning(self, "No Image", "Please load a pre-op image first.")
            return
        
        # Check if YOLO model is loaded
        preop_model_path = self.config_handler.get_preop_model_path()
        if not preop_model_path or not os.path.exists(preop_model_path):
            QMessageBox.warning(self, "Model Missing", 
                              "Please load a pre-op YOLO model in Tab 1 (Configuration) first.")
            return
        
        try:
            # Update status
            self.preop_status_label.setText("Preprocessing image... Please wait.")
            self.btn_start_analysis.setEnabled(False)
            self.btn_preprocess_image.setEnabled(False)
            QApplication.processEvents()  # Update UI
            
            # Load YOLO model if not already loaded
            if self.preop_yolo_model is None:
                print(f"[UI] Loading YOLO model: {preop_model_path}")
                self.preop_yolo_model = load_yolo_model(preop_model_path)
            
            # Get parameters from handler (or UI inputs if changed)
            self._save_preprocessing_params_to_handler()
            
            inner_exclude = self.config_handler.get_inner_exclude_ratio()
            outer_include = self.config_handler.get_outer_include_ratio()
            crop_width = self.config_handler.get_crop_width_ratio()
            crop_height = self.config_handler.get_crop_height_ratio()
            eyelid_upper = self.config_handler.get_eyelid_trim_upper_ratio()
            eyelid_lower = self.config_handler.get_eyelid_trim_lower_ratio()
            yolo_conf = self.config_handler.get_yolo_confidence()
            
            # Create output directory
            output_dir = "output/ui_preprocess"
            os.makedirs(output_dir, exist_ok=True)
            
            # Preprocess image
            print(f"[UI] Starting preprocessing with parameters:")
            print(f"  Inner exclude: {inner_exclude}")
            print(f"  Outer include: {outer_include} (stored but not used by preprocess)")
            print(f"  Crop width: {crop_width}, height: {crop_height} (stored but not used by preprocess)")
            print(f"  Eyelid upper: {eyelid_upper}, lower: {eyelid_lower}")
            print(f"  YOLO confidence: {yolo_conf} (stored but not used by preprocess)")
            
            # Note: preprocess_single only accepts inner_exclude_ratio, eyelid ratios, and trim_eyelids
            # Other parameters (outer_include, crop ratios, yolo_conf) are stored in handler for other uses
            preop_result = preprocess_single(
                self.current_preop_image_path,
                self.preop_yolo_model,
                ImageType.PREOP,
                output_dir,
                reference_image=None,
                apply_histogram_match=False,
                trim_eyelids=True,  # Enable eyelid trimming based on UI parameters
                eyelid_upper_ratio=eyelid_upper,
                eyelid_lower_ratio=eyelid_lower,
                inner_exclude_ratio=inner_exclude,
                verbose=True
            )
            
            # Store result in handler
            self.config_handler.set_preop_result(preop_result)
            self.preprocessed_preop_image = preop_result.processed_image
            
            # Convert to QPixmap and display
            height, width = preop_result.processed_image.shape[:2]
            if len(preop_result.processed_image.shape) == 2:
                # Grayscale
                q_image = QImage(preop_result.processed_image.data, width, height, 
                               preop_result.processed_image.strides[0], QImage.Format_Grayscale8)
            else:
                # BGR to RGB
                rgb_image = cv2.cvtColor(preop_result.processed_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, 
                               rgb_image.strides[0], QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            self.preop_canvas.set_image(pixmap)
            self.canvas.set_image(pixmap)
            
            # Update status
            file_name = os.path.basename(self.current_preop_image_path)
            self.preop_status_label.setText(
                f"✓ Preprocessed: {file_name}\n"
                f"Limbus center: {preop_result.limbus_info.center}, "
                f"radius: {preop_result.limbus_info.radius}"
            )
            
            print(f"[UI] Preprocessing complete!")
            print(f"[UI] Limbus center: {preop_result.limbus_info.center}")
            print(f"[UI] Limbus radius: {preop_result.limbus_info.radius}")
            
            QMessageBox.information(self, "Preprocessing Complete", 
                                  f"Image preprocessed successfully!\n\n"
                                  f"Limbus detected at center: {preop_result.limbus_info.center}\n"
                                  f"Radius: {preop_result.limbus_info.radius} pixels")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[UI ERROR] Preprocessing failed: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Preprocessing Error", 
                              f"Failed to preprocess image:\n{error_msg}")
            self.preop_status_label.setText(f"Error: {error_msg}")
        finally:
            self.btn_start_analysis.setEnabled(True)
            self.btn_preprocess_image.setEnabled(True)
    
    def mask_eye(self):
        """Handle mask eye button click - triggers preprocessing again"""
        if not self.current_preop_image_path:
            QMessageBox.warning(self, "No Image", "Please load a pre-op image first.")
            return
        
        # Trigger preprocessing again
        print(f"[UI] Mask Eye: Triggering preprocessing again...")
        self.preprocess_preop_image()
    
    def add_mask(self):
        """Handle add mask button click - opens cropped image and enables live mask updates"""
        if not self.current_preop_image_path:
            QMessageBox.warning(self, "No Image", "Please load a pre-op image first.")
            return
        
        # Check if preprocessing has been done
        preop_result = self.config_handler.get_preop_result()
        if preop_result is None:
            QMessageBox.warning(self, "No Preprocessing", 
                              "Please preprocess the image first before adding mask.")
            return
        
        # Check if YOLO model is loaded
        if self.preop_yolo_model is None:
            preop_model_path = self.config_handler.get_preop_model_path()
            if not preop_model_path or not os.path.exists(preop_model_path):
                QMessageBox.warning(self, "Model Missing", 
                                  "Please load a pre-op YOLO model in Tab 1 (Configuration) first.")
                return
            self.preop_yolo_model = load_yolo_model(preop_model_path)
        
        try:
            # Get the cropped image from preprocessing result
            self.cropped_image_for_mask = preop_result.original_image.copy()
            
            # Detect limbus in the cropped image
            # Note: YOLO confidence is set when loading the model, not passed to detect_limbus
            self.limbus_info_for_mask = detect_limbus(
                self.preop_yolo_model, 
                self.cropped_image_for_mask
            )
            
            if self.limbus_info_for_mask is None:
                QMessageBox.warning(self, "Limbus Detection Failed", 
                                  "Could not detect limbus in cropped image. Using stored limbus info.")
                # Fallback to stored limbus info
                self.limbus_info_for_mask = preop_result.limbus_info
            else:
                print(f"[UI] Add Mask: Limbus detected at center={self.limbus_info_for_mask.center}, radius={self.limbus_info_for_mask.radius}")
            
            # Enable live mask mode
            self.mask_live_mode = True
            
            # Update mask display with current parameters
            self.update_mask_live()
            
            # Update status
            file_name = os.path.basename(self.current_preop_image_path)
            self.preop_status_label.setText(
                f"✓ Add Mask: Live mask mode enabled. Adjust parameters to see changes. ({file_name})"
            )
            
            print(f"[UI] Add Mask: Live mask mode enabled")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[UI ERROR] Add Mask failed: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Add Mask Error", 
                              f"Failed to add mask:\n{error_msg}")
            self.preop_status_label.setText(f"Error: {error_msg}")
            self.mask_live_mode = False
    
    def update_mask_live(self):
        """Update mask display in real-time based on current parameter values"""
        if not self.mask_live_mode or self.cropped_image_for_mask is None or self.limbus_info_for_mask is None:
            return
        
        try:
            # Get current parameter values
            inner_exclude = float(self.inner_exclude_ratio_input.text())
            outer_include = float(self.outer_include_ratio_input.text())
            eyelid_upper = float(self.eyelid_trim_upper_ratio_input.text())
            eyelid_lower = float(self.eyelid_trim_lower_ratio_input.text())
            
            # Create ring mask with current parameters
            ring_mask = create_ring_mask(
                self.cropped_image_for_mask.shape[:2],
                self.limbus_info_for_mask.center,
                self.limbus_info_for_mask.radius,
                inner_ratio=inner_exclude,
                outer_ratio=outer_include
            )
            
            # Apply eyelid trimming if needed
            ring_mask = trim_eyelids_from_mask(
                ring_mask,
                self.limbus_info_for_mask.center,
                self.limbus_info_for_mask.radius,
                upper_ratio=eyelid_upper,
                lower_ratio=eyelid_lower
            )
            
            # Apply mask to cropped image
            masked_image = apply_mask(self.cropped_image_for_mask, ring_mask)
            
            # Draw limbus center and radius on the image for visualization
            vis_image = masked_image.copy()
            cv2.circle(vis_image, self.limbus_info_for_mask.center, self.limbus_info_for_mask.radius, (0, 255, 0), 2)
            # Draw center point proportional to limbus radius (2.5% of radius, minimum 1 pixel)
            center_point_radius = max(1, int(self.limbus_info_for_mask.radius * 0.025))
            cv2.circle(vis_image, self.limbus_info_for_mask.center, center_point_radius, (0, 0, 255), -1)
            
            # Convert to QPixmap and display
            height, width = vis_image.shape[:2]
            if len(vis_image.shape) == 2:
                # Grayscale
                q_image = QImage(vis_image.data, width, height, 
                               vis_image.strides[0], QImage.Format_Grayscale8)
            else:
                # BGR to RGB
                rgb_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, 
                               rgb_image.strides[0], QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            self.preop_canvas.set_image(pixmap)
            self.canvas.set_image(pixmap)
            
        except Exception as e:
            # Silently fail during live updates to avoid spam
            print(f"[UI] Update mask live error: {e}")
    
    def detect_feature(self):
        """Handle detect feature button click - performs SuperPoint feature detection"""
        if not self.current_preop_image_path:
            QMessageBox.warning(self, "No Image", "Please load a pre-op image first.")
            return
        
        # Check if preprocessing has been done
        preop_result = self.config_handler.get_preop_result()
        if preop_result is None:
            QMessageBox.warning(self, "No Preprocessing", 
                              "Please preprocess the image first before detecting features.")
            return
        
        try:
            # Update status
            self.preop_status_label.setText("Initializing SuperPoint models... Please wait.")
            self.btn_start_analysis.setEnabled(False)
            QApplication.processEvents()  # Update UI
            
            # Initialize SuperPoint models if not already initialized
            if self.feature_extractor is None:
                print("[UI] Initializing SuperPoint models...")
                self.feature_config = AccuracyConfig()
                
                # Update config with matching confidence threshold from handler
                matching_conf_threshold = self.config_handler.get_matching_confidence_threshold()
                self.feature_config.MIN_MATCH_CONFIDENCE = matching_conf_threshold
                print(f"[UI] Using matching confidence threshold: {matching_conf_threshold}")
                
                self.feature_extractor, self.feature_matcher, self.feature_device = initialize_models(self.feature_config)
                print("[UI] SuperPoint models initialized")
            
            # Save processed image temporarily for feature extraction
            output_dir = "output/ui_preprocess"
            os.makedirs(output_dir, exist_ok=True)
            temp_image_path = os.path.join(output_dir, "temp_preop_for_features.jpg")
            cv2.imwrite(temp_image_path, preop_result.processed_image)
            
            # Update status
            self.preop_status_label.setText("Extracting features with SuperPoint... Please wait.")
            QApplication.processEvents()
            
            # Extract features from preprocessed preop image
            print("[UI] Extracting features from preop image...")
            self.preop_features = extract_features(
                temp_image_path,
                self.feature_extractor,
                self.feature_device,
                mask=preop_result.ring_mask,
                config=self.feature_config
            )
            
            num_keypoints = self.preop_features['keypoints'].shape[1]
            print(f"[UI] Features extracted: {num_keypoints} keypoints")
            
            # Save keypoints visualization
            self.preop_status_label.setText("Saving keypoints visualization...")
            QApplication.processEvents()
            
            try:
                # Convert keypoints to numpy if needed
                kpts = self.preop_features["keypoints"][0]
                if torch.is_tensor(kpts):
                    kpts = kpts.detach().cpu().numpy()
                
                # Save visualization
                features_output_dir = "output/ui_features"
                os.makedirs(features_output_dir, exist_ok=True)
                keypoints_image_path = os.path.join(features_output_dir, "preop_keypoints.jpg")
                
                save_keypoints_image(
                    temp_image_path,
                    kpts,
                    keypoints_image_path,
                    center=preop_result.limbus_info.center,
                    radius=preop_result.limbus_info.radius,
                    color=(0, 255, 0),
                )
                
                # Load and display the keypoints visualization
                keypoints_image = cv2.imread(keypoints_image_path)
                if keypoints_image is not None:
                    height, width = keypoints_image.shape[:2]
                    rgb_image = cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB)
                    q_image = QImage(rgb_image.data, width, height, 
                                   rgb_image.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.preop_canvas.set_image(pixmap)
                    self.canvas.set_image(pixmap)
                    
                    print(f"[UI] Keypoints visualization saved: {keypoints_image_path}")
                else:
                    # Fallback: display processed image if visualization failed
                    feature_image = preop_result.processed_image.copy()
                    height, width = feature_image.shape[:2]
                    rgb_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB)
                    q_image = QImage(rgb_image.data, width, height, 
                                   rgb_image.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.preop_canvas.set_image(pixmap)
                    self.canvas.set_image(pixmap)
                    
            except Exception as e:
                print(f"[UI WARNING] Failed to save keypoints visualization: {e}")
                # Fallback: display processed image
                feature_image = preop_result.processed_image.copy()
                height, width = feature_image.shape[:2]
                rgb_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, 
                               rgb_image.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.preop_canvas.set_image(pixmap)
                self.canvas.set_image(pixmap)
            
            # Clean up temp file
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            # Update status
            file_name = os.path.basename(self.current_preop_image_path)
            self.preop_status_label.setText(
                f"✓ Feature Detection Complete: {num_keypoints} keypoints detected ({file_name})"
            )
            
            print(f"[UI] Detect Feature: SuperPoint feature detection completed successfully")
            print(f"[UI] Keypoints: {num_keypoints}")
            
            QMessageBox.information(self, "Feature Detection Complete", 
                                  f"SuperPoint feature detection completed successfully!\n\n"
                                  f"Keypoints detected: {num_keypoints}\n"
                                  f"Features are ready for matching.")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[UI ERROR] Detect Feature failed: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Detect Feature Error", 
                              f"Failed to detect features:\n{error_msg}")
            self.preop_status_label.setText(f"Error: {error_msg}")
        finally:
            self.btn_start_analysis.setEnabled(True)
    
    def _save_preprocessing_params_to_handler(self):
        """Save current UI parameter values to handler"""
        try:
            inner_exclude = float(self.inner_exclude_ratio_input.text())
            outer_include = float(self.outer_include_ratio_input.text())
            crop_width = float(self.crop_width_ratio_input.text())
            crop_height = float(self.crop_height_ratio_input.text())
            eyelid_upper = float(self.eyelid_trim_upper_ratio_input.text())
            eyelid_lower = float(self.eyelid_trim_lower_ratio_input.text())
            
            # Save to handler
            self.config_handler.set_inner_exclude_ratio(inner_exclude)
            self.config_handler.set_outer_include_ratio(outer_include)
            self.config_handler.set_crop_width_ratio(crop_width)
            self.config_handler.set_crop_height_ratio(crop_height)
            self.config_handler.set_eyelid_trim_upper_ratio(eyelid_upper)
            self.config_handler.set_eyelid_trim_lower_ratio(eyelid_lower)
            
            print(f"[UI] Parameters saved to handler: inner={inner_exclude:.2f}, outer={outer_include:.2f}, "
                  f"crop={crop_width:.2f}x{crop_height:.2f}, eyelid={eyelid_upper:.2f}/{eyelid_lower:.2f}")
        except (ValueError, AttributeError) as e:
            # AttributeError can occur if fields don't exist yet
            print(f"[UI WARNING] Could not save parameter to handler: {e}")
    
    def rotate_image(self):
        if self.current_preop_image:
            self.rotation_angle = (self.rotation_angle + 90) % 360
            
            # Create a transform for rotation
            transform = self.current_preop_image.transformed(
                self.current_preop_image.transform().rotate(90)
            )
            self.current_preop_image = transform
            
            # Update both canvases
            self.preop_canvas.set_image(self.current_preop_image)
            self.canvas.set_image(self.current_preop_image)
            
            print(f"Rotated image to {self.rotation_angle}°")
    
    def increment_value(self, input_field, min_val=0.0, max_val=10.0):
        """Increment value in input field by 0.05"""
        try:
            current_val = float(input_field.text())
            new_val = min(max_val, current_val + 0.05)
            input_field.setText(f"{new_val:.2f}")
            # Save to handler
            self._save_preprocessing_params_to_handler()
            # Trigger live mask update if in mask mode
            if self.mask_live_mode:
                self.update_mask_live()
        except ValueError:
            input_field.setText("0.00")
    
    def decrement_value(self, input_field, min_val=0.0, max_val=10.0):
        """Decrement value in input field by 0.05"""
        try:
            current_val = float(input_field.text())
            new_val = max(min_val, current_val - 0.05)
            input_field.setText(f"{new_val:.2f}")
            # Save to handler
            self._save_preprocessing_params_to_handler()
            # Trigger live mask update if in mask mode
            if self.mask_live_mode:
                self.update_mask_live()
        except ValueError:
            input_field.setText("0.00")
    
    def create_arrow_buttons(self, input_field, min_val=0.0, max_val=10.0):
        """Create up and down arrow buttons for a value input field"""
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(2)
        
        btn_up = QPushButton("▲")
        btn_up.setFixedSize(25, 15)
        btn_up.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #ccc;
                border-radius: 2px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #b0b0b0;
            }
        """)
        btn_up.clicked.connect(lambda: self.increment_value(input_field, min_val, max_val))
        
        btn_down = QPushButton("▼")
        btn_down.setFixedSize(25, 15)
        btn_down.setStyleSheet(btn_up.styleSheet())
        btn_down.clicked.connect(lambda: self.decrement_value(input_field, min_val, max_val))
        
        button_layout.addWidget(btn_up)
        button_layout.addWidget(btn_down)
        
        return button_container
    
    def set_toric_axis(self):
        angle = self.toric_angle_input.text()
        self.toric_status.setText(f"Toric Axis Set to {angle} degrees.")
        print(f"Toric axis set to: {angle}")
    
    def set_incision_axis(self):
        angle = self.incision_angle_input.text()
        self.incision_status.setText(f"Incision Axis Set to {angle}°")
        print(f"Incision axis set to: {angle}")
    
    def select_video_input(self):
        """Open popup dialog to choose between video file or live camera"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Input Source")
        dialog.setModal(True)
        dialog.setFixedSize(350, 280)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title label
        title_label = QLabel("Choose Input Source:")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Button layout
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        # Load Video File button
        btn_load_video = QPushButton("Load Video File")
        btn_load_video.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        btn_load_video.clicked.connect(lambda: self.load_video_file(dialog))
        button_layout.addWidget(btn_load_video)
        
        # Use Live Camera button
        btn_live_camera = QPushButton("Use Live Camera")
        btn_live_camera.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_live_camera.clicked.connect(lambda: self.open_live_camera(dialog))
        button_layout.addWidget(btn_live_camera)
        
        layout.addLayout(button_layout)
        
        # Cancel button
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #9e9e9e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        btn_cancel.clicked.connect(dialog.reject)
        layout.addWidget(btn_cancel)
        
        dialog.exec_()
    
    def load_video_file(self, dialog):
        """Open file browser to select video file"""
        dialog.accept()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if file_path:
            self.tracking_video_path = file_path
            file_name = os.path.basename(file_path)
            self.video_source_label.setText(f"Input Source: File - {file_name}")
            # Save video path to handler for main.py
            self.config_handler.set_intraop_video_path(file_path)
            print(f"Selected video: {file_path}")
            print(f"[UI] Video path saved to handler for main.py: {file_path}")
    
    def open_live_camera(self, dialog):
        """Open live camera feed preview (like browsing a video file)"""
        dialog.accept()
        # Get camera index from dropdown
        try:
            camera_index = self.camera_input.currentData()
            if camera_index is None:
                # Fallback: extract from text if data not available
                camera_text = self.camera_input.currentText()
                import re
                match = re.search(r'\d+', camera_text)
                camera_index = int(match.group()) if match else 0
        except:
            camera_index = 0
        
        # Import camera preview function from main.py
        from main import open_camera_preview
        
        # Open camera preview window (like browsing a video file)
        success = open_camera_preview(camera_index)
        
        if success:
            # Set video path to None to indicate camera mode
            self.tracking_video_path = None
            # Set handler video path to camera index string for main.py to detect camera mode
            self.config_handler.set_intraop_video_path(f"camera:{camera_index}")
            self.video_source_label.setText(f"Input Source: Live Camera {camera_index}")
            print(f"Using live camera: {camera_index}")
        else:
            QMessageBox.warning(self, "Camera Error", 
                              f"Could not open camera {camera_index}.\n"
                              "Please check if the camera is connected and try a different camera index.")
    
    def _start_intraop_display_thread(self):
        angles = {
            "reference": self.config_handler.get_reference_angle(),
            "toric": self.config_handler.get_toric_angle(),
            "incision": self.config_handler.get_incision_angle(),
        }

        self._stop_intraop_display_thread()
        self.display_frame_queue = queue.Queue(maxsize=5)
        self.intraop_display_thread = IntraopDisplayThread(self.display_frame_queue, angles)
        self.intraop_display_thread.frame_ready.connect(self._update_video_display_pixmap)
        self.intraop_display_thread.rotation_updated.connect(self._update_rotation_label)
        self.intraop_display_thread.start()

    def _stop_intraop_display_thread(self):
        if self.intraop_display_thread:
            try:
                self.intraop_display_thread.stop()
                self.intraop_display_thread.wait(1000)
            except Exception:
                pass
        self.intraop_display_thread = None
        self.display_frame_queue = None

    def _update_video_display_pixmap(self, pixmap: QPixmap):
        scaled_pixmap = pixmap.scaled(
            self.video_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_display.setPixmap(scaled_pixmap)

    def _update_rotation_label(self, rotation_angle: float, confidence: float):
        if self.is_tracking:
            self.rotation_label.setText(
                f"Current Angle of rotation: {rotation_angle:+.2f} deg ({confidence*100:.1f}% conf)"
            )

    def _enqueue_display_frame(self, frame, rotation_angle, confidence) -> bool:
        if self.display_frame_queue is None:
            return False
        try:
            self.display_frame_queue.put_nowait((frame, rotation_angle, confidence))
            return True
        except queue.Full:
            return False
        except Exception:
            return False
    
    def start_tracking(self):
        # Check if video source is selected
        if self.tracking_video_path is None:
            # Check if camera mode - get from handler or input field
            intraop_video_path = self.config_handler.get_intraop_video_path()
            if intraop_video_path and "camera:" in intraop_video_path.lower():
                # Camera mode already set via "Use Live Camera"
                print(f"[TRACKING] Using camera from handler: {intraop_video_path}")
            else:
                # Try to get camera index from dropdown
                try:
                    camera_index = self.camera_input.currentData()
                    if camera_index is None:
                        # Fallback: extract from text if data not available
                        camera_text = self.camera_input.currentText()
                        import re
                        match = re.search(r'\d+', camera_text)
                        camera_index = int(match.group()) if match else 0
                    # Set camera in handler
                    self.config_handler.set_intraop_video_path(f"camera:{camera_index}")
                    print(f"[TRACKING] Using camera from dropdown: {camera_index}")
                except:
                    QMessageBox.warning(self, "No Video Source", 
                                      "Please select a video file or camera before starting tracking.")
                    return
        elif not os.path.exists(self.tracking_video_path):
            QMessageBox.warning(self, "File Error", 
                              f"Video file not found: {self.tracking_video_path}")
            return
        
        # Check if preop is preprocessed
        preop_result = self.config_handler.get_preop_result()
        if preop_result is None:
            QMessageBox.warning(self, "No Pre-op Data", 
                              "Please preprocess the pre-op image first before starting tracking.")
            return
        
        # Import full video processing (spawns live + analysis threads) from main
        from main import process_video_for_ui
        import threading
        
        self.is_tracking = True
        self.tracking_status_label.setText("Tracking started...")
        self.btn_start_tracking.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)
        
        # Create quit flag and pause flag for video processing
        self.tracking_quit_flag = [False]
        self.tracking_pause_flag = [False]
        
        # Set focus to video display for keyboard events
        self.video_display.setFocus()

        # Start display thread to render frames with angle overlays
        self._start_intraop_display_thread()
        
        # Define frame callback to update UI
        def frame_callback(frame, rotation_angle, confidence):
            """Callback to update video display with processed frame"""
            if not self.is_tracking:
                return
            # Push frame to dedicated display thread; fallback to direct draw if full/unavailable
            if self._enqueue_display_frame(frame.copy(), rotation_angle, confidence):
                return

            # Fallback rendering (should rarely be used)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self._update_video_display_pixmap(pixmap)
            self._update_rotation_label(rotation_angle, confidence)
        
        # Start video processing in a thread
        def run_video_processing():
            try:
                # process_video_for_ui spins two threads internally:
                # - main thread for live video display
                # - background analysis thread for intra-op video analysis
                stats_logger = process_video_for_ui(
                    frame_callback=frame_callback, 
                    quit_flag_ref=self.tracking_quit_flag,
                    pause_flag_ref=self.tracking_pause_flag,
                    show_reference=self.check_ref_lines_tracking.isChecked(),
                    show_toric=self.check_toric_tracking.isChecked(),
                    show_incision=self.check_incision_tracking.isChecked(),
                    show_limbus=self.check_iris_ring.isChecked(),
                    show_reference_ref=self.tracking_show_reference_ref,
                    show_toric_ref=self.tracking_show_toric_ref,
                    show_incision_ref=self.tracking_show_incision_ref,
                    show_limbus_ref=self.tracking_show_limbus_ref
                )
                # Store stats logger for later saving
                self.tracking_stats_logger = stats_logger
                # Save statistics when finished (if not already saved by stop_tracking)
                if self.is_tracking:  # Only save if still tracking (natural end)
                    self.save_tracking_stats()
            except Exception as e:
                print(f"[ERROR] Video processing error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Update UI when finished
                self.is_tracking = False
                self.tracking_status_label.setText("Tracking finished.")
                self.btn_start_tracking.setStyleSheet("""
                    QPushButton {
                        background-color: #9e9e9e;
                        color: white;
                        border: none;
                        padding: 10px;
                        border-radius: 3px;
                        font-weight: bold;
                    }
                """)
                self._stop_intraop_display_thread()
        
        # Start thread
        self.tracking_thread = threading.Thread(target=run_video_processing, daemon=True)
        self.tracking_thread.start()
        
        print("Started tracking")
    
    def pause_tracking(self):
        if self.is_tracking:
            if hasattr(self, 'tracking_pause_flag'):
                if self.tracking_pause_flag[0]:
                    # Resume
                    self.tracking_pause_flag[0] = False
                    self.tracking_status_label.setText("Tracking resumed...")
                    print("Resumed tracking")
                else:
                    # Pause
                    self.tracking_pause_flag[0] = True
                    self.tracking_status_label.setText("Tracking paused.")
                    print("Paused tracking")
    
    def stop_tracking(self):
        self.is_tracking = False
        if hasattr(self, 'tracking_quit_flag'):
            self.tracking_quit_flag[0] = True
        
        # Wait for thread to finish (with timeout) so we can get the stats logger
        if hasattr(self, 'tracking_thread') and self.tracking_thread is not None:
            self.tracking_thread.join(timeout=3.0)  # Wait up to 3 seconds
        
        # Save statistics after thread finishes
        self.save_tracking_stats()
        self.tracking_status_label.setText("Video tracking finished.")
        self.btn_start_tracking.setStyleSheet("""
            QPushButton {
                background-color: #9e9e9e;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)
        self._stop_intraop_display_thread()
        print("Stopped tracking")
    
    def save_tracking_stats(self):
        """Save rotation statistics to text file"""
        if not hasattr(self, 'tracking_stats_logger') or self.tracking_stats_logger is None:
            print("[WARNING] No statistics logger available. Stats may not be saved.")
            return
        
        try:
            all_stats = self.tracking_stats_logger.get_all()
            if all_stats:
                video_output_dir = "output/video_output"
                os.makedirs(video_output_dir, exist_ok=True)
                # Primary stats file (existing behavior)
                stats_path = os.path.join(video_output_dir, "rotation_stats.txt")
                with open(stats_path, 'w') as f:
                    f.write("Frame-by-Frame Rotation Statistics\n")
                    f.write("=" * 50 + "\n\n")
                    for info in all_stats:
                        f.write(f"Frame {info['frame_num']}:\n")
                        f.write(f"  Rotation: {info['rotation_deg']:.2f}°\n")
                        f.write(f"  Confidence: {info['confidence']*100:.1f}%\n")
                        f.write(f"  Matches: {info['num_matches']}\n")
                        f.write(f"  Reliable: {info['is_reliable']}\n\n")
                
                # Live-style log file (appended format requested)
                log_stat_path = os.path.join(video_output_dir, "log_stat.txt")
                with open(log_stat_path, 'w') as f:
                    f.write("Frame-by-Frame Rotation Statistics\n\n")
                    f.write("==============================================\n\n")
                    for info in all_stats:
                        f.write(f"Frame {info['frame_num']}:\n")
                        f.write(f"  Rotation: {info['rotation_deg']:.2f}°\n")
                        f.write(f"  Confidence: {info['confidence']*100:.1f}%\n")
                        f.write(f"  Matches: {info['num_matches']}\n")
                        f.write(f"  Reliable: {info['is_reliable']}\n\n")
                
                print(f"\n[STATS] Rotation statistics saved: {stats_path}")
                print(f"[STATS] Live log saved: {log_stat_path}")
                print(f"  Total analyzed frames: {len(all_stats)}")
            else:
                print("[WARNING] No statistics collected. Stats file not created.")
        except Exception as e:
            print(f"[ERROR] Failed to save statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def keyPressEvent(self, event):
        """Handle keyboard events for video control"""
        if self.is_tracking:
            if event.key() == Qt.Key_Q:
                # Stop tracking
                print("\n[QUIT] 'q' pressed - Stopping tracking")
                self.stop_tracking()
            elif event.key() == Qt.Key_P:
                # Pause/Resume tracking
                self.pause_tracking()
        super().keyPressEvent(event)
    
    def submit_all_axes(self):
        """Submit all axis angles to handler"""
        try:
            # Get values from UI inputs
            try:
                ref_angle = float(self.ref_angle_input.text())
            except ValueError:
                ref_angle = 0.0
            
            try:
                toric_angle = float(self.toric_angle_input.text())
            except ValueError:
                toric_angle = 0.0
            
            try:
                incision_angle = float(self.incision_angle_input.text())
            except ValueError:
                incision_angle = 0.0
            
            # Save to handler
            self.config_handler.set_reference_angle(ref_angle)
            self.config_handler.set_toric_angle(toric_angle)
            self.config_handler.set_incision_angle(incision_angle)
            
            # Also save manual offsets if they exist
            try:
                manual_x = int(self.manual_x_offset.text())
            except ValueError:
                manual_x = 0
            
            try:
                manual_y = int(self.manual_y_offset.text())
            except ValueError:
                manual_y = 0
            
            self.config_handler.set_manual_x_offset(manual_x)
            self.config_handler.set_manual_y_offset(manual_y)
            
            print("All axes submitted")
            print(f"Reference: {ref_angle}")
            print(f"Toric: {toric_angle}")
            print(f"Incision: {incision_angle}")
            print(f"Manual X offset: {manual_x}")
            print(f"Manual Y offset: {manual_y}")
            
            # Show success message
            QMessageBox.information(self, "Axes Saved", 
                                  f"All axis angles have been saved successfully!\n\n"
                                  f"Reference: {ref_angle}°\n"
                                  f"Toric: {toric_angle}°\n"
                                  f"Incision: {incision_angle}°\n"
                                  f"Manual X offset: {manual_x}px\n"
                                  f"Manual Y offset: {manual_y}px")
            
            # Update canvas display with new values
            self.update_axis_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit axes: {str(e)}")
            print(f"[ERROR] Failed to submit axes: {e}")
    
    def update_axis_display(self):
        """Update the axis display in tab 3 with current handler values"""
        # Get limbus info from preop result (with manual offsets applied)
        preop_result = self.config_handler.get_preop_result()
        if preop_result and preop_result.limbus_info:
            # Apply manual offsets to limbus center
            limbus_center = preop_result.limbus_info.center
            manual_x = self.config_handler.get_manual_x_offset()
            manual_y = self.config_handler.get_manual_y_offset()
            adjusted_center = (
                limbus_center[0] + manual_x,
                limbus_center[1] + manual_y
            )
            self.canvas.set_limbus_info(adjusted_center, preop_result.limbus_info.radius)
        
        # Get angles from handler
        ref_angle = self.config_handler.get_reference_angle()
        toric_angle = self.config_handler.get_toric_angle()
        incision_angle = self.config_handler.get_incision_angle()
        
        self.canvas.set_angles(ref_angle, toric_angle, incision_angle)
    
    def update_axis_display_live(self):
        """Update the axis display in real-time from UI input fields (before submitting)"""
        # Get limbus info from preop result (with manual offsets applied)
        preop_result = self.config_handler.get_preop_result()
        if preop_result and preop_result.limbus_info:
            # Get manual offsets from UI input fields
            try:
                manual_x = int(self.manual_x_offset.text() or "0")
            except ValueError:
                manual_x = 0
            try:
                manual_y = int(self.manual_y_offset.text() or "0")
            except ValueError:
                manual_y = 0
            
            # Apply manual offsets to limbus center
            limbus_center = preop_result.limbus_info.center
            adjusted_center = (
                limbus_center[0] + manual_x,
                limbus_center[1] + manual_y
            )
            self.canvas.set_limbus_info(adjusted_center, preop_result.limbus_info.radius)
        
        # Get angles from UI input fields
        try:
            ref_angle = float(self.ref_angle_input.text() or "0")
        except ValueError:
            ref_angle = 0.0
        
        try:
            toric_angle = float(self.toric_angle_input.text() or "0")
        except ValueError:
            toric_angle = 0.0
        
        try:
            incision_angle = float(self.incision_angle_input.text() or "0")
        except ValueError:
            incision_angle = 0.0
        
        self.canvas.set_angles(ref_angle, toric_angle, incision_angle)
    
    def on_tab_changed(self, index):
        """Handle tab change - load limbus detection image when tab 3 (index 2) is selected"""
        if index == 2:  # Tab 3: Pre-op Axis Setup (0-indexed: 0=Config, 1=Pre-op Image, 2=Axis Setup, 3=Tracking)
            self.load_limbus_detection_image()
    
    def load_limbus_detection_image(self):
        """Load and display the limbus detection image in tab 3"""
        limbus_image_path = os.path.join("output", "ui_preprocess", "3_preop_cropped.jpg")
        
        if os.path.exists(limbus_image_path):
            try:
                # Load image using OpenCV
                image = cv2.imread(limbus_image_path)
                if image is not None:
                    # Convert BGR to RGB for display
                    height, width = image.shape[:2]
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    q_image = QImage(rgb_image.data, width, height, 
                                   rgb_image.strides[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.canvas.set_image(pixmap)
                    
                    # Get limbus info from preop result (with manual offsets applied)
                    preop_result = self.config_handler.get_preop_result()
                    if preop_result and preop_result.limbus_info:
                        # Apply manual offsets to limbus center
                        limbus_center = preop_result.limbus_info.center
                        manual_x = self.config_handler.get_manual_x_offset()
                        manual_y = self.config_handler.get_manual_y_offset()
                        adjusted_center = (
                            limbus_center[0] + manual_x,
                            limbus_center[1] + manual_y
                        )
                        self.canvas.set_limbus_info(adjusted_center, preop_result.limbus_info.radius)
                    else:
                        # Fallback: try to detect limbus from the image
                        try:
                            # Note: YOLO confidence is set when loading the model
                            limbus_detected = detect_limbus(self.preop_yolo_model, image)
                            if limbus_detected:
                                manual_x = self.config_handler.get_manual_x_offset()
                                manual_y = self.config_handler.get_manual_y_offset()
                                adjusted_center = (
                                    limbus_detected.center[0] + manual_x,
                                    limbus_detected.center[1] + manual_y
                                )
                                self.canvas.set_limbus_info(adjusted_center, limbus_detected.radius)
                        except:
                            pass
                    
                    # Get angles from handler
                    ref_angle = self.config_handler.get_reference_angle()
                    toric_angle = self.config_handler.get_toric_angle()
                    incision_angle = self.config_handler.get_incision_angle()
                    
                    self.canvas.set_angles(ref_angle, toric_angle, incision_angle)
                    
                    print(f"[UI] Loaded limbus detection image: {limbus_image_path}")
                else:
                    # Image file exists but couldn't be loaded
                    self.canvas.pixmap = None
                    self.canvas.update()
                    print(f"[UI] Could not load image: {limbus_image_path}")
            except Exception as e:
                print(f"[UI ERROR] Failed to load limbus detection image: {e}")
                self.canvas.pixmap = None
                self.canvas.update()
        else:
            # Image doesn't exist - show nothing
            self.canvas.pixmap = None
            self.canvas.update()
            print(f"[UI] Limbus detection image not found: {limbus_image_path}")
    
    def update_canvas(self):
        self.canvas.show_ref_line = self.check_ref_line.isChecked()
        self.canvas.show_toric_axis = self.check_toric.isChecked()
        self.canvas.show_incision_axis = self.check_incision.isChecked()
        self.canvas.show_iris_box = self.check_iris.isChecked()
        self.canvas.update()
    
    def _update_tracking_checkbox(self, checkbox_type: str, state: int):
        """Update tracking checkbox state reference for dynamic updates during tracking"""
        checked = (state == 2)  # Qt.Checked = 2
        if checkbox_type == 'reference':
            self.tracking_show_reference_ref[0] = checked
        elif checkbox_type == 'toric':
            self.tracking_show_toric_ref[0] = checked
        elif checkbox_type == 'incision':
            self.tracking_show_incision_ref[0] = checked
        elif checkbox_type == 'limbus':
            self.tracking_show_limbus_ref[0] = checked
    
    def reset_all(self):
        # Delete output folder before resetting
        self.delete_output_folder()
        self._stop_intraop_display_thread()
        
        # Stop tracking if active
        if hasattr(self, 'is_tracking') and self.is_tracking:
            self.stop_tracking()
        
        # Reset handler to defaults
        self.config_handler.reset_to_defaults()
        
        # Set default model paths to intraop_latest.pt for both preop and intraop
        default_model = "model\\intraop_latest.pt"
        self.config_handler.set_preop_model_path(default_model)
        self.config_handler.set_intraop_model_path(default_model)
        
        # Reset configuration tab
        self.radio_normal.setChecked(True)
        self.confidence_input.setText(str(self.config_handler.get_yolo_confidence()))
        self.matching_confidence_input.setText(str(self.config_handler.get_matching_confidence_threshold()))
        
        # Update model status labels with default model
        default_model_name = os.path.basename(default_model)
        self.label_preop_status.setText(f"Pre-op Model: {default_model_name}")
        self.label_intraop_status.setText(f"Intra-op Model: {default_model_name}")
        
        # Reset pre-op image tab
        self.radio_file.setChecked(True)
        self.current_preop_image = None
        self.current_preop_image_path = None
        self.preprocessed_preop_image = None
        self.rotation_angle = 0
        self.preop_status_label.setText("No image loaded.")
        # Reset to handler defaults
        handler = get_config_handler()
        self.inner_exclude_ratio_input.setText(str(handler.get_inner_exclude_ratio()))
        self.outer_include_ratio_input.setText(str(handler.get_outer_include_ratio()))
        self.crop_width_ratio_input.setText(str(handler.get_crop_width_ratio()))
        self.crop_height_ratio_input.setText(str(handler.get_crop_height_ratio()))
        self.eyelid_trim_upper_ratio_input.setText(str(handler.get_eyelid_trim_upper_ratio()))
        self.eyelid_trim_lower_ratio_input.setText(str(handler.get_eyelid_trim_lower_ratio()))
        
        # Reset axis setup tab
        self.ref_angle_input.setText("0")
        self.manual_y_offset.setText("0")
        self.manual_x_offset.setText("0")
        self.toric_angle_input.setText("0")
        self.incision_angle_input.setText("0")
        self.check_ref_line.setChecked(True)
        self.check_toric.setChecked(True)
        self.check_incision.setChecked(True)
        self.check_iris.setChecked(True)
        
        # Reset live tracking tab
        self.is_tracking = False
        self.tracking_video_path = None
        # Reset camera dropdown to first available camera
        if hasattr(self, 'camera_input') and isinstance(self.camera_input, QComboBox) and self.camera_input.count() > 0:
            self.camera_input.setCurrentIndex(0)
        self.video_source_label.setText("Input Source: No file selected")
        self.tracking_status_label.setText("Not tracking.")
        self.rotation_label.setText("Current Angle of rotation: 0.0 deg")
        self.radius_input.clear()
        
        # Reset tracking checkboxes
        if hasattr(self, 'check_ref_lines_tracking'):
            self.check_ref_lines_tracking.setChecked(True)
        if hasattr(self, 'check_toric_tracking'):
            self.check_toric_tracking.setChecked(True)
        if hasattr(self, 'check_incision_tracking'):
            self.check_incision_tracking.setChecked(True)
        if hasattr(self, 'check_iris_ring'):
            self.check_iris_ring.setChecked(True)
        
        # Reset tracking checkbox state references
        if hasattr(self, 'tracking_show_reference_ref'):
            self.tracking_show_reference_ref[0] = True
        if hasattr(self, 'tracking_show_toric_ref'):
            self.tracking_show_toric_ref[0] = True
        if hasattr(self, 'tracking_show_incision_ref'):
            self.tracking_show_incision_ref[0] = True
        if hasattr(self, 'tracking_show_limbus_ref'):
            self.tracking_show_limbus_ref[0] = True
        
        # Clear video display
        if hasattr(self, 'video_display'):
            self.video_display.clear()
        
        # Clear canvases
        self.preop_canvas.pixmap = None
        self.preop_canvas.update()
        self.canvas.pixmap = None
        self.canvas.update()
        
        # Show confirmation message
        QMessageBox.information(
            self, 
            "Reset Complete", 
            "Application state has been reset:\n\n"
            "- All application state cleared\n"
            "- Output folder deleted\n"
            "- Default models set (intraop_latest.pt for both preop and intraop)\n"
            "- All UI elements reset to defaults"
        )
        
        print("Application state reset - all cleared, output folder deleted, default models set")
    
    def delete_output_folder(self):
        """Delete the output folder and all its contents"""
        try:
            import shutil
            output_folder = "output"
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
                print(f"[INFO] Deleted output folder: {output_folder}")
            else:
                print(f"[INFO] Output folder does not exist: {output_folder}")
        except Exception as e:
            print(f"[WARNING] Failed to delete output folder: {e}")
            # Don't show error to user, just log it

if __name__ == '__main__':
    # Required for PyInstaller executables
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ToricTrackerUI()
    window.show()
    sys.exit(app.exec_())