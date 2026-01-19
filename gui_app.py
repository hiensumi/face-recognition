import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import threading
import time
import os
import tempfile
import shutil
import subprocess
import pygame
import albumentations as A
from moviepy import VideoFileClip

from face_recog import FaceDatabase, smart_knn_recognition, calculate_iou
from backend_mediapipe import FaceDetector, FaceEmbedder
from backend_insightface import InsightFaceEngine


class AudioPlayer:
    """Simple audio player using MoviePy to extract audio and pygame.mixer for playback."""
    def __init__(self):
        self.audio_path = None
        self.temp_files = []
        self.loaded = False
        self.has_audio = False
        try:
            # Reduce latency by pre-initializing mixer with small buffer
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
        except Exception:
            pass

    def load_audio(self, video_path):
        """Extract audio from video_path to a temp file and load into pygame.mixer.
        Returns True if audio was loaded, False otherwise."""
        self.unload()
        # Prefer fast ffmpeg extraction if available to avoid heavy moviepy overhead
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.close()
                audio_tmp = tmp.name
                # Extract audio quickly to WAV (fast and compatible with pygame)
                cmd = [ffmpeg_path, '-y', '-i', video_path, '-vn', '-ar', '44100', '-ac', '2', '-f', 'wav', audio_tmp]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(audio_tmp):
                    try:
                        pygame.mixer.music.load(audio_tmp)
                        self.audio_path = audio_tmp
                        self.temp_files.append(audio_tmp)
                        self.has_audio = True
                        self.loaded = True
                        return True
                    except Exception as e:
                        print(f"pygame failed to load ffmpeg-extracted audio: {e}")
                # fallback to moviepy below
            except Exception as e:
                print(f"ffmpeg audio extract failed: {e}")

        # Fallback: use MoviePy (slower)
        try:
            clip = VideoFileClip(video_path)
        except Exception as e:
            print(f"Audio extract: failed to open video with moviepy: {e}")
            return False

        if not clip.audio:
            clip.close()
            return False

        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            audio_tmp = tmp.name
            # write audio
            clip.audio.write_audiofile(audio_tmp, logger=None)
            clip.close()
            self.audio_path = audio_tmp
            self.temp_files.append(audio_tmp)
            try:
                pygame.mixer.music.load(self.audio_path)
                self.has_audio = True
                self.loaded = True
                return True
            except Exception as e:
                print(f"pygame failed to load audio: {e}")
                return False
        except Exception as e:
            print(f"Audio extract/write failed: {e}")
            try:
                clip.close()
            except Exception:
                pass
            return False

    def play(self, start=0.0):
        if not self.loaded or not self.has_audio:
            return
        try:
            # pygame supports start parameter for some formats
            pygame.mixer.music.play(loops=0, start=float(start))
        except Exception:
            try:
                pygame.mixer.music.play()
            except Exception as e:
                print(f"pygame play failed: {e}")

    def pause(self):
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def unpause(self):
        try:
            pygame.mixer.music.unpause()
        except Exception:
            pass

    def stop(self):
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def set_volume(self, vol):
        try:
            pygame.mixer.music.set_volume(float(vol))
        except Exception:
            pass

    def unload(self):
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        # try to remove temp files
        for f in list(self.temp_files):
            try:
                os.remove(f)
                self.temp_files.remove(f)
            except PermissionError:
                print(f"PermissionError: cannot delete temp audio {f} (still in use)")
            except Exception:
                pass
        self.audio_path = None
        self.loaded = False
        self.has_audio = False


class FaceRecognitionApp:
    def _load_insightface_model(self, config, was_running):
        self.profile_type = "insightface"
        self.insight_engine = InsightFaceEngine(
            model_name=config["model_name"],
            root_dir=config["root_dir"]
        )
        self.face_db = FaceDatabase(
            index_path=config["index_path"],
            json_path=config["json_path"]
        )
        try:
            dpg.set_value("enroll_status", f"InsightFace model {config['model_name']} loaded.")
            dpg.configure_item("enroll_status", color=(0, 255, 0))
            self.is_loading_model = False
        except Exception:
            pass
        # Restart camera after model loads if it was running before
        if was_running:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.is_running = True
            if self.video_thread is None or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.video_loop)
                self.video_thread.daemon = True
                self.video_thread.start()
        
        # Refresh database list now that the model is loaded
        try:
            self.refresh_db_list()
        except Exception as e:
            print(f"Error refreshing database list after InsightFace load: {e}")

    def __init__(self):
        # Model Configuration
        self.models_config = {
            "ResNet50": {
                "profile": "mediapipe",
                "model_path": "./models/arcface_w600k_r50.onnx",
                "index_path": "./database/arcface_w600k_r50/faces.index",
                "json_path": "./database/arcface_w600k_r50/id_to_name.json"
            },
            "MobileFaceNet": {
                "profile": "mediapipe",
                "model_path": "./models/arcface_w600k_mbf.onnx",
                "index_path": "./database/arcface_w600k_mbf/faces.index",
                "json_path": "./database/arcface_w600k_mbf/id_to_name.json"
            },
            "Buffalo_S": {
                "type": "insightface",
                "model_name": "buffalo_s",
                "root_dir": ".",
                "index_path": "./database/buffalo_s/faces.index",
                "json_path": "./database/buffalo_s/id_to_name.json"
            },
            "Buffalo_L": {
                "type": "insightface",
                "model_name": "buffalo_l",
                "root_dir": ".",
                "index_path": "./database/buffalo_l/faces.index",
                "json_path": "./database/buffalo_l/id_to_name.json"
            }
        }
        self.current_model_name = "ResNet50"
        self.rolling_update_enabled = False
        self.teacher_system = None # Will hold {'embedder': ..., 'db': ...} if enabled
        self.insight_engine = None

        # CSRT multi-object tracker state
        # Dictionary: track_id -> {'tracker': cv2.TrackerCSRT, 'bbox': tuple, 'missed_frames': int}
        self.csrt_trackers = {}
        self.next_track_id = 0
        self.max_missed_frames = 15  # Remove tracker after this many missed frames
        # Cache for track_id -> recognized identity info
        # Key: track_id, Value: {'name': str, 'similarity': float, 'embedding': np.array, 'color': tuple}
        self.track_identities = {}

        self.detection_confidence = 0.7
        # MediaPipe Tasks API uses model_path instead of model_selection
        self.detector_model_path = "./models/blaze_face_short_range.tflite"  # Default to short-range
        # UI profile selection: controls detector+embedder pairing
        self.profile = "MediaPipe+ArcFace"  # or 'InsightFace'

        # Initialize active system after configuration variables are set
        self.load_active_system()
        
        self.is_running = False
        self.video_thread = None
        self.cap = None
        
        self.similarity_threshold = 0.7
        self.use_simple_detection = False
        self.adaptive_threshold = 0.85
        self.use_alignment = True
        
        self.tracked_faces = []
        self.frame_count = 0
        self.skip_frames = 10
        self.learning_threshold = 0.75
        self.learning_tracker = {}
        
        # Camera settings
        self.camera_index = 0
        
        # Enrollment State
        self.is_enrolling = False
        self.enroll_queue = []
        self.enroll_frame_base = None
        self.current_enroll_face = None
        self.last_frame = None
        
        # Augmentation Pipeline
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.ISONoise(p=0.2),
        ])
        
        # Video Player State
        self.video_path = ""
        self.is_playing_video = False
        self.video_thread = None
        self.video_cap = None
        self.audio_player = AudioPlayer()
        self.video_lock = threading.Lock()
        self.video_tracked_faces = []
        self.video_learning_tracker = {}
        self.video_frame_count = 0
        self.last_video_frame = None
        self.enrolling_from_video = False
        self.video_total_frames = 0
        self.check_rotation = True
        self.detect_while_playback = False
        self.volume = 1.0
        self.is_muted = False
        self.is_processing = False
        self.processed_video_path = ""
        self.processing_paused_for_input = False
        self.unknown_face_data = None
        self.skipped_faces_embeddings = []
        # Processing Frame Preview
        self.preview_width = 320
        self.preview_height = 240
        self.preview_texture_data = np.zeros((self.preview_height, self.preview_width, 4), dtype=np.float32)
        
        # Texture dimensions
        self.tex_width = 640
        self.tex_height = 480
        self.texture_data = np.zeros((self.tex_height, self.tex_width, 4), dtype=np.float32)
        self.video_player_texture_data = np.zeros((self.tex_height, self.tex_width, 4), dtype=np.float32)
        
        self.setup_gui()
        self.is_loading_model = False


    def load_active_system(self):
        # ensure current_model_name matches selected profile; if not, pick first matching
        available = self.get_models_for_profile()
        if self.current_model_name not in available:
            if available:
                self.current_model_name = available[0]
            else:
                pass

        config = self.models_config[self.current_model_name]
        if config.get("type") == "insightface":
            try:
                dpg.set_value("enroll_status", f"Loading InsightFace model: {config['model_name']}... (see terminal for progress)")
                dpg.configure_item("enroll_status", color=(255, 0, 0))
                self.is_loading_model = True
                # Immediately blank the video frame
                blank_frame = np.zeros((self.tex_height, self.tex_width, 4), dtype=np.float32)
                dpg.set_value("video_texture", blank_frame)
            except Exception:
                pass
            # Stop camera while loading model
            was_running = self.is_running
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            # Load model in a daemon thread
            t = threading.Thread(target=self._load_insightface_model, args=(config, was_running), daemon=True)
            t.start()
        else:
            print(f"Loading MediaPipe Engine: {self.current_model_name}...")
            self.profile_type = "mediapipe"
            self.insight_engine = None
            self.face_detector = FaceDetector(
                min_detection_confidence=self.detection_confidence,
                model_path=self.detector_model_path
            )
            self.face_embedder = FaceEmbedder(model_path=config["model_path"])
            self.face_db = FaceDatabase(
                index_path=config["index_path"],
                json_path=config["json_path"]
            )
        print(f"Loaded {self.current_model_name} with {self.face_db.index.ntotal} faces.")

    def get_available_teachers(self):
        return ["None"] + [m for m in self.models_config.keys() if m != self.current_model_name]

    def get_models_for_profile(self):
        """Return list of model names matching current profile."""
        prof = (self.profile or "").lower()
        res = []
        for name, cfg in self.models_config.items():
            # media pipe profile marked by 'profile': 'mediapipe' or default
            p = cfg.get('profile') or cfg.get('type') or ''
            if prof.startswith('mediapipe'):
                if str(p).lower() in ['', 'mediapipe']:
                    res.append(name)
            elif prof.startswith('insight'):
                if str(p).lower() == 'insightface' or str(p).lower() == 'insight':
                    res.append(name)
        return res

    def change_profile(self, sender, app_data):
        # app_data is display name like 'MediaPipe+ArcFace' or 'InsightFace'
        self.profile = app_data
        # refresh model list to models matching profile
        models = self.get_models_for_profile()
        if models:
            # set model combo items and select first
            try:
                dpg.configure_item("combo_model", items=models)
                self.current_model_name = models[0]
                dpg.set_value("combo_model", self.current_model_name)
                self.load_active_system()
                
                # Update Teacher dropdown (include cross-profile models)
                teachers = self.get_available_teachers()
                dpg.configure_item("combo_teacher", items=teachers)
                dpg.set_value("combo_teacher", "None")
                
                # Reset Rolling Update state
                self.rolling_update_enabled = False
                self.teacher_system = None
                try:
                    dpg.set_value("chk_rolling", False)
                except:
                    pass
            except Exception as e:
                print(f"Error updating profile UI: {e}")
        
        # Refresh database list for MediaPipe profiles (InsightFace does it after background loading)
        if not str(self.profile).lower().startswith('insight'):
            try:
                self.refresh_db_list()
            except Exception as e:
                print(f"Error refreshing database list: {e}")
            
        # Immediately apply UI visibility changes for the new profile
        try:
            # Update detection range visibility
            if str(self.profile).lower().startswith('insight'):
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=False)
            else:
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=True)

            # Trigger tab-change logic so shared settings visibility updates immediately
            try:
                active_tab = dpg.get_value("main_tab_bar")
            except Exception:
                active_tab = None
            try:
                self.on_tab_change(None, active_tab)
            except Exception:
                pass
        except Exception:
            pass

    def on_tab_change(self, sender, app_data):
        """Handle tab changes and apply UI visibility rules.
        `app_data` is typically the active tab tag (e.g., 'tab_database').
        """
        try:
            # active = app_data or dpg.get_value("main_tab_bar")
            active = ""
            if type(app_data) is str:
                active = app_data
            else:
                active = dpg.get_value("main_tab_bar")
                if type(app_data) is not str:
                    active = dpg.get_item_alias(active) 
        except Exception:
            active = None

        # Apply visibility rules per active tab:
        # - Database Manager: hide Global Settings and Enrollment
        # - Video Player: show Global Settings, hide Enrollment
        # - Recognition: show both
        try:
            if active == "tab_database":
                if dpg.does_item_exist("shared_settings_group"):
                    dpg.configure_item("shared_settings_group", show=False)
                if dpg.does_item_exist("enrollment_group"):
                    dpg.configure_item("enrollment_group", show=False)
            elif active == "tab_video":
                if dpg.does_item_exist("shared_settings_group"):
                    dpg.configure_item("shared_settings_group", show=True)
                if dpg.does_item_exist("enrollment_group"):
                    dpg.configure_item("enrollment_group", show=False)
            else:
                # default: show everything
                if dpg.does_item_exist("shared_settings_group"):
                    dpg.configure_item("shared_settings_group", show=True)
                if dpg.does_item_exist("enrollment_group"):
                    dpg.configure_item("enrollment_group", show=True)
        except Exception:
            pass

        # Ensure detection range visibility matches selected profile
        try:
            if str(self.profile).lower().startswith('insight'):
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=False)
            else:
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=True)
        except Exception:
            pass

    def load_teacher_system(self):
        teacher_name = dpg.get_value("combo_teacher")
        if not teacher_name or teacher_name == "None" or teacher_name == self.current_model_name:
            print("No teacher selected.")
            self.rolling_update_enabled = False
            dpg.set_value("chk_rolling", False)
            self.teacher_system = None
            return

        config = self.models_config[teacher_name]
        print(f"Loading teacher model: {teacher_name}...")
        self.teacher_system = {
            'embedder': FaceEmbedder(model_path=config["model_path"]),
            'db': FaceDatabase(index_path=config["index_path"], json_path=config["json_path"])
        }
        print(f"Teacher loaded.")

    def change_model(self, sender, app_data):
        if app_data != self.current_model_name:
            self.current_model_name = app_data
            self.load_active_system()
            
            # Reset Rolling Update
            self.rolling_update_enabled = False
            self.teacher_system = None
            dpg.set_value("chk_rolling", False)
            
            # Update Teacher Options
            teachers = self.get_available_teachers()
            dpg.configure_item("combo_teacher", items=teachers)
            dpg.set_value("combo_teacher", "None")
            
            self.refresh_db_list()

    def toggle_rolling_update(self, sender, app_data):
        self.rolling_update_enabled = app_data
        if self.rolling_update_enabled:
            if not self.teacher_system:
                self.load_teacher_system()
        else:
            self.teacher_system = None # Free memory? Or keep it? Let's free it to be safe.

    def update_camera_source(self, sender, app_data):
        self.camera_index = app_data
        if self.is_running:
            self.stop_camera()
            self.start_camera()

    def handle_enrollment(self):
        # dpg.get_value returns the integer ID of the active tab, not the tag string
        active_tab_id = dpg.get_value("main_tab_bar")
        # Get the alias/tag from the ID
        try:
            active_tab = dpg.get_item_alias(active_tab_id)
        except Exception:
            active_tab = None
        
        if active_tab == "tab_recognition":
            self.start_enrollment_process()
        elif active_tab == "tab_video":
            self.start_video_enrollment_process()

    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Face Recognition Manager', width=1300, height=900)
        
        # Texture Registry
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.tex_width, height=self.tex_height, default_value=self.texture_data, format=dpg.mvFormat_Float_rgba, tag="video_texture")
            dpg.add_raw_texture(width=self.tex_width, height=self.tex_height, default_value=self.video_player_texture_data, format=dpg.mvFormat_Float_rgba, tag="video_player_texture")
            dpg.add_raw_texture(width=150, height=150, default_value=np.zeros((150, 150, 4), dtype=np.float32), format=dpg.mvFormat_Float_rgba, tag="unknown_face_texture")
            dpg.add_raw_texture(width=self.preview_width, height=self.preview_height, default_value=self.preview_texture_data, format=dpg.mvFormat_Float_rgba, tag="process_preview_texture")
            
        with dpg.window(label="Main Window", tag="Primary Window"):
            with dpg.tab_bar(tag="main_tab_bar", callback=self.on_tab_change):
                # Tab 1: Recognition
                with dpg.tab(label="Recognition", tag="tab_recognition"):
                    with dpg.group(horizontal=True):
                        dpg.add_image("video_texture")
                        
                        with dpg.group():
                            dpg.add_text("Controls")
                            dpg.add_input_int(label="Camera Index", default_value=self.camera_index, callback=self.update_camera_source, width=150)
                            dpg.add_button(label="Start Camera", callback=self.start_camera, tag="btn_start")
                            dpg.add_button(label="Stop Camera", callback=self.stop_camera, tag="btn_stop", show=False)
                            dpg.add_text("FPS: 0.0", tag="fps_text")
                            
                            dpg.add_text("", tag="enroll_status", color=(0, 255, 0))

                # Tab 2: Video Player
                with dpg.tab(label="Video Player", tag="tab_video"):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_image("video_player_texture")
                            dpg.add_slider_int(label="Progress", tag="video_progress", default_value=0, max_value=100, width=self.tex_width, callback=self.seek_video)
                            
                        with dpg.group():
                            dpg.add_button(label="Select Video File", callback=lambda: dpg.show_item("file_dialog_id"))
                            dpg.add_text("No file selected", tag="video_path_text", wrap=200)
                            dpg.add_separator()
                            
                            dpg.add_text("Processing")
                            dpg.add_image("process_preview_texture", width=self.preview_width, height=self.preview_height)
                            dpg.add_button(label="Process & Save", callback=self.start_processing_video, tag="btn_process")
                            dpg.add_button(label="Stop Processing", callback=self.stop_processing_video, tag="btn_stop_process", show=False)
                            dpg.add_progress_bar(label="Progress", tag="process_progress", default_value=0.0, width=150)
                            dpg.add_text("Idle", tag="process_status")
                            
                            dpg.add_separator()
                            dpg.add_text("Playback")
                            dpg.add_button(label="Play Video", callback=lambda: self.start_video(is_processed=False), tag="btn_play_video")
                            dpg.add_button(label="Play Result", callback=self.play_processed_video, tag="btn_play_result", show=False)
                            dpg.add_button(label="Stop", callback=self.stop_video, tag="btn_stop_video", show=False)
                            dpg.add_text("Frame: 0", tag="video_frame_text")
                            
                            dpg.add_separator()
                            dpg.add_text("Video Settings")
                            
                            dpg.add_slider_float(label="Volume", default_value=100.0, max_value=100.0, width=150, callback=self.update_volume)
                            dpg.add_checkbox(label="Mute Audio", default_value=False, callback=self.update_mute)
                            dpg.add_checkbox(label="Detect During Playback (Makes audio delayed, mute audio if needed)", tag="chk_detect_playback", default_value=self.detect_while_playback, callback=self.update_detect_while_playback)
                            
                            dpg.add_text("", tag="video_enroll_status", color=(0, 255, 0))

                # Tab 3: Database Manager
                with dpg.tab(label="Database Manager", tag="tab_database"):
                    dpg.add_button(label="Refresh List", callback=self.refresh_db_list)
                    dpg.add_button(label="Save Database", callback=self.save_database)
                    dpg.add_separator()
                    
                    # Table for users
                    with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, tag="users_table"):
                        dpg.add_table_column(label="Name")
                        dpg.add_table_column(label="Face Count")
                        dpg.add_table_column(label="Actions")

            dpg.add_separator()
            with dpg.group(tag="shared_settings_group"):
                dpg.add_text("Global Settings")
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Profile")
                        dpg.add_combo(label="Profile", items=["MediaPipe+ArcFace", "InsightFace"], default_value=self.profile, callback=self.change_profile, width=200)
                        dpg.add_text("Models")
                        dpg.add_combo(label="Model", tag="combo_model", items=self.get_models_for_profile(), default_value=self.current_model_name, callback=self.change_model, width=200)
                        dpg.add_combo(label="Teacher", tag="combo_teacher", items=self.get_available_teachers(), default_value="None", width=200)
                        dpg.add_checkbox(label="Rolling Update", tag="chk_rolling", default_value=False, callback=self.toggle_rolling_update)
                    
                    dpg.add_spacer(width=20)

                    with dpg.group():
                        dpg.add_text("Thresholds")
                        dpg.add_slider_float(label="Similarity Threshold", default_value=self.similarity_threshold, max_value=1.0, callback=self.update_threshold, width=200)
                        dpg.add_slider_float(label="Detection Confidence", default_value=self.detection_confidence, max_value=1.0, callback=self.update_detection_confidence, width=200)
                        dpg.add_combo(label="Detection Range", tag="combo_detection_range", items=["New MediaPipe only supports Short Range"], default_value="New MediaPipe only supports Short Range", callback=self.update_detector_model, width=200, show=(not str(self.profile).lower().startswith('insight')))
                        dpg.add_slider_float(label="Adaptive Threshold", default_value=self.adaptive_threshold, max_value=1.0, callback=self.update_adaptive_threshold, width=200)
                        dpg.add_slider_float(label="Learning Threshold", default_value=self.learning_threshold, max_value=1.0, callback=self.update_learning_threshold, width=200)
                    
                    dpg.add_spacer(width=20)
                    
                    with dpg.group():
                        dpg.add_text("Performance & Models")
                        dpg.add_slider_int(label="Skip Frames", default_value=self.skip_frames, max_value=30, callback=self.update_skip_frames, width=200)
                        dpg.add_checkbox(label="Simple Detection (Faster)", default_value=self.use_simple_detection, callback=self.update_detection_mode)
                        dpg.add_checkbox(label="Align Faces", default_value=self.use_alignment, callback=self.update_alignment_mode)
                        dpg.add_checkbox(label="Check Rotation (Slow)", default_value=self.check_rotation, callback=self.update_check_rotation)

                dpg.add_separator()
                with dpg.group(horizontal=True, tag="enrollment_group"):
                    dpg.add_text("Enrollment")
                    dpg.add_button(label="Enroll Face", callback=self.handle_enrollment, width=150, height=30)
                    dpg.add_checkbox(label="Auto-save New Faces", default_value=False, tag="chk_autosave")

        # Enrollment Modal
        with dpg.window(label="Enroll Face", modal=True, show=False, tag="enroll_modal", width=300, height=220, pos=[750, 250], no_close=True):
            dpg.add_text("Identified as: Unknown", tag="enroll_id_text")
            dpg.add_input_text(label="Name", tag="enroll_input_name")
            dpg.add_checkbox(label="Augment Data", tag="chk_augment", default_value=True)
            dpg.add_input_int(label="Augment Count", tag="input_augment_count", default_value=20, min_value=1, max_value=100, width=100)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_enrollment)
                dpg.add_button(label="Skip", callback=self.skip_enrollment)
                dpg.add_button(label="Cancel All", callback=self.cancel_enrollment)

        # Label Unknown Face Modal (For Video Processing)
        with dpg.window(label="Label Unknown Face", modal=True, show=False, tag="label_modal", width=300, height=300, pos=[400, 200], no_close=True):
            dpg.add_image("unknown_face_texture", width=150, height=150)
            dpg.add_input_text(label="Name", tag="label_input_name")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save & Continue", callback=self.save_label_unknown)
                dpg.add_button(label="Skip", callback=self.skip_label_unknown)

        # File Dialog
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.select_video_file, tag="file_dialog_id", width=700, height=400):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".avi", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mov", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
        # Initial DB load
        self.refresh_db_list()
        # Ensure detection range visibility matches profile at startup
        try:
            if str(self.profile).lower().startswith('insight'):
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=False)
        except Exception:
            pass

    def update_threshold(self, sender, app_data):
        self.similarity_threshold = app_data

    def update_detection_mode(self, sender, app_data):
        self.use_simple_detection = app_data

    def update_adaptive_threshold(self, sender, app_data):
        self.adaptive_threshold = app_data

    def update_alignment_mode(self, sender, app_data):
        self.use_alignment = app_data

    def update_skip_frames(self, sender, app_data):
        self.skip_frames = app_data

    def update_learning_threshold(self, sender, app_data):
        self.learning_threshold = app_data

    def update_detection_confidence(self, sender, app_data):
        self.detection_confidence = app_data
        # Update detector
        try:
            self.face_detector.set_confidence(self.detection_confidence)
        except Exception as e:
            print(f"Warning: failed to set detection confidence: {e}")

    def update_detect_while_playback(self, sender, app_data):
        """Toggle whether to run detections during video playback."""
        try:
            self.detect_while_playback = bool(app_data)
        except Exception:
            # best-effort fallback
            self.detect_while_playback = False

    def update_detector_model(self, sender, app_data):
        # app_data is the detection range selection (currently only 'Short Range' with Tasks API)
        # MediaPipe Tasks API uses model_path - update path based on selection
        if app_data == "Short Range":
            self.detector_model_path = "./models/blaze_face_short_range.tflite"
        else:
            # Default to short range if unknown selection
            self.detector_model_path = "./models/blaze_face_short_range.tflite"
        try:
            # Recreate detector with new model path
            self.face_detector = FaceDetector(
                min_detection_confidence=self.detection_confidence,
                model_path=self.detector_model_path
            )
        except Exception as e:
            print(f"Warning: failed to update detector model: {e}")

    def update_check_rotation(self, sender, app_data):
        self.check_rotation = app_data

    def update_volume(self, sender, app_data):
        self.volume = app_data / 100.0
        if self.audio_player and not self.is_muted:
            try:
                self.audio_player.set_volume(self.volume)
            except Exception:
                pass

    def update_mute(self, sender, app_data):
        self.is_muted = app_data
        if self.audio_player:
            try:
                self.audio_player.set_volume(0.0 if self.is_muted else self.volume)
            except Exception:
                pass
        # Immediately apply UI changes for the new profile (use same logic as tab change)
        try:
            self.on_tab_change(None, None)
        except Exception:
            # best-effort: directly set detection range visibility
            try:
                if str(self.profile).lower().startswith('insight'):
                    if dpg.does_item_exist("combo_detection_range"):
                        dpg.configure_item("combo_detection_range", show=False)
                else:
                    if dpg.does_item_exist("combo_detection_range"):
                        dpg.configure_item("combo_detection_range", show=True)
            except Exception:
                pass
            # Always enforce detection-range visibility per profile
            if str(self.profile).lower().startswith('insight'):
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=False)
            else:
                if dpg.does_item_exist("combo_detection_range"):
                    dpg.configure_item("combo_detection_range", show=True)
        except Exception:
            pass

    def seek_video(self, sender, app_data):
        with self.video_lock:
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, app_data)
                self.video_frame_count = app_data
                
                # Seek audio
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    timestamp = app_data / fps
                    # Adjust wall-clock start time so audio and video align
                    try:
                        self.start_time = time.time() - timestamp
                    except Exception:
                        pass
                    # Seek audio if available by restarting playback at timestamp
                    try:
                        if self.audio_player and hasattr(self.audio_player, 'play') and self.audio_player.has_audio:
                            self.audio_player.stop()
                            self.audio_player.play(start=timestamp)
                    except Exception:
                        pass

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.tex_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.tex_height)
            self.is_running = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            dpg.configure_item("btn_start", show=False)
            dpg.configure_item("btn_stop", show=True)

    def stop_camera(self):
        self.is_running = False
        if self.video_thread:
            self.video_thread.join()
        if self.cap:
            self.cap.release()
        dpg.configure_item("btn_start", show=True)
        dpg.configure_item("btn_stop", show=False)
        
        # Clear texture
        empty_texture = np.zeros((self.tex_height, self.tex_width, 4), dtype=np.float32)
        dpg.set_value("video_texture", empty_texture)

    def detect_faces(self, frame_rgb):
        dets = []
        
        def _run_detection(img):
            d = []
            if self.use_simple_detection:
                detection = self.face_detector.detect(img)
                if detection.detections:
                    for face in detection.detections:
                        bbox = face.location_data.relative_bounding_box
                        h, w = img.shape[:2]
                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        x2 = int((bbox.xmin + bbox.width) * w)
                        y2 = int((bbox.ymin + bbox.height) * h)
                        d.append((x1, y1, x2, y2))
            else:
                faces = self.face_detector.detect_improved(img)
                for face in faces:
                    d.append(face['bbox'])
            return d

        # 1. Standard Detection
        dets = _run_detection(frame_rgb)

        # 2. Rotation Check - try multiple orientations if no faces found
        if not dets and self.check_rotation:
            h, w = frame_rgb.shape[:2]
            
            # 90° CW rotation
            frame_cw = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
            dets_cw = _run_detection(frame_cw)
            if dets_cw:
                for (rx1, ry1, rx2, ry2) in dets_cw:
                    # Convert rotated coords back to original frame
                    # Rotated (x', y') → Original (y', h - 1 - x')
                    nx1 = ry1
                    ny1 = h - 1 - rx2
                    nx2 = ry2
                    ny2 = h - 1 - rx1
                    dets.append((min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)))
            
            # 90° CCW rotation
            if not dets:
                frame_ccw = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                dets_ccw = _run_detection(frame_ccw)
                if dets_ccw:
                    for (rx1, ry1, rx2, ry2) in dets_ccw:
                        # Convert rotated coords back to original frame
                        # Rotated (x', y') → Original (w - 1 - y', x')
                        nx1 = w - 1 - ry2
                        ny1 = rx1
                        nx2 = w - 1 - ry1
                        ny2 = rx2
                        dets.append((min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)))
            
            # 180° rotation (upside down)
            if not dets:
                frame_180 = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                dets_180 = _run_detection(frame_180)
                if dets_180:
                    for (rx1, ry1, rx2, ry2) in dets_180:
                        # Convert rotated coords back to original frame
                        # Rotated (x', y') → Original (w - 1 - x', h - 1 - y')
                        nx1 = w - 1 - rx2
                        ny1 = h - 1 - ry2
                        nx2 = w - 1 - rx1
                        ny2 = h - 1 - ry1
                        dets.append((min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)))
        
        return dets

    def _try_rotated_embedding(self, face_crop, embed_func, min_threshold=0.35):
        """Try different rotations of face crop to find best embedding match.
        
        Args:
            face_crop: Face crop image (RGB for MediaPipe, BGR for InsightFace)
            embed_func: Function that takes face crop and returns (embedding, name, similarity)
            min_threshold: Minimum similarity to accept without trying rotations
            
        Returns:
            (best_embedding, best_name, best_similarity)
        """
        # Try original orientation first
        emb, name, similarity = embed_func(face_crop)
        
        # If already good match, return immediately
        if similarity >= min_threshold and name != "unknown":
            return emb, name, similarity
        
        best_result = (emb, name, similarity)
        
        # Define rotations to try: 90° CW, 90° CCW, 180°
        rotations = [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180
        ]
        
        for rotation in rotations:
            rotated_crop = cv2.rotate(face_crop, rotation)
            r_emb, r_name, r_sim = embed_func(rotated_crop)
            
            # Keep best result
            if r_sim > best_result[2]:
                best_result = (r_emb, r_name, r_sim)
            
            # Early exit if we found a good match
            if r_sim >= min_threshold and r_name != "unknown":
                return best_result
        
        return best_result

    def check_face_quality_for_enrollment(self, face_crop, min_size=80):
        """Check if a face crop is of sufficient quality for ENROLLMENT.
        
        Uses a weighted scoring system where each criterion contributes to the final score.
        
        Args:
            face_crop: Face crop image (BGR format for InsightFace, RGB for MediaPipe)
            min_size: Minimum width/height in pixels
            
        Returns:
            (is_valid, score, reason) - True if quality is acceptable, quality score 0-1, reason
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0, "empty"
        
        h, w = face_crop.shape[:2]
        
        # Size check - need larger face for enrollment
        if w < min_size or h < min_size:
            return False, 0.0, "too_small"
        
        # Define weights for each quality criterion (must sum to 1.0)
        WEIGHTS = {
            'detection': 0.10,      # Face detection confidence
            'pose_yaw': 0.15,       # Yaw angle (side profile) - important
            'pose_pitch': 0.10,     # Pitch angle (looking up/down)
            'landmarks': 0.30,      # Landmark quality/proportions
            'sharpness': 0.20,      # Image sharpness (blur detection) - important
            'brightness': 0.08,     # Lighting conditions
            'contrast': 0.07,       # Image contrast
        }
        
        # Initialize all criteria scores to 1.0 (perfect)
        criteria_scores = {k: 1.0 for k in WEIGHTS}
        issues = []
        
        # Try to detect face and get landmarks to assess quality
        try:
            if self.profile_type == "insightface" and self.insight_engine is not None:
                # InsightFace path - use detection score and pose
                # Note: face_crop may be RGB or BGR depending on caller - detect and handle
                # InsightFace expects BGR
                if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                    # Check if it's likely RGB by looking at the call context
                    # We'll standardize by treating input as BGR (most callers pass BGR for InsightFace)
                    face_bgr = face_crop  # Assume BGR from InsightFace callers
                else:
                    face_bgr = face_crop
                    
                pad = int(max(h, w) * 0.15)
                padded = cv2.copyMakeBorder(face_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                
                faces = self.insight_engine.app.get(padded)
                if not faces or len(faces) == 0:
                    return False, 0.0, "no_face_detected"
                
                face_obj = faces[0]
                
                # Check that detected face covers a significant portion of the crop
                # This catches false positives where detector finds a "face" in a small region of non-face
                det_bbox = face_obj.bbox  # [x1, y1, x2, y2]
                padded_h, padded_w = padded.shape[:2]
                det_w = det_bbox[2] - det_bbox[0]
                det_h = det_bbox[3] - det_bbox[1]
                det_area = det_w * det_h
                crop_area = padded_w * padded_h
                coverage = det_area / crop_area if crop_area > 0 else 0
                
                # If detected face is less than 40% of the crop, it's likely a false positive
                # (a real face crop should have the face covering most of the area)
                if coverage < 0.35:
                    return False, 0.0, f"false_positive(coverage={coverage:.0%})"
                
                # 1. Detection confidence score - hard reject low confidence detections
                det_score = face_obj.det_score if hasattr(face_obj, 'det_score') else 0.5
                if det_score < 0.6:
                    return False, 0.0, f"low_detection({det_score:.2f})"
                criteria_scores['detection'] = np.clip((det_score - 0.6) / 0.4, 0.0, 1.0)
                if det_score < 0.75:
                    issues.append(f"low_det({det_score:.2f})")
                
                # 2. Pose check (yaw, pitch) - want frontal face
                if hasattr(face_obj, 'pose'):
                    pose = face_obj.pose  # [pitch, yaw, roll] in degrees
                    yaw = abs(pose[1]) if len(pose) > 1 else 0
                    pitch = abs(pose[0]) if len(pose) > 0 else 0
                    
                    # Hard reject severe side profile (yaw > 35°)
                    if yaw > 35:
                        return False, 0.0, f"side_profile(yaw={yaw:.0f}°)"
                    # Yaw score: 0° = 1.0, 35°+ = 0.0
                    criteria_scores['pose_yaw'] = np.clip(1.0 - (yaw / 35.0), 0.0, 1.0)
                    if yaw > 20:
                        issues.append(f"tilted_yaw({yaw:.0f}°)")
                    
                    # Pitch score: 0° = 1.0, 35°+ = 0.0
                    criteria_scores['pose_pitch'] = np.clip(1.0 - (pitch / 35.0), 0.0, 1.0)
                    if pitch > 25:
                        issues.append(f"tilted_pitch({pitch:.0f}°)")
                
                # 3. Check landmarks quality - eyes, nose, mouth should be visible
                if hasattr(face_obj, 'kps') and face_obj.kps is not None:
                    kps = face_obj.kps  # 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
                    landmark_score = 1.0
                    
                    # Check if keypoints are within face bounds (not clipped)
                    padded_h, padded_w = padded.shape[:2]
                    margin_x = padded_w * 0.05  # 5% margin
                    margin_y = padded_h * 0.05
                    
                    # Check each keypoint - eyes (0,1) and nose (2) are critical
                    critical_kps_visible = 0
                    for i, kp in enumerate(kps[:3]):  # left_eye, right_eye, nose
                        in_bounds = (margin_x < kp[0] < padded_w - margin_x and 
                                     margin_y < kp[1] < padded_h - margin_y)
                        if in_bounds:
                            critical_kps_visible += 1
                    
                    # Hard reject if both eyes are not visible
                    if critical_kps_visible < 2:
                        return False, 0.0, "partial_face(missing_eyes)"
                    
                    # Penalize if not all critical landmarks visible
                    if critical_kps_visible < 3:
                        landmark_score -= 0.3
                        issues.append("partial_landmarks")
                    
                    # Check eye distance ratio (ideal: ~0.35, acceptable: 0.2-0.5)
                    eye_dist = np.linalg.norm(kps[0] - kps[1])
                    face_width = max(w, 1)
                    eye_ratio = eye_dist / face_width
                    
                    # Hard reject if eyes too close (likely false detection or extreme angle)
                    if eye_ratio < 0.15:
                        return False, 0.0, "invalid_face_geometry"
                    
                    ideal_ratio = 0.35
                    ratio_deviation = abs(eye_ratio - ideal_ratio) / 0.20
                    landmark_score -= min(ratio_deviation * 0.3, 0.4)
                    if eye_ratio < 0.2 or eye_ratio > 0.55:
                        issues.append("unusual_proportions")
                    
                    # Check vertical arrangement: eyes should be above nose
                    eye_y = (kps[0][1] + kps[1][1]) / 2
                    nose_y = kps[2][1]
                    if nose_y < eye_y:  # Nose above eyes = upside down or wrong detection
                        return False, 0.0, "invalid_face_orientation"
                    
                    criteria_scores['landmarks'] = np.clip(landmark_score, 0.0, 1.0)
                else:
                    # No keypoints available - cannot verify face structure
                    return False, 0.0, "no_landmarks"
                
            else:
                # MediaPipe path - use FaceLandmarker
                try:
                    # Ensure contiguous array
                    if not face_crop.flags['C_CONTIGUOUS']:
                        face_crop = np.ascontiguousarray(face_crop)
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
                    result = self.face_embedder.aligner.landmarker.detect(mp_image)
                    
                    if not result.face_landmarks:
                        return False, 0.0, "no_landmarks"
                    
                    landmarks = result.face_landmarks[0]
                    
                    # Check key landmark positions
                    def get_pt(idx):
                        pt = landmarks[idx]
                        return np.array([pt.x * w, pt.y * h])
                    
                    # Check if landmark is within the image bounds with margin
                    def is_visible(pt, margin_ratio=0.05):
                        margin_x = w * margin_ratio
                        margin_y = h * margin_ratio
                        return (margin_x < pt[0] < w - margin_x and 
                                margin_y < pt[1] < h - margin_y)
                    
                    # Eye landmarks
                    left_eye_inner = get_pt(133)
                    left_eye_outer = get_pt(33)
                    right_eye_inner = get_pt(362)
                    right_eye_outer = get_pt(263)
                    nose_tip = get_pt(1)
                    
                    left_eye_center = (left_eye_inner + left_eye_outer) / 2
                    right_eye_center = (right_eye_inner + right_eye_outer) / 2
                    
                    # HARD REJECT: Check that both eyes are visible in frame
                    left_eye_visible = is_visible(left_eye_center)
                    right_eye_visible = is_visible(right_eye_center)
                    if not (left_eye_visible and right_eye_visible):
                        return False, 0.0, "partial_face(eyes_not_visible)"
                    
                    # HARD REJECT: Check eye distance ratio (should be reasonable)
                    eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
                    eye_ratio = eye_dist / w
                    if eye_ratio < 0.15:
                        return False, 0.0, "invalid_face_geometry(eyes_too_close)"
                    if eye_ratio > 0.7:
                        return False, 0.0, "invalid_face_geometry(eyes_too_far)"
                    
                    # HARD REJECT: Check vertical arrangement - eyes must be above nose
                    eye_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    if nose_tip[1] < eye_y:
                        return False, 0.0, "invalid_face_orientation"
                    
                    # HARD REJECT: Eyes should be in top half of image for a proper face crop
                    if eye_y > h * 0.65:
                        return False, 0.0, "partial_face(eyes_too_low)"
                    
                    # 1. Head tilt score based on eye height difference
                    eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
                    max_acceptable_diff = h * 0.15
                    tilt_ratio = eye_height_diff / max_acceptable_diff
                    criteria_scores['pose_pitch'] = np.clip(1.0 - tilt_ratio, 0.0, 1.0)
                    if eye_height_diff > h * 0.10:
                        issues.append("tilted_head")
                    
                    # 2. Yaw score based on nose offset from eye center
                    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    nose_offset = abs(nose_tip[0] - eye_center_x) / w
                    # HARD REJECT if nose is way off center (extreme side profile)
                    if nose_offset > 0.25:
                        return False, 0.0, f"side_profile(nose_offset={nose_offset:.0%})"
                    max_acceptable_offset = 0.15
                    criteria_scores['pose_yaw'] = np.clip(1.0 - (nose_offset / max_acceptable_offset), 0.0, 1.0)
                    if nose_offset > 0.10:
                        issues.append(f"side_profile({nose_offset:.2f})")
                    
                    # 3. Eye openness score
                    left_eye_top = get_pt(159)
                    left_eye_bottom = get_pt(145)
                    left_ear = np.linalg.norm(left_eye_top - left_eye_bottom)
                    left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
                    
                    landmark_score = 1.0
                    if left_eye_width > 0:
                        left_eye_ratio = left_ear / left_eye_width
                        # Eye aspect ratio: 0.15 = closed (score 0), 0.3+ = open (score 1)
                        eye_openness = np.clip((left_eye_ratio - 0.15) / 0.15, 0.0, 1.0)
                        landmark_score *= (0.7 + 0.3 * eye_openness)
                        if left_eye_ratio < 0.15:
                            issues.append("eyes_closed")
                    
                    # 4. Face coverage score - face should fill most of the crop
                    all_x = [landmarks[i].x for i in range(len(landmarks))]
                    all_y = [landmarks[i].y for i in range(len(landmarks))]
                    face_span_x = max(all_x) - min(all_x)
                    face_span_y = max(all_y) - min(all_y)
                    
                    # HARD REJECT if face doesn't cover enough of the crop
                    if face_span_x < 0.3 or face_span_y < 0.4:
                        return False, 0.0, f"partial_face(span={face_span_x:.0%}x{face_span_y:.0%})"
                    
                    coverage_x = min(face_span_x / 0.6, 1.0)
                    coverage_y = min(face_span_y / 0.7, 1.0)
                    coverage_score = (coverage_x + coverage_y) / 2
                    landmark_score *= coverage_score
                    if face_span_x < 0.5 or face_span_y < 0.6:
                        issues.append("partial_face")
                    
                    criteria_scores['landmarks'] = np.clip(landmark_score, 0.0, 1.0)
                    # MediaPipe doesn't give detection confidence, assume good
                    criteria_scores['detection'] = 1.0
                    
                except Exception as e:
                    # Fallback - can't check landmarks, use neutral scores
                    criteria_scores['landmarks'] = 0.5
                    criteria_scores['detection'] = 0.5
            
            # Image quality checks
            # Convert to grayscale based on profile type
            if self.profile_type == "insightface":
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            
            # Sharpness score based on Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Hard reject only if extremely blurry (unusable)
            if laplacian_var < 15:
                return False, 0.0, "too_blurry"
            # Map: 15-100 = 0.0-1.0 linear, >100 = 1.0 (lowered thresholds)
            criteria_scores['sharpness'] = np.clip((laplacian_var - 15) / 85, 0.0, 1.0)
            if laplacian_var < 50:
                issues.append("blurry")
            
            # Brightness score (optimal: 80-180, falloff outside)
            mean_brightness = np.mean(gray)
            if mean_brightness < 80:
                criteria_scores['brightness'] = np.clip(mean_brightness / 80, 0.0, 1.0)
            elif mean_brightness > 180:
                criteria_scores['brightness'] = np.clip((255 - mean_brightness) / 75, 0.0, 1.0)
            else:
                criteria_scores['brightness'] = 1.0
            if mean_brightness < 40 or mean_brightness > 220:
                issues.append("bad_lighting")
            
            # Contrast score based on standard deviation (optimal: >40)
            std_brightness = np.std(gray)
            criteria_scores['contrast'] = np.clip(std_brightness / 40, 0.0, 1.0)
            if std_brightness < 25:
                issues.append("low_contrast")
            
        except Exception as e:
            # If analysis fails, be conservative with all scores
            for k in criteria_scores:
                criteria_scores[k] = 0.5
            issues.append("analysis_error")
        
        # Hard reject if any critical criterion is too low
        HARD_REJECT_THRESHOLDS = {
            'sharpness': 0.1,    # Reject if too blurry (lowered - most blur caught by laplacian)
            'pose_yaw': 0.30,     # Reject if too much side profile
            'landmarks': 0.35,    # Reject if landmarks are poor quality
            'detection': 0.25,    # Reject if detection confidence too low
        }
        for criterion, min_score in HARD_REJECT_THRESHOLDS.items():
            if criteria_scores[criterion] < min_score:
                return False, criteria_scores[criterion], f"failed_{criterion}"
        
        # Calculate weighted final score
        final_score = sum(criteria_scores[k] * WEIGHTS[k] for k in WEIGHTS)
        final_score = np.clip(final_score, 0.0, 1.0)
        
        # Threshold for enrollment (raised for quality)
        if final_score < 0.5:
            return False, float(final_score), "|".join(issues) if issues else "low_quality"
        
        return True, float(final_score), "|".join(issues) if issues else "ok"

    def _create_tracker(self):
        """Create an OpenCV tracker with fallback options."""
        # Try CSRT first (best quality), then KCF, then others
        tracker_creators = [
            # CSRT - best accuracy (requires opencv-contrib)
            lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create') else None,
            lambda: cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else None,
            lambda: cv2.TrackerCSRT.create() if hasattr(cv2, 'TrackerCSRT') else None,
            # KCF - good speed/accuracy balance (requires opencv-contrib)
            lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create') else None,
            lambda: cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else None,
            # MIL - usually available in base opencv
            lambda: cv2.legacy.TrackerMIL_create() if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMIL_create') else None,
            lambda: cv2.TrackerMIL_create() if hasattr(cv2, 'TrackerMIL_create') else None,
            lambda: cv2.TrackerMIL.create() if hasattr(cv2, 'TrackerMIL') else None,
        ]
        
        for creator in tracker_creators:
            try:
                tracker = creator()
                if tracker is not None:
                    return tracker
            except Exception:
                continue
        
        # If no tracker available, return None - we'll use IoU-only tracking
        return None

    def update_tracker(self, frame_bgr, detections):
        """
        Update trackers with current frame detections.
        Uses OpenCV tracker if available, otherwise falls back to IoU-only matching.
        
        Args:
            frame_bgr: Current frame in BGR format (for OpenCV)
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            List of tracks with format:
            [{'track_id': int, 'bbox': (x1, y1, x2, y2), 'is_new': bool}, ...]
        """
        h, w = frame_bgr.shape[:2]
        result = []
        matched_detection_indices = set()
        trackers_to_remove = []
        
        # Step 1: Update existing trackers and match with detections
        for track_id, track_data in list(self.csrt_trackers.items()):
            tracker = track_data.get('tracker')
            tracked_bbox = track_data['bbox']  # Use last known bbox as fallback
            tracker_success = False
            
            # Try to update with OpenCV tracker if available
            if tracker is not None:
                try:
                    success, bbox_xywh = tracker.update(frame_bgr)
                    if success:
                        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                        x, y, bw, bh = [int(v) for v in bbox_xywh]
                        x1, y1, x2, y2 = x, y, x + bw, y + bh
                        # Clamp to frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        tracked_bbox = (x1, y1, x2, y2)
                        tracker_success = True
                except Exception:
                    pass
            
            # Find best matching detection (by IoU)
            best_iou = 0.3  # Minimum IoU threshold
            best_det_idx = -1
            for det_idx, det_bbox in enumerate(detections):
                if det_idx in matched_detection_indices:
                    continue
                iou = calculate_iou(tracked_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Matched with detection - use detection bbox (more accurate)
                matched_detection_indices.add(best_det_idx)
                final_bbox = detections[best_det_idx]
                track_data['missed_frames'] = 0
                # Re-init tracker with detection bbox for better accuracy
                if tracker is not None:
                    dx1, dy1, dx2, dy2 = final_bbox
                    new_tracker = self._create_tracker()
                    if new_tracker is not None:
                        new_tracker.init(frame_bgr, (dx1, dy1, dx2 - dx1, dy2 - dy1))
                        track_data['tracker'] = new_tracker
            elif tracker_success:
                # No matching detection but tracker succeeded - use tracker prediction
                final_bbox = tracked_bbox
                track_data['missed_frames'] += 1
            else:
                # No detection match and tracker failed
                track_data['missed_frames'] += 1
                final_bbox = tracked_bbox
            
            track_data['bbox'] = final_bbox
            
            # Remove if missed too many frames
            if track_data['missed_frames'] > self.max_missed_frames:
                trackers_to_remove.append(track_id)
            else:
                result.append({
                    'track_id': track_id,
                    'bbox': final_bbox,
                    'is_new': False
                })
        
        # Step 2: Remove stale trackers and their identities
        for track_id in trackers_to_remove:
            del self.csrt_trackers[track_id]
            if track_id in self.track_identities:
                del self.track_identities[track_id]
        
        # Step 3: Create new trackers for unmatched detections
        for det_idx, det_bbox in enumerate(detections):
            if det_idx not in matched_detection_indices:
                x1, y1, x2, y2 = det_bbox
                new_tracker = self._create_tracker()
                if new_tracker is not None:
                    new_tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
                
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.csrt_trackers[track_id] = {
                    'tracker': new_tracker,  # May be None if no tracker available
                    'bbox': det_bbox,
                    'missed_frames': 0
                }
                
                result.append({
                    'track_id': track_id,
                    'bbox': det_bbox,
                    'is_new': True
                })
        
        return result

    def process_faces(self, frame_rgb, frame_count, previous_tracked_faces, enable_learning=False, learning_tracker=None, tracker=None):
        new_tracked_faces = []
        if self.profile_type == "insightface" and self.insight_engine is not None:
            # Unified pass: detect, align, embed in one
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            faces = self.insight_engine.process_frame(frame_bgr, self.detection_confidence)
            for face in faces:
                emb = face['embedding']
                name = "unknown"
                similarity = 0.0
                color = (0, 0, 255)
                
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                face_crop_bgr = frame_bgr[y1:y2, x1:x2]
                
                is_quality_ok, score, quality_reason = self.check_face_quality_for_enrollment(face_crop_bgr)
                
                if emb is not None:
                    result = smart_knn_recognition(self.face_db.index, emb, self.face_db.id_to_name)
                    name = result['name']
                    similarity = result['confidence']
                    
                    # If low similarity, try rotations on the face crop
                    if self.check_rotation and (name == "unknown" or similarity < 0.35):
                        if face_crop_bgr.size > 0 and face_crop_bgr.shape[0] > 20:
                            def insight_embed_func(crop):
                                # Use InsightFace to detect+embed on the rotated crop
                                try:
                                    # Pad the crop slightly to help detection
                                    h, w = crop.shape[:2]
                                    pad = int(max(h, w) * 0.1)
                                    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                                    
                                    rotated_faces = self.insight_engine.app.get(padded)
                                    if rotated_faces and len(rotated_faces) > 0:
                                        rec_emb = rotated_faces[0].embedding
                                        if rec_emb is not None:
                                            rec_result = smart_knn_recognition(self.face_db.index, rec_emb, self.face_db.id_to_name)
                                            return rec_emb, rec_result['name'], rec_result['confidence']
                                except Exception as e:
                                    pass
                                return None, "unknown", 0.0
                            
                            rot_emb, rot_name, rot_sim = self._try_rotated_embedding(
                                face_crop_bgr, insight_embed_func, min_threshold=0.35
                            )
                            if rot_sim > similarity:
                                emb, name, similarity = rot_emb, rot_name, rot_sim
                        
                    # Apply user's similarity threshold
                    if similarity < self.similarity_threshold:
                        name = "unknown"
                        color = (0, 0, 255)
                        
                    elif name != "unknown":
                        color = (0, 255, 0)
                        if result.get('is_ambiguous', False):
                            color = (0, 255, 255)
                    if enable_learning and learning_tracker is not None:
                        name, similarity, color = self._handle_learning(name, similarity, color, emb, None, learning_tracker)
                
                    new_tracked_faces.append({
                        'bbox': face['bbox'],
                        'name': name,
                        'similarity': similarity,
                        'color': color,
                        'embedding': emb,
                        'quality_ok': is_quality_ok,
                        'quality_score': score,
                        'quality_reason': quality_reason
                    })
        else:
            # MediaPipe path with CSRT tracking
            # Step 1: Detect faces
            current_detections = self.detect_faces(frame_rgb)
            
            # Step 2: Update tracker with detections (CSRT needs BGR)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            tracks = self.update_tracker(frame_bgr, current_detections)
            
            # Step 3: Process each track
            for track_info in tracks:
                track_id = track_info['track_id']
                bbox = track_info['bbox']
                x1, y1, x2, y2 = bbox
                
                # Clamp bbox to frame bounds
                h, w = frame_rgb.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bbox = (x1, y1, x2, y2)
                
                face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                if face_crop_rgb.size == 0 or face_crop_rgb.shape[0] < 20:
                    continue
                
                # Check if we already have a recognized identity for this track
                cached_identity = self.track_identities.get(track_id)
                
                # Use cached identity if available
                # For unknown faces, re-check every N frames in case they get enrolled
                RECHECK_UNKNOWN_INTERVAL = 30  # Re-check unknown faces every 30 frames
                
                use_cache = False
                if cached_identity is not None:
                    if cached_identity['name'] != "unknown":
                        # Known person - always use cache
                        use_cache = True
                    else:
                        # Unknown person - use cache but re-check periodically
                        frames_since_scan = cached_identity.get('frames_since_scan', 0) + 1
                        cached_identity['frames_since_scan'] = frames_since_scan
                        if frames_since_scan < RECHECK_UNKNOWN_INTERVAL:
                            use_cache = True
                        else:
                            # Time to re-check this unknown face
                            cached_identity['frames_since_scan'] = 0
                            use_cache = False
                
                if use_cache:
                    # Use cached identity, skip embedding
                    new_tracked_faces.append({
                        'bbox': bbox,
                        'name': cached_identity['name'],
                        'similarity': cached_identity['similarity'],
                        'color': cached_identity['color'],
                        'embedding': cached_identity.get('embedding'),
                        'track_id': track_id,
                        'quality_ok': cached_identity.get('quality_ok', True),
                        'quality_score': cached_identity.get('quality_score', 1.0),
                        'quality_reason': cached_identity.get('quality_reason', "")
                    })
                else:
                    # New track or unknown - run full embedding + recognition
                    is_quality_ok, score, quality_reason = self.check_face_quality_for_enrollment(face_crop_rgb)
                    
                    emb = self.face_embedder.embed(face_crop_rgb, use_aligner=self.use_alignment)
                    name = "unknown"
                    similarity = 0.0
                    color = (0, 0, 255)
                    is_ambiguous = False
                    
                    if emb is not None:
                        result = smart_knn_recognition(self.face_db.index, emb, self.face_db.id_to_name)
                        name = result['name']
                        similarity = result['confidence']
                        is_ambiguous = result.get('is_ambiguous', False)
                        
                        # If low similarity, try rotations on the face crop
                        if self.check_rotation and (name == "unknown" or similarity < 0.35):
                            def mediapipe_embed_func(crop):
                                r_emb = self.face_embedder.embed(crop, use_aligner=self.use_alignment)
                                if r_emb is not None:
                                    r_result = smart_knn_recognition(self.face_db.index, r_emb, self.face_db.id_to_name)
                                    return r_emb, r_result['name'], r_result['confidence']
                                return None, "unknown", 0.0
                            
                            rot_emb, rot_name, rot_sim = self._try_rotated_embedding(
                                face_crop_rgb, mediapipe_embed_func, min_threshold=0.35
                            )
                            if rot_sim > similarity:
                                emb, name, similarity = rot_emb, rot_name, rot_sim
                        
                        # Apply user's similarity threshold
                        if similarity < self.similarity_threshold:
                            name = "unknown"
                            color = (0, 0, 255)
                        elif name != "unknown":
                            color = (0, 255, 0)
                            if is_ambiguous:
                                color = (0, 255, 255)
                        
                        if enable_learning and learning_tracker is not None:
                            name, similarity, color = self._handle_learning(name, similarity, color, emb, face_crop_rgb, learning_tracker)
                    
                    # Cache the identity for this track
                    self.track_identities[track_id] = {
                        'name': name,
                        'similarity': similarity,
                        'color': color,
                        'embedding': emb,
                        'quality_ok': is_quality_ok,
                        'quality_score': score,
                        'quality_reason': quality_reason,
                        'frames_since_scan': 0  # For periodic re-checking of unknown faces
                    }
                    
                    new_tracked_faces.append({
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity,
                        'color': color,
                        'embedding': emb,
                        'track_id': track_id,
                        'quality_ok': is_quality_ok,
                        'quality_score': score,
                        'quality_reason': quality_reason
                    })
                    # elif best_iou > 0.1:
                    #     new_tracked_faces.append({
                    #         'bbox': bbox,
                    #         'name': "scanning...",
                    #         'similarity': 0.0,
                    #         'color': (200, 200, 200),
                    #         'embedding': None,
                    #         'quality_ok': True,
                    #         'quality_score': 1.0,
                    #         'quality_reason': ""
                    #     })
        
        return new_tracked_faces

    def _handle_learning(self, name, similarity, color, emb, face_crop_rgb, learning_tracker):
        # Rolling Update
        if name == "unknown" and self.rolling_update_enabled and self.teacher_system:
            try:
                teacher_emb = self.teacher_system['embedder'].embed(face_crop_rgb)
                if teacher_emb is not None:
                    teacher_res = smart_knn_recognition(self.teacher_system['db'].index, teacher_emb, self.teacher_system['db'].id_to_name)
                    if teacher_res['name'] != "unknown" and teacher_res['confidence'] > 0.75:
                        t_name = teacher_res['name']
                        print(f"Rolling Update: Teaching {t_name} to {self.current_model_name}")
                        self.face_db.add_face(emb, t_name)
                        self.face_db.prune_embeddings(t_name, max_faces=100)
                        self.face_db.save()
                        return t_name, teacher_res['confidence'], (0, 255, 0)
            except Exception as e:
                print(f"Rolling update error: {e}")

        # Adaptive Learning
        if name != "unknown" and similarity > self.learning_threshold:
            learning_tracker[name] = learning_tracker.get(name, 0) + 1
            if learning_tracker[name] > 5:
                if similarity < self.adaptive_threshold:
                    print(f"Updating template for {name} (sim: {similarity:.2f})...")
                    self.face_db.add_face(emb, name)
                    self.face_db.prune_embeddings(name, max_faces=100)
                    self.face_db.save()
                learning_tracker[name] = 0
        else:
            if name in learning_tracker:
                learning_tracker[name] = 0
        
        return name, similarity, color

    def draw_faces(self, frame, tracked_faces):
        for face in tracked_faces:
            x1, y1, x2, y2 = face['bbox']
            color = face['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{face['name']} ({face['similarity']:.2f})" if face['name'] != "scanning..." else face['name']
            if y1 > 20: cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else: cv2.putText(frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    def video_loop(self):
        fps_start_time = time.time()
        fps_counter = 0

        # Reset CSRT trackers for fresh video loop
        self.csrt_trackers = {}
        self.next_track_id = 0
        self.track_identities = {}  # Clear identity cache too
        
        while self.is_running:
            if self.is_loading_model:
                # Ensure camera is released while loading
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                blank_frame = np.zeros((self.tex_height, self.tex_width, 4), dtype=np.float32)
                dpg.set_value("video_texture", blank_frame)
                time.sleep(0.01)
                continue

            if self.cap is None:
                time.sleep(0.01)
                continue

            if self.is_enrolling:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            self.last_frame = frame.copy() # Store for enrollment
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FPS Calculation
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps = fps_counter / elapsed
                dpg.set_value("fps_text", f"FPS: {fps:.1f}")
                fps_counter = 0
                fps_start_time = time.time()

            # Process Faces
            self.tracked_faces = self.process_faces(
                frame_rgb, 
                self.frame_count, 
                self.tracked_faces, 
                enable_learning=True, 
                learning_tracker=self.learning_tracker, 
            )
            self.frame_count += 1

            # Draw results on frame
            self.draw_faces(frame, self.tracked_faces)

            # Update DPG Texture
            # DPG expects RGBA float32 0-1
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_float = frame_rgba.astype(np.float32) / 255.0
            dpg.set_value("video_texture", frame_float)
            time.sleep(0.01)

    def start_enrollment_process(self):
        if not self.tracked_faces:
            dpg.set_value("enroll_status", "No faces detected.")
            return
            
        self.is_enrolling = True
        self.enrolling_from_video = False
        self.enroll_queue = list(self.tracked_faces)
        if self.last_frame is not None:
            self.enroll_frame_base = self.last_frame.copy()
        else:
            # Fallback if no frame captured yet
            self.enroll_frame_base = np.zeros((self.tex_height, self.tex_width, 3), dtype=np.uint8)
            
        self.process_enroll_queue()

    def start_video_enrollment_process(self):
        if not self.video_tracked_faces:
            dpg.set_value("video_enroll_status", "No faces detected.")
            return
            
        self.is_enrolling = True
        self.enrolling_from_video = True
        self.enroll_queue = list(self.video_tracked_faces)
        if self.last_video_frame is not None:
            self.enroll_frame_base = self.last_video_frame.copy()
        else:
            self.enroll_frame_base = np.zeros((self.tex_height, self.tex_width, 3), dtype=np.uint8)
            
        self.process_enroll_queue()

    def process_enroll_queue(self):
        if not self.enroll_queue:
            self.is_enrolling = False
            dpg.configure_item("enroll_modal", show=False)
            if self.enrolling_from_video:
                dpg.set_value("video_enroll_status", "Enrollment finished.")
            else:
                dpg.set_value("enroll_status", "Enrollment finished.")
            return

        self.current_enroll_face = self.enroll_queue.pop(0)
        
        # Skip if no embedding
        if self.current_enroll_face.get('embedding') is None:
            self.process_enroll_queue()
            return

        # Update texture with highlight
        display_frame = self.enroll_frame_base.copy()
        x1, y1, x2, y2 = self.current_enroll_face['bbox']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Update DPG Texture
        frame_rgba = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
        frame_float = frame_rgba.astype(np.float32) / 255.0
        
        if self.enrolling_from_video:
            dpg.set_value("video_player_texture", frame_float)
        else:
            dpg.set_value("video_texture", frame_float)
        
        # Update Modal
        current_name = self.current_enroll_face['name']
        dpg.set_value("enroll_id_text", f"Identified as: {current_name}")
        dpg.set_value("enroll_input_name", current_name if current_name != "unknown" else "")
        dpg.configure_item("enroll_modal", show=True)

    def save_enrollment(self):
        name = dpg.get_value("enroll_input_name").strip()
        if not name:
            return # Or show error
            
        if self.current_enroll_face:
            # 1. Add the original detected embedding
            self.face_db.add_face(self.current_enroll_face['embedding'], name)
            
            # 2. Augmentation
            if dpg.get_value("chk_augment"):
                x1, y1, x2, y2 = self.current_enroll_face['bbox']
                # Ensure coords are within bounds
                h, w = self.enroll_frame_base.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = self.enroll_frame_base[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Convert to RGB for Albumentations and Embedder
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    # Generate augmentations
                    augment_count = dpg.get_value("input_augment_count") or 20
                    for _ in range(augment_count):
                        try:
                            augmented = self.aug_transform(image=face_crop_rgb)['image']
                            emb = None
                            if self.profile_type == "insightface" and self.insight_engine is not None:
                                # InsightFace requires BGR and full detection pass
                                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                                faces = self.insight_engine.process_frame(augmented_bgr, threshold=0.1)
                                if faces and faces[0].get('embedding') is not None:
                                    emb = faces[0]['embedding']
                            else:
                                # MediaPipe path
                                emb = self.face_embedder.embed(augmented, use_aligner=self.use_alignment)
                            if emb is not None:
                                self.face_db.add_face(emb, name)
                        except Exception as e:
                            print(f"Augmentation error: {e}")

            self.face_db.prune_embeddings(name, max_faces=100)
            self.face_db.save()
            self.refresh_db_list()
            
        self.process_enroll_queue()

    def skip_enrollment(self):
        self.process_enroll_queue()

    def cancel_enrollment(self):
        self.enroll_queue = []
        self.process_enroll_queue()

    def refresh_db_list(self):
        # Clear existing rows
        if dpg.does_item_exist("users_table"):
            # Delete all children of the table (rows)
            # DPG doesn't have a direct "clear table" so we delete children
            children = dpg.get_item_children("users_table", 1) # 1 is slot for rows
            if children:
                for child in children:
                    dpg.delete_item(child)
        
        # Aggregate data
        name_counts = {}
        for id_str, name in self.face_db.id_to_name.items():
            name_counts[name] = name_counts.get(name, 0) + 1
            
        for name, count in name_counts.items():
            with dpg.table_row(parent="users_table"):
                dpg.add_text(name)
                dpg.add_text(str(count))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Delete", user_data=name, callback=self.delete_user)
                    dpg.add_button(label="Prune (Max 50)", user_data=name, callback=self.prune_user)

    def delete_user(self, sender, app_data, user_data):
        name = user_data
        print(f"Deleting {name}...")
        
        # Find IDs
        ids_to_remove = [int(k) for k, v in self.face_db.id_to_name.items() if v == name]
        
        if ids_to_remove:
            self.face_db.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            for pid in ids_to_remove:
                del self.face_db.id_to_name[str(pid)]
            self.face_db.save()
            self.refresh_db_list()

    def prune_user(self, sender, app_data, user_data):
        name = user_data
        self.face_db.prune_embeddings(name, max_faces=50)
        self.face_db.save()
        self.refresh_db_list()

    def save_database(self):
        self.face_db.save()
        dpg.set_value("enroll_status", "Database saved.")

    def select_video_file(self, sender, app_data):
        self.video_path = app_data['file_path_name']
        dpg.set_value("video_path_text", f"Selected: {self.video_path}")

    def start_video(self, is_processed=False, video_path=None):
        path = video_path if video_path is not None else self.video_path
        if not path:
            return

        # Ensure previous thread is stopped
        if self.is_playing_video:
            self.stop_video()

        if self.video_thread and self.video_thread.is_alive():
            # Wait for it to finish, but don't block forever
            self.video_thread.join()
            if self.video_thread.is_alive():
                print("Warning: Previous video thread did not exit cleanly.")

        self.is_playing_video = True
        dpg.configure_item("btn_play_video", show=False)
        dpg.configure_item("btn_stop_video", show=True)

        # Open capture and immediately grab first frame to show UI quickly
        self.video_cap = cv2.VideoCapture(path)

        # Setup Progress Bar
        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dpg.configure_item("video_progress", max_value=self.video_total_frames)
        self.video_frame_count = 0

        # Try to read one frame to populate the texture quickly (non-blocking UI feel)
        try:
            ret, frame = self.video_cap.read()
            if ret:
                # show first frame immediately
                frame = cv2.resize(frame, (self.tex_width, self.tex_height))
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_float = frame_rgba.astype(np.float32) / 255.0
                dpg.set_value("video_player_texture", frame_float)
                self.last_video_frame = frame.copy()
                # reset position to the frame just read so playback proceeds consistently
                try:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_frame_count = 0
                except Exception:
                    pass
        except Exception:
            pass

        # record wall-clock start time for sync
        self.start_time = time.time()

        # start audio extraction/playback in background to avoid blocking UI
        def _async_audio_start(p):
            try:
                loaded = False
                if hasattr(self.audio_player, 'load_audio'):
                    loaded = self.audio_player.load_audio(p)
                if loaded:
                    try:
                        # set volume and start playing at current elapsed
                        self.audio_player.set_volume(self.volume)
                        # compute elapsed to start audio at correct position
                        elapsed = time.time() - self.start_time
                        # clamp elapsed
                        if elapsed < 0:
                            elapsed = 0.0
                        self.audio_player.play(start=elapsed)
                    except Exception as e:
                        print(f"Warning: audio play failed: {e}")
            except Exception as e:
                print(f"Warning: async audio start failed: {e}")

        try:
            threading.Thread(target=_async_audio_start, args=(path,), daemon=True).start()
        except Exception:
            pass

        # start video playback thread
        self.video_thread = threading.Thread(target=self.video_player_loop, args=(is_processed,), daemon=True)
        self.video_thread.start()

    def stop_video(self):
        self.is_playing_video = False
        
        # Only join if we are NOT in the video thread (avoid deadlock)
        if self.video_thread and self.video_thread.is_alive() and threading.current_thread() != self.video_thread:
             self.video_thread.join()

        with self.video_lock:
            if self.video_cap:
                self.video_cap.release()
            # stop and cleanup audio player
            if self.audio_player:
                try:
                    self.audio_player.stop()
                    self.audio_player.unload()
                except Exception:
                    pass
        dpg.configure_item("btn_play_video", show=True)
        dpg.configure_item("btn_stop_video", show=False)

    def video_player_loop(self, is_processed=False):
        # Wall-clock based sync
        if not self.video_cap or not self.video_cap.isOpened():
            dpg.set_value("process_status", "Failed to open video for playback")
            self.is_playing_video = False
            dpg.configure_item("btn_play_video", show=True)
            dpg.configure_item("btn_stop_video", show=False)
            return

        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Ensure start_time exists
        if not hasattr(self, 'start_time') or self.start_time is None:
            self.start_time = time.time()

        while self.is_playing_video:
            loop_start_time = time.time()

            # Wall clock elapsed since playback start
            elapsed = time.time() - self.start_time
            target_frame = int(elapsed * fps)

            # If target beyond end, stop
            if self.video_total_frames and target_frame >= self.video_total_frames:
                self.stop_video()
                break

            with self.video_lock:
                if not self.video_cap or not self.video_cap.isOpened():
                    break

                # If we are behind, fast-forward using grab to avoid decoding every skipped frame
                if self.video_frame_count < target_frame:
                    frames_to_skip = target_frame - self.video_frame_count
                    try:
                        # Use grab() for faster skipping when available
                        for _ in range(frames_to_skip - 1):
                            if not self.video_cap.grab():
                                break
                        # finally read the target frame
                        ret, frame = self.video_cap.read()
                        self.video_frame_count = target_frame
                    except Exception:
                        # fallback to set position
                        try:
                            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                            ret, frame = self.video_cap.read()
                            self.video_frame_count = target_frame
                        except Exception:
                            ret = False
                            frame = None
                else:
                    ret, frame = self.video_cap.read()

            if not ret:
                self.stop_video()
                break

            # Resize for display consistency
            frame = cv2.resize(frame, (self.tex_width, self.tex_height))
            self.last_video_frame = frame.copy()

            if not is_processed and self.detect_while_playback:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # If playback detection enabled, prefer processing every frame for responsiveness
                prev_skip = self.skip_frames
                try:
                    if self.detect_while_playback:
                        self.skip_frames = 0

                    # Process Faces
                    self.video_tracked_faces = self.process_faces(
                        frame_rgb,
                        self.video_frame_count,
                        self.video_tracked_faces,
                        enable_learning=False
                    )
                finally:
                    # restore previous skip setting
                    self.skip_frames = prev_skip

                # Draw results
                self.draw_faces(frame, self.video_tracked_faces)

            self.video_frame_count += 1

            # Update Texture
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_float = frame_rgba.astype(np.float32) / 255.0
            dpg.set_value("video_player_texture", frame_float)
            dpg.set_value("video_frame_text", f"Frame: {self.video_frame_count}/{self.video_total_frames}")
            dpg.set_value("video_progress", self.video_frame_count)

            # If we are ahead of wall-clock, sleep a bit
            elapsed_loop = time.time() - loop_start_time
            target_time_per_frame = 1.0 / fps
            desired_elapsed = (self.video_frame_count / fps) - (time.time() - self.start_time)
            if desired_elapsed > 0:
                # we're ahead, wait a small amount
                time.sleep(min(desired_elapsed, target_time_per_frame))
            else:
                # keep loop responsive
                time.sleep(0.001)

    def start_processing_video(self):
        if not self.video_path:
            dpg.set_value("process_status", "No video selected")
            return
        
        if self.is_processing:
            return

        self.is_processing = True
        dpg.configure_item("btn_process", show=False)
        dpg.configure_item("btn_stop_process", show=True)
        dpg.set_value("process_status", "Processing...")
        dpg.set_value("process_progress", 0.0)
        self.skipped_faces_embeddings = []
        
        self.process_thread = threading.Thread(target=self.process_video_loop, daemon=True)
        self.process_thread.start()

    def stop_processing_video(self):
        self.is_processing = False
        dpg.configure_item("btn_process", show=True)
        dpg.configure_item("btn_stop_process", show=False)
        dpg.set_value("process_status", "Stopped")

    def process_video_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.is_processing = False
            dpg.configure_item("btn_process", show=True)
            dpg.configure_item("btn_stop_process", show=False)
            dpg.set_value("process_status", "Failed to open video")
            return

        # Use source video's properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <=0 or height <=0:
            width, height = 1280, 720
        # Use source video's FPS to maintain audio sync
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:  # Sanity check
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path
        base, ext = os.path.splitext(self.video_path)
        self.processed_video_path = f"{base}_processed.mp4"
        
        # Video Writer - use mp4v codec (reliable across platforms)
        # We'll re-encode with ffmpeg at the end for proper H.264 + audio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.processed_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Warning: VideoWriter failed to open, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.processed_video_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
            self.processed_video_path = self.processed_video_path.replace('.mp4', '.avi')
        
        frame_idx = 0
        faces_at_frame = []
        frames_with_skipped_faces = []
        local_tracked_faces = []
        video_process_learning_tracker = {}
        
        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1280, 720))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            local_tracked_faces = self.process_faces(
                frame_rgb, 
                frame_idx, 
                local_tracked_faces, 
                enable_learning=False,
                learning_tracker=video_process_learning_tracker
            )
            
            # Check for unknown faces to label
            for face in local_tracked_faces:
                if face['name'] == "unknown" and face.get('embedding') is not None:
                    # Check if similar to skipped faces
                    is_skipped = False
                    for skipped_emb in self.skipped_faces_embeddings:
                        # Cosine similarity
                        sim = np.dot(face['embedding'], skipped_emb)
                        if sim > 0.85:
                            is_skipped = True
                            break
                        
                    if is_skipped or face['quality_ok'] == False:
                        if(len(frames_with_skipped_faces) and frames_with_skipped_faces[-1] != frame_idx):
                            frames_with_skipped_faces.append(frame_idx)
                        continue

                    # Pause and ask user
                    self.processing_paused_for_input = True
                    self.unknown_face_data = face
                    
                    h, w = frame_rgb.shape[:2]
                    x1, y1, x2, y2 = face['bbox']
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame_rgb[y1:y2, x1:x2]
                        
                        # Try to find the best orientation AND apply alignment for display
                        best_crop = face_crop
                        best_aligned = None
                        
                        if face_crop.size > 0:
                            best_score = -1  # Track best detection score
                            rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
                            
                            for rotation in rotations:
                                try:
                                    if rotation is None:
                                        rotated = face_crop
                                    else:
                                        rotated = cv2.rotate(face_crop, rotation)
                                    
                                    # Try to detect face in this orientation
                                    if self.profile_type == "insightface" and self.insight_engine is not None:
                                        rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
                                        pad = int(max(rotated.shape[:2]) * 0.1)
                                        padded = cv2.copyMakeBorder(rotated_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                                        faces_detected = self.insight_engine.app.get(padded)
                                        if faces_detected and len(faces_detected) > 0:
                                            # Face detected - use detection score as metric
                                            det_score = faces_detected[0].det_score if hasattr(faces_detected[0], 'det_score') else 0.5
                                            if det_score > best_score:
                                                best_score = det_score
                                                best_crop = rotated
                                                # Get aligned face from InsightFace
                                                if hasattr(faces_detected[0], 'normed_embedding'):
                                                    # The face object contains the aligned face
                                                    pass
                                                # Try to get aligned face crop
                                                try:
                                                    kps = faces_detected[0].kps
                                                    if kps is not None:
                                                        # Use InsightFace's built-in alignment
                                                        from insightface.utils import face_align
                                                        aligned = face_align.norm_crop(padded, kps, image_size=112)
                                                        best_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                                                except Exception:
                                                    pass
                                    else:
                                        # MediaPipe path - use aligner to check for face and get aligned
                                        try:
                                            aligned = self.face_embedder.aligner.align_face(rotated)
                                            # If alignment didn't just resize (detected landmarks), it's a valid face
                                            # Check if we got actual alignment by comparing aspect ratios
                                            # A successful alignment means landmarks were found
                                            test_emb = self.face_embedder.model.get_feat(aligned).flatten()
                                            if test_emb is not None:
                                                # Use embedding norm as a quality metric
                                                norm = np.linalg.norm(test_emb)
                                                if norm > best_score:
                                                    best_score = norm
                                                    best_crop = rotated
                                                    best_aligned = aligned
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        
                        # Use aligned face for display if available, otherwise use best rotated crop
                        if best_aligned is not None:
                            self.unknown_face_data['frame_crop'] = best_aligned
                        else:
                            self.unknown_face_data['frame_crop'] = best_crop
                        
                        # Update Texture
                        if self.unknown_face_data['frame_crop'].size > 0:
                            face_img = cv2.resize(self.unknown_face_data['frame_crop'], (150, 150))
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2RGBA)
                            face_img = face_img.astype(np.float32) / 255.0
                            dpg.set_value("unknown_face_texture", face_img)
                    else:
                        # Fallback if face is invalid/zero-size
                        print(f"Skipping invalid crop: {x1},{y1},{x2},{y2}")
                        # Optional: Set texture to black/empty to prevent confusion
                        empty_face = np.zeros((150, 150, 4), dtype=np.float32)
                        dpg.set_value("unknown_face_texture", empty_face)
                    
                    dpg.configure_item("label_modal", show=True)
                    dpg.set_value("label_input_name", "")
                    
                    # Wait for user input
                    while self.processing_paused_for_input and self.is_processing:
                        time.sleep(0.1)
                    
                    # After input, re-identify this face immediately so it's labeled in this frame
                    if self.unknown_face_data and self.unknown_face_data.get('name') != "unknown":
                         face['name'] = self.unknown_face_data['name']
                         face['color'] = (0, 255, 0)

            # Draw
            self.draw_faces(frame, local_tracked_faces)
            
            try:
                preview_frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGBA)
                preview_float = preview_frame.astype(np.float32) / 255.0
                dpg.set_value("process_preview_texture", preview_float)
            except Exception:
                pass
            
            faces_at_frame.append(local_tracked_faces)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                dpg.set_value("process_progress", progress)
                dpg.set_value("process_status", f"Processing: {int(progress*100)}%")

        dpg.set_value("process_status", "Finalizing video...")
        # Reset video capture to beginning for second pass
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Fix passed frames with latest knowledge
        frame_idx = 0
        frames_with_skipped_faces_index = 0
        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized_frame = cv2.resize(frame, (1280, 720))
            
            if len(frames_with_skipped_faces) and frame_idx == frames_with_skipped_faces[frames_with_skipped_faces_index]:
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                local_tracked_faces = self.process_faces(
                    frame_rgb, 
                    frame_idx, 
                    local_tracked_faces, 
                    enable_learning=False
                )
                
                try:
                    preview_frame = cv2.resize(resized_frame, (self.preview_width, self.preview_height))
                    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGBA)
                    preview_float = preview_frame.astype(np.float32) / 255.0
                    dpg.set_value("process_preview_texture", preview_float)
                except Exception as e:
                    print(e)
               
                frames_with_skipped_faces_index += 1 
            else:
                local_tracked_faces = faces_at_frame[frame_idx]
                
            for idx, faces in enumerate(local_tracked_faces):
                # Draw to original frame with original resolution
                x1, y1, x2, y2 = faces['bbox']
                x1 = int(x1 * (width / 1280))
                y1 = int(y1 * (height / 720))
                x2 = int(x2 * (width / 1280))
                y2 = int(y2 * (height / 720))
                local_tracked_faces[idx]['bbox'] = (x1, y1, x2, y2)
                
            self.draw_faces(frame, local_tracked_faces)
                
            out.write(cv2.resize(frame, (int(width), int(height)), ))
            frame_idx += 1

        cap.release()
        out.release()
        faces_at_frame = []
        frames_with_skipped_faces = []
        
        # Merge Audio if ffmpeg is available
        if self.is_processing: # Only if finished naturally
            dpg.set_value("process_status", "Merging Audio...")
            audio_merged = False
            
            # 1. Try ffmpeg - re-encode to H.264 and add audio
            try:
                import subprocess
                temp_output = f"{base}_temp_audio.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", self.processed_video_path,  # Video (may be mp4v or XVID)
                    "-i", self.video_path,             # Original with audio
                    "-map", "0:v:0",                   # Video from first input
                    "-map", "1:a:0?",                  # Audio from second input (optional)
                    "-c:v", "libx264",                 # Re-encode to H.264 for compatibility
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-vsync", "cfr",                   # Constant frame rate for sync
                    "-c:a", "aac",
                    "-async", "1",                     # Audio sync at start
                    "-shortest",
                    temp_output
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"ffmpeg stderr: {result.stderr}")
                    raise Exception(f"ffmpeg returned {result.returncode}")
                
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    os.replace(temp_output, self.processed_video_path)
                    audio_merged = True
            except FileNotFoundError:
                print("ffmpeg not found in PATH")
            except Exception as e:
                print(f"ffmpeg merge failed: {e}")

            # 2. Try moviepy if ffmpeg failed
            if not audio_merged:
                dpg.set_value("process_status", "Trying moviepy...")
                try:
                    # Load clips
                    original_clip = VideoFileClip(self.video_path)
                    processed_clip = VideoFileClip(self.processed_video_path)
                    
                    if original_clip.audio:
                        final_clip = processed_clip.with_audio(original_clip.audio)
                        temp_output = f"{base}_temp_moviepy.mp4"
                        final_clip.write_videofile(temp_output, codec='libx264', audio_codec='aac', logger=None)
                        
                        original_clip.close()
                        processed_clip.close()
                        final_clip.close()
                        
                        if os.path.exists(temp_output):
                            os.replace(temp_output, self.processed_video_path)
                            audio_merged = True
                    else:
                        original_clip.close()
                        processed_clip.close()
                        print("No audio in original video")
                        
                except ImportError:
                    print("moviepy not installed")
                except Exception as e:
                    print(f"moviepy merge failed: {e}")

            if not audio_merged:
                 dpg.set_value("process_status", "Audio merge failed (ffmpeg/moviepy missing)")

        self.is_processing = False
        
        dpg.configure_item("btn_process", show=True)
        dpg.configure_item("btn_stop_process", show=False)
        dpg.set_value("process_status", "Complete!")
        dpg.set_value("process_progress", 1.0)
        dpg.configure_item("btn_play_result", show=True)

    def save_label_unknown(self):
        name = dpg.get_value("label_input_name").strip()
        if name and self.unknown_face_data:
            # Add to DB
            self.face_db.add_face(self.unknown_face_data['embedding'], name)
            self.face_db.save()
            self.unknown_face_data['name'] = name # Update local data to reflect change
            self.refresh_db_list()
            
        dpg.configure_item("label_modal", show=False)
        self.processing_paused_for_input = False

    def skip_label_unknown(self):
        if self.unknown_face_data and self.unknown_face_data.get('embedding') is not None:
            self.skipped_faces_embeddings.append(self.unknown_face_data['embedding'])
            
        dpg.configure_item("label_modal", show=False)
        self.processing_paused_for_input = False

    def play_processed_video(self):
        if not self.processed_video_path or not os.path.exists(self.processed_video_path):
            dpg.set_value("process_status", "Processed file not found")
            return
        self.start_video(is_processed=True, video_path=self.processed_video_path)

    def _force_exit(self, sender, app_data):
        try:
            self.stop_camera()
        except Exception:
            pass
        os._exit(0)

    def run(self):
        dpg.set_exit_callback(self._force_exit)
        dpg.start_dearpygui()
        dpg.destroy_context()
        self.stop_camera()
        os._exit(0)

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
