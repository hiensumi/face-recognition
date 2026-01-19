import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.model_zoo import get_model

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_path="./models/blaze_face_short_range.tflite"):
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path
        self._init_detector()

    def _init_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def set_confidence(self, min_detection_confidence):
        self.min_detection_confidence = min_detection_confidence
        self._init_detector()

    def detect(self, rgb_frame):
        # New API requires mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

    def detect_improved(self, rgb_frame):
        faces = []
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        h, w = rgb_frame.shape[:2]

        for detection in detection_result.detections:
            # New API returns absolute pixel coordinates (bbox.origin_x, bbox.origin_y)
            bbox = detection.bounding_box
            
            expand_x = bbox.width * 0.2
            expand_y = bbox.height * 0.2
            
            # Use pixel values directly
            x1 = int(max(0, bbox.origin_x - expand_x/2))
            y1 = int(max(0, bbox.origin_y - expand_y/2))
            x2 = int(min(w, bbox.origin_x + bbox.width + expand_x/2))
            y2 = int(min(h, bbox.origin_y + bbox.height + expand_y/2))

            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(detection.categories[0].score),
                'landmarks': None,
                'embedding': None
            })
        return faces

class FaceAligner:
    def __init__(self, model_path="./models/face_landmarker.task"):
        """
        Requires 'face_landmarker.task' in the project root.
        This replaces the legacy FaceMesh solution.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.75,
            min_face_presence_confidence=0.75
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.arcface_dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        self.landmark_indices = {
            'left_eye_corners': [33, 133],
            'right_eye_corners': [362, 263],
            'nose_tip': 1,
            'left_mouth': 61,
            'right_mouth': 291
        }

    def align_face(self, face_rgb):
        try:
            # Ensure array is contiguous (required by MediaPipe)
            if not face_rgb.flags['C_CONTIGUOUS']:
                face_rgb = np.ascontiguousarray(face_rgb)
            
            # 1. Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
            
            # 2. Detect landmarks
            result = self.landmarker.detect(mp_image)
            
            # 3. Check if face found (result.face_landmarks is a list of lists)
            if not result.face_landmarks:
                return cv2.resize(face_rgb, (112, 112))
            
            # Get the first face's landmarks
            landmarks = result.face_landmarks[0]
            h, w = face_rgb.shape[:2]
            
            def get_pt(idx):
                # Note: Landmarker still returns NormalizedLandmark (0.0 - 1.0)
                pt = landmarks[idx]
                return np.array([pt.x * w, pt.y * h], dtype=np.float32)

            le_c = np.mean([get_pt(idx) for idx in self.landmark_indices['left_eye_corners']], axis=0)
            re_c = np.mean([get_pt(idx) for idx in self.landmark_indices['right_eye_corners']], axis=0)
            nose = get_pt(self.landmark_indices['nose_tip'])
            lm = get_pt(self.landmark_indices['left_mouth'])
            rm = get_pt(self.landmark_indices['right_mouth'])
            
            src_points = np.array([le_c, re_c, nose, lm, rm], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src_points, self.arcface_dst)
            
            if M is None:
                return cv2.resize(face_rgb, (112, 112))
            
            aligned = cv2.warpAffine(face_rgb, M, (112, 112), flags=cv2.INTER_CUBIC)
            return aligned
            
        except Exception as e:
            # print(f"Align Error: {e}") # Debug if needed
            return cv2.resize(face_rgb, (112, 112))

class FaceEmbedder:
    def __init__(self, model_path="./models/arcface_w600k_r50.onnx"):
        self.model = get_model(model_path)
        self.model.prepare(ctx_id=0)
        self.aligner = FaceAligner() # Uses the new Task-based Aligner

    def embed(self, face_rgb, use_aligner=True):
        try:
            if use_aligner:
                aligned_face_rgb = self.aligner.align_face(face_rgb)
            else:
                aligned_face_rgb = cv2.resize(face_rgb, (112, 112))
            
            # Note: Despite documentation saying BGR, the old working code used RGB directly
            # and it worked correctly. Keeping RGB input for consistency with training data.
            emb = self.model.get_feat(aligned_face_rgb).flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            return None