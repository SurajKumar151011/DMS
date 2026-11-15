# detector.py
import cv2
import numpy as np
import mediapipe as mp
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
from typing import Tuple, Any

def initialize_face_mesh():
    """Initialize MediaPipe Face Mesh model safely with stable detection confidence."""
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

def process_frame(frame: np.ndarray, face_mesh: Any) -> Tuple[np.ndarray, Any]:
    """Process frame through MediaPipe Face Mesh."""
    if frame is None:
        return frame, None
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    return frame, results