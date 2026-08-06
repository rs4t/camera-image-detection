import os
import argparse
import time
import urllib.request
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# Face mesh landmark indices used for simple expression heuristics.
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


BACKEND_MAP = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

MODEL_URLS = {
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def to_px(landmark, w: int, h: int) -> np.ndarray:
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def detect_shocked(face_landmarks, frame_w: int, frame_h: int) -> bool:
    points = face_landmarks

    mouth_top = to_px(points[MOUTH_TOP], frame_w, frame_h)
    mouth_bottom = to_px(points[MOUTH_BOTTOM], frame_w, frame_h)

    left_eye_top = to_px(points[LEFT_EYE_TOP], frame_w, frame_h)
    left_eye_bottom = to_px(points[LEFT_EYE_BOTTOM], frame_w, frame_h)
    right_eye_top = to_px(points[RIGHT_EYE_TOP], frame_w, frame_h)
    right_eye_bottom = to_px(points[RIGHT_EYE_BOTTOM], frame_w, frame_h)

    left_eye_outer = to_px(points[LEFT_EYE_OUTER], frame_w, frame_h)
    right_eye_outer = to_px(points[RIGHT_EYE_OUTER], frame_w, frame_h)

    face_scale = distance(left_eye_outer, right_eye_outer) + 1e-6
    mouth_open_ratio = distance(mouth_top, mouth_bottom) / face_scale
    left_eye_open = distance(left_eye_top, left_eye_bottom) / face_scale
    right_eye_open = distance(right_eye_top, right_eye_bottom) / face_scale
    avg_eye_open_ratio = (left_eye_open + right_eye_open) / 2.0

    # Tuned for a clear surprised / shocked look.
    shocked = mouth_open_ratio > 0.18 and avg_eye_open_ratio > 0.055
    return shocked


def detect_eyes_closed(face_landmarks, frame_w: int, frame_h: int) -> bool:
    points = face_landmarks

    left_eye_top = to_px(points[LEFT_EYE_TOP], frame_w, frame_h)
    left_eye_bottom = to_px(points[LEFT_EYE_BOTTOM], frame_w, frame_h)
    right_eye_top = to_px(points[RIGHT_EYE_TOP], frame_w, frame_h)
    right_eye_bottom = to_px(points[RIGHT_EYE_BOTTOM], frame_w, frame_h)

    left_eye_outer = to_px(points[LEFT_EYE_OUTER], frame_w, frame_h)
    right_eye_outer = to_px(points[RIGHT_EYE_OUTER], frame_w, frame_h)
    face_scale = distance(left_eye_outer, right_eye_outer) + 1e-6

    left_eye_open = distance(left_eye_top, left_eye_bottom) / face_scale
    right_eye_open = distance(right_eye_top, right_eye_bottom) / face_scale
    avg_eye_open_ratio = (left_eye_open + right_eye_open) / 2.0
    return avg_eye_open_ratio < 0.030


def detect_tongue(face_landmarks, frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    points = face_landmarks

    mouth_top = to_px(points[MOUTH_TOP], w, h)
    mouth_bottom = to_px(points[MOUTH_BOTTOM], w, h)
    mouth_left = to_px(points[MOUTH_LEFT], w, h)
    mouth_right = to_px(points[MOUTH_RIGHT], w, h)

    mouth_width = distance(mouth_left, mouth_right) + 1e-6
    mouth_open = distance(mouth_top, mouth_bottom)
    mouth_open_ratio = mouth_open / mouth_width
    if mouth_open_ratio < 0.42:
        return False

    x0 = int(max(0, min(mouth_left[0], mouth_right[0]) - 0.10 * mouth_width))
    x1 = int(min(w, max(mouth_left[0], mouth_right[0]) + 0.10 * mouth_width))
    y0 = int(max(0, mouth_top[1]))
    y1 = int(min(h, mouth_bottom[1] + 0.70 * mouth_open))
    if x1 <= x0 or y1 <= y0:
        return False

    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 40), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 40), (179, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    return red_ratio > 0.10


def normalize_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def gesture_to_action(gesture_name: str) -> str | None:
    label = normalize_label(gesture_name)
    mapping = {
        "thumb_up": "thumbs_up",
        "thumbs_up": "thumbs_up",
        "victory": "victory",
        "pointing_up": "pointing_up",
    }
    return mapping.get(label)


def get_face_center_xy(face_landmarks) -> Tuple[float, float]:
    left_eye = face_landmarks[LEFT_EYE_OUTER]
    right_eye = face_landmarks[RIGHT_EYE_OUTER]
    nose = face_landmarks[NOSE_TIP]
    cx = float((left_eye.x + right_eye.x + nose.x) / 3.0)
    cy = float((left_eye.y + right_eye.y + nose.y) / 3.0)
    return cx, cy


def get_mouth_center_xy(face_landmarks) -> Tuple[float, float]:
    mouth_top = face_landmarks[MOUTH_TOP]
    mouth_bottom = face_landmarks[MOUTH_BOTTOM]
    mouth_left = face_landmarks[MOUTH_LEFT]
    mouth_right = face_landmarks[MOUTH_RIGHT]
    cx = float((mouth_top.x + mouth_bottom.x + mouth_left.x + mouth_right.x) / 4.0)
    cy = float((mouth_top.y + mouth_bottom.y + mouth_left.y + mouth_right.y) / 4.0)
    return cx, cy


def is_pointing_at_self(hand_landmarks, face_center_xy: Tuple[float, float] | None) -> bool:
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]

    if face_center_xy is None:
        return False

    face_cx, face_cy = face_center_xy

    near_face_center_x = abs(index_tip.x - face_cx) < 0.15
    hand_below_face = wrist.y > face_cy + 0.08
    finger_above_wrist = index_tip.y < wrist.y - 0.05
    return near_face_center_x and hand_below_face and finger_above_wrist


def hands_touching_head(hand_landmarks_list, face_landmarks) -> bool:
    if len(hand_landmarks_list) < 2:
        return False

    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    face_w = max_x - min_x
    face_h = max_y - min_y

    # Head contact zone around upper/sides of face (intentionally relaxed).
    zone_min_x = min_x - 0.45 * face_w
    zone_max_x = max_x + 0.45 * face_w
    zone_min_y = min_y - 0.50 * face_h
    zone_max_y = min_y + 0.85 * face_h

    touching_count = 0
    for hand in hand_landmarks_list:
        palm_points = [hand[0], hand[5], hand[17], hand[9]]
        palm_x = float(sum(p.x for p in palm_points) / len(palm_points))
        palm_y = float(sum(p.y for p in palm_points) / len(palm_points))

        palm_in_zone = zone_min_x <= palm_x <= zone_max_x and zone_min_y <= palm_y <= zone_max_y
        fingertip_in_zone = any(
            zone_min_x <= hand[idx].x <= zone_max_x and zone_min_y <= hand[idx].y <= zone_max_y
            for idx in (4, 8, 12, 16, 20)
        )
        in_zone = palm_in_zone or fingertip_in_zone
        if in_zone:
            touching_count += 1

    return touching_count >= 2


def lm_distance(hand_landmarks, i: int, j: int) -> float:
    p1 = hand_landmarks[i]
    p2 = hand_landmarks[j]
    return float(np.hypot(p1.x - p2.x, p1.y - p2.y))


def is_ok_gesture(hand_landmarks) -> bool:
    hand_scale = lm_distance(hand_landmarks, 0, 9) + 1e-6
    thumb_index_touch = lm_distance(hand_landmarks, 4, 8) < 0.35 * hand_scale
    middle_extended = is_finger_extended(hand_landmarks, 12, 10, margin=0.01)
    ring_extended = is_finger_extended(hand_landmarks, 16, 14, margin=0.01)
    pinky_extended = is_finger_extended(hand_landmarks, 20, 18, margin=0.01)
    return thumb_index_touch and middle_extended and ring_extended and pinky_extended


def hand_orientation(hand_landmarks) -> str:
    wrist = hand_landmarks[0]
    middle_tip = hand_landmarks[12]
    dx = abs(middle_tip.x - wrist.x)
    dy = abs(middle_tip.y - wrist.y)
    if dy > dx * 1.3:
        return "vertical"
    if dx > dy * 1.3:
        return "horizontal"
    return "diagonal"


def is_t_sign(hand_a, hand_b) -> bool:
    orient_a = hand_orientation(hand_a)
    orient_b = hand_orientation(hand_b)
    has_vertical = orient_a == "vertical" or orient_b == "vertical"
    has_horizontal = orient_a == "horizontal" or orient_b == "horizontal"
    if not (has_vertical and has_horizontal):
        return False

    wrist_gap = lm_distance(hand_a, 0, 0) if hand_a is hand_b else float(
        np.hypot(hand_a[0].x - hand_b[0].x, hand_a[0].y - hand_b[0].y)
    )
    return wrist_gap < 0.55


def is_two_hand_heart(hand_a, hand_b) -> bool:
    wrists_dist = float(np.hypot(hand_a[0].x - hand_b[0].x, hand_a[0].y - hand_b[0].y)) + 1e-6
    thumbs_close = float(np.hypot(hand_a[4].x - hand_b[4].x, hand_a[4].y - hand_b[4].y)) < 0.48 * wrists_dist
    index_close = float(np.hypot(hand_a[8].x - hand_b[8].x, hand_a[8].y - hand_b[8].y)) < 0.48 * wrists_dist

    tips = np.array(
        [
            [hand_a[4].x, hand_a[4].y],
            [hand_a[8].x, hand_a[8].y],
            [hand_b[4].x, hand_b[4].y],
            [hand_b[8].x, hand_b[8].y],
        ],
        dtype=np.float32,
    )
    cluster_w = float(np.max(tips[:, 0]) - np.min(tips[:, 0]))
    cluster_h = float(np.max(tips[:, 1]) - np.min(tips[:, 1]))
    compact_cluster = cluster_w < 0.60 * wrists_dist and cluster_h < 0.60 * wrists_dist
    tips_center_y = float(np.mean(tips[:, 1]))
    wrists_center_y = float((hand_a[0].y + hand_b[0].y) / 2.0)
    tips_above_wrists = tips_center_y < wrists_center_y
    return thumbs_close and index_close and compact_cluster and tips_above_wrists


def is_finger_extended(hand_landmarks, tip_idx: int, pip_idx: int, margin: float = 0.02) -> bool:
    return hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y - margin


def is_finger_curled(hand_landmarks, tip_idx: int, pip_idx: int, margin: float = 0.02) -> bool:
    return hand_landmarks[tip_idx].y > hand_landmarks[pip_idx].y + margin


def is_middle_finger_gesture(hand_landmarks) -> bool:
    # Thumb is intentionally ignored: middle finger should count with thumb in or out.
    middle_extended = is_finger_extended(hand_landmarks, 12, 10, margin=0.015)
    index_not_extended = not is_finger_extended(hand_landmarks, 8, 6, margin=0.005)
    ring_not_extended = not is_finger_extended(hand_landmarks, 16, 14, margin=0.005)
    pinky_not_extended = not is_finger_extended(hand_landmarks, 20, 18, margin=0.005)
    return middle_extended and index_not_extended and ring_not_extended and pinky_not_extended


def is_closed_fist_landmark(hand_landmarks) -> bool:
    hand_scale = lm_distance(hand_landmarks, 0, 9) + 1e-6
    tip_to_wrist = [
        lm_distance(hand_landmarks, 8, 0),
        lm_distance(hand_landmarks, 12, 0),
        lm_distance(hand_landmarks, 16, 0),
        lm_distance(hand_landmarks, 20, 0),
    ]
    curled = sum(1 for d in tip_to_wrist if d < 0.95 * hand_scale)
    index_not_extended = not is_finger_extended(hand_landmarks, 8, 6, margin=0.005)
    middle_not_extended = not is_finger_extended(hand_landmarks, 12, 10, margin=0.005)
    ring_not_extended = not is_finger_extended(hand_landmarks, 16, 14, margin=0.005)
    pinky_not_extended = not is_finger_extended(hand_landmarks, 20, 18, margin=0.005)
    mostly_closed = index_not_extended and middle_not_extended and ring_not_extended and pinky_not_extended
    return mostly_closed and curled >= 3


def is_fist_up(hand_landmarks) -> bool:
    wrist = hand_landmarks[0]
    knuckles_y = [hand_landmarks[idx].y for idx in (5, 9, 13, 17)]
    return float(np.mean(knuckles_y)) < wrist.y - 0.02


def is_shush_gesture(hand_landmarks, mouth_center_xy: Tuple[float, float] | None) -> bool:
    if mouth_center_xy is None:
        return False

    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    index_pip = hand_landmarks[6]

    index_extended = is_finger_extended(hand_landmarks, 8, 6, margin=0.01)
    middle_not_extended = not is_finger_extended(hand_landmarks, 12, 10, margin=0.005)
    ring_not_extended = not is_finger_extended(hand_landmarks, 16, 14, margin=0.005)
    pinky_not_extended = not is_finger_extended(hand_landmarks, 20, 18, margin=0.005)

    finger_vertical = abs(index_tip.y - index_pip.y) > abs(index_tip.x - index_pip.x)
    finger_pointing_up = index_tip.y < index_pip.y

    mouth_x, mouth_y = mouth_center_xy
    near_mouth_x = abs(index_tip.x - mouth_x) < 0.10
    near_mouth_y = abs(index_tip.y - mouth_y) < 0.10
    hand_below_tip = wrist.y > index_tip.y

    return (
        index_extended
        and middle_not_extended
        and ring_not_extended
        and pinky_not_extended
        and finger_vertical
        and finger_pointing_up
        and near_mouth_x
        and near_mouth_y
        and hand_below_tip
    )


def choose_action(gesture_result, face_result) -> str | None:
    labels: list[str] = []
    pointing_hand_index: int | None = None

    if gesture_result.gestures:
        for idx, gesture_list in enumerate(gesture_result.gestures):
            if not gesture_list:
                continue
            label = normalize_label(gesture_list[0].category_name)
            labels.append(label)
            if label == "pointing_up" and pointing_hand_index is None:
                pointing_hand_index = idx

    hands = gesture_result.hand_landmarks or []

    mouth_center_xy: Tuple[float, float] | None = None
    if face_result.face_landmarks:
        mouth_center_xy = get_mouth_center_xy(face_result.face_landmarks[0])

    for hand in hands:
        if is_shush_gesture(hand, mouth_center_xy):
            return "shh"

    closed_fists = [hand for hand in hands if is_closed_fist_landmark(hand)]
    fists_up = [hand for hand in closed_fists if is_fist_up(hand)]

    if len(fists_up) >= 2:
        return "yeah"
    if len(closed_fists) == 1:
        return "angry"

    if len(hands) >= 2 and is_two_hand_heart(hands[0], hands[1]):
        return "heart"

    if len(hands) >= 2 and is_t_sign(hands[0], hands[1]):
        return "pause"

    middle_count = 0
    ok_count = 0
    for hand_landmarks in hands:
        if is_middle_finger_gesture(hand_landmarks):
            middle_count += 1
        if is_ok_gesture(hand_landmarks):
            ok_count += 1
    if middle_count >= 2:
        return "middle_double"
    if middle_count == 1:
        return "middle_single"

    if ok_count >= 1:
        return "ok"

    open_palm_count = sum(1 for name in labels if name == "open_palm")
    if open_palm_count >= 2:
        return "cinema"

    face_center_xy: Tuple[float, float] | None = None
    if face_result.face_landmarks:
        face_center_xy = get_face_center_xy(face_result.face_landmarks[0])

    if (
        pointing_hand_index is not None
        and gesture_result.hand_landmarks
        and pointing_hand_index < len(gesture_result.hand_landmarks)
    ):
        if is_pointing_at_self(gesture_result.hand_landmarks[pointing_hand_index], face_center_xy):
            return "pointing_self"

    if open_palm_count == 1:
        return "stop"

    for label in labels:
        action = gesture_to_action(label)
        if action is not None:
            return action

    return None


def draw_landmark_points(
    frame: np.ndarray,
    landmarks,
    color: Tuple[int, int, int],
    radius: int,
    every_n: int,
) -> None:
    h, w = frame.shape[:2]
    for idx, lm in enumerate(landmarks):
        if idx % every_n != 0:
            continue
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)


def find_image_path(folder: str, base_name: str) -> str | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def fit_image_to_panel(image: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]
    scale = min(panel_w / img_w, panel_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def make_placeholder(label: str, panel_w: int, panel_h: int, color: Tuple[int, int, int]) -> np.ndarray:
    img = np.full((panel_h, panel_w, 3), color, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (panel_w - 20, panel_h - 20), (255, 255, 255), 2)
    cv2.putText(img, label, (35, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def load_reaction_images(panel_w: int, panel_h: int) -> Dict[str, np.ndarray]:
    asset_dir = os.path.join(os.path.dirname(__file__), "images")
    base_names = {
        "neutral": "neutral",
        "thumbs_up": "thumbs-up",
        "shocked": "shocked",
        "victory": "victory",
        "pointing_up": "pointing-up",
        "cinema": "two-open-palms",
        "stop": "one-open-palm",
        "pointing_self": "pointing-at-yourself",
        "idle": "idle",
        "yeah": "two-closed-fists-up",
        "middle_single": "middle-finger",
        "middle_double": "double-middle-finger",
        "heart": "two-hand-heart",
        "pause": "t-sign",
        "ok": "ok-sign",
        "angry": "one-closed-fist",
        "tongue": "tongue-out",
        "sleep": "sleep",
        "shh": "shush-sign",
    }

    images: Dict[str, np.ndarray] = {}
    idle_fallback = make_placeholder("IDLE", panel_w, panel_h, (50, 50, 70))
    idle_path = find_image_path(asset_dir, "idle")
    idle_img = cv2.imread(idle_path) if idle_path is not None else None
    if idle_img is not None:
        idle_fallback = fit_image_to_panel(idle_img, panel_w, panel_h)

    for key, base_name in base_names.items():
        path = find_image_path(asset_dir, base_name)
        img = cv2.imread(path) if path is not None else None
        if img is None:
            images[key] = idle_fallback.copy()
        else:
            images[key] = fit_image_to_panel(img, panel_w, panel_h)

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe face + gesture reaction board")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use (default: 0)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "any", "dshow", "msmf"],
        default="auto",
        help="OpenCV camera backend (default: auto)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and print available camera indices, then continue launching",
    )
    parser.add_argument(
        "--scan-max-index",
        type=int,
        default=8,
        help="Highest camera index to probe when using --list-cameras (default: 8)",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=420,
        help="Square size for each side panel in pixels (default: 420)",
    )
    parser.add_argument(
        "--hide-landmarks",
        action="store_true",
        help="Start with face/hand landmarks hidden",
    )
    return parser.parse_args()


def open_camera(index: int, backend: str) -> cv2.VideoCapture:
    if backend == "auto":
        # Try multiple backends because virtual/phone cameras can be backend-specific on Windows.
        for name in ["any", "dshow", "msmf"]:
            cap = cv2.VideoCapture(index, BACKEND_MAP[name])
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(index)

    return cv2.VideoCapture(index, BACKEND_MAP[backend])


def list_available_cameras(max_index: int, backend: str) -> None:
    print(f"Scanning camera indices 0..{max_index} with backend={backend}...")
    found = []

    for idx in range(max_index + 1):
        cap = open_camera(idx, backend)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                found.append(idx)
        cap.release()

    if found:
        print("Found camera indices:", ", ".join(str(x) for x in found))
    else:
        print("No working camera indices found in scan range.")


def ensure_task_models() -> Dict[str, str]:
    model_dir = os.path.join(os.path.dirname(__file__), "assets", "models")
    os.makedirs(model_dir, exist_ok=True)

    local_paths = {
        "face_landmarker": os.path.join(model_dir, "face_landmarker.task"),
        "gesture_recognizer": os.path.join(model_dir, "gesture_recognizer.task"),
    }

    for key, path in local_paths.items():
        if os.path.exists(path):
            continue
        print(f"Downloading model: {key}...")
        try:
            urllib.request.urlretrieve(MODEL_URLS[key], path)
        except Exception as ex:
            raise RuntimeError(
                f"Failed to download model '{key}'. Check internet access and retry. Error: {ex}"
            ) from ex

    return local_paths


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        list_available_cameras(args.scan_max_index, args.backend)

    cam = open_camera(args.camera_index, args.backend)
    if not cam.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {args.camera_index}. "
            "Try --list-cameras to find the correct index."
        )

    ret, sample = cam.read()
    if not ret:
        raise RuntimeError("Could not read from webcam")

    sample_h, sample_w = sample.shape[:2]
    panel_h = max(200, args.square_size)
    camera_panel_w = max(1, int(sample_w * (panel_h / max(1, sample_h))))
    output_panel_w = panel_h

    reactions = load_reaction_images(output_panel_w, panel_h)

    model_paths = ensure_task_models()

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_paths["face_landmarker"]),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    gesture_options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_paths["gesture_recognizer"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    current_label = "neutral"
    show_landmarks = not args.hide_landmarks
    previous_gray: np.ndarray | None = None
    idle_seconds = 0.0
    previous_tick = time.monotonic()
    eyes_closed_seconds = 0.0

    print(
        f"Using camera index={args.camera_index}, backend={args.backend}, panel_height={panel_h}. "
        "Press Q or ESC to quit."
    )

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (camera_panel_w, panel_h), interpolation=cv2.INTER_AREA)

        now = time.monotonic()
        dt = now - previous_tick
        previous_tick = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            motion_score = float(np.mean(cv2.absdiff(gray, previous_gray)))
            if motion_score < 2.0:
                idle_seconds += dt
            else:
                idle_seconds = 0.0
        previous_gray = gray
        is_idle = idle_seconds >= 1.5

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic_ns() / 1_000_000)

        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        gesture_result = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)

        detected_action = None
        shocked = False

        detected_action = choose_action(gesture_result, face_result)

        if gesture_result.hand_landmarks:
            for hand_landmarks in gesture_result.hand_landmarks:
                if show_landmarks:
                    draw_landmark_points(frame, hand_landmarks, color=(0, 255, 0), radius=3, every_n=1)

        shocked_face = False
        tongue_face = False
        eyes_closed_now = False
        both_hands_on_head = False

        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                if show_landmarks:
                    draw_landmark_points(frame, face_landmarks, color=(0, 180, 255), radius=1, every_n=2)
                if detect_shocked(face_landmarks, frame.shape[1], frame.shape[0]):
                    shocked_face = True
                if detect_tongue(face_landmarks, frame):
                    tongue_face = True
                if detect_eyes_closed(face_landmarks, frame.shape[1], frame.shape[0]):
                    eyes_closed_now = True

            if gesture_result.hand_landmarks:
                both_hands_on_head = hands_touching_head(
                    gesture_result.hand_landmarks,
                    face_result.face_landmarks[0],
                )

        shocked = shocked_face and both_hands_on_head

        if eyes_closed_now:
            eyes_closed_seconds += dt
        else:
            eyes_closed_seconds = 0.0
        sleeping = eyes_closed_seconds >= 0.5

        if shocked:
            current_label = "shocked"
        elif tongue_face:
            current_label = "tongue"
        elif detected_action is not None:
            current_label = detected_action
        elif sleeping:
            current_label = "sleep"
        elif is_idle:
            current_label = "idle"
        else:
            current_label = "neutral"

        right_panel = reactions[current_label]

        cv2.putText(
            frame,
            f"Detected: {current_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Landmarks: {'ON' if show_landmarks else 'OFF'} (H to toggle)",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
        )
        cv2.putText(frame, "CAMERA", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(right_panel, "OUTPUT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        combined = np.hstack((frame, right_panel))
        cv2.imshow("Face + Gesture Reaction Board", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("h"), ord("H")):
            show_landmarks = not show_landmarks
        if key == 27 or key in (ord("q"), ord("Q")):
            break

    cam.release()
    face_landmarker.close()
    gesture_recognizer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
