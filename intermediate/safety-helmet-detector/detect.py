"""Real-Time Safety Helmet & Gear Detection.

AIML Club - Oriental College of Technology, Bhopal.

Extends the starter pipeline with a live webcam / video-file inference loop built
on Ultralytics YOLO. Every frame is annotated through Ultralytics' own renderer
and then receives a HUD overlay reporting the rolling frame rate, the number of
helmet and no-helmet detections in the current frame, and the compliance status
those detections imply.

Examples:
    python detect.py
    python detect.py --model helmet-yolov8n.pt --conf 0.5
    python detect.py --source campus-footage.mp4
"""

import argparse
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

# A BGR image exactly as OpenCV produces it.
Frame = NDArray[np.uint8]

# Number of recent frame durations averaged for a stable FPS readout.
FPS_WINDOW = 30

# Consecutive unreadable frames tolerated before the stream is declared finished.
# A single dropped webcam frame should not tear down a long-running session.
MAX_CONSECUTIVE_READ_FAILURES = 10

# Words that turn an otherwise positive helmet class into a violation.
NEGATION_TOKENS = frozenset({"no", "not", "without", "missing", "unprotected"})

# BGR colour tuples.
GREEN = (0, 200, 0)
RED = (0, 0, 255)
GREY = (200, 200, 200)
WHITE = (255, 255, 255)

# Geometry of the translucent HUD banner.
BANNER_WIDTH_RATIO = 0.34
BANNER_LINE_HEIGHT = 28
BANNER_PADDING = 12
BANNER_ALPHA = 0.45

WINDOW_NAME = "Safety Helmet Detection"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the detection pipeline.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]`` when omitted,
            which keeps the function directly testable.

    Returns:
        The populated argument namespace.

    Raises:
        SystemExit: If the user passes an invalid argument.
    """
    parser = argparse.ArgumentParser(
        description="Real-time safety helmet detection with Ultralytics YOLO.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO weights (default: %(default)s). Supply custom weights "
        "trained on helmet classes to get real helmet/no-helmet detection.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for the default webcam, or a path to a video "
        "file (default: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for a detection to be kept "
        "(default: %(default)s).",
    )
    return parser.parse_args(argv)


def classify_label(label: str) -> str:
    """Normalise a model class name into a compliance category.

    Datasets name the same concept in many ways ("helmet", "Helmet", "no-helmet",
    "No_Helmet", "without helmet"), so the raw class name is lower-cased and
    tokenised before being matched instead of being compared literally.

    Args:
        label: Raw class name reported by the model.

    Returns:
        "helmet" for compliant headgear, "no_helmet" for a violation, and
        "other" when the label does not describe a helmet at all.
    """
    normalized = " ".join(label.lower().replace("-", " ").replace("_", " ").split())

    # "Helmetless" is a violation even though it contains no negation token.
    if "helmetless" in normalized:
        return "no_helmet"

    if "helmet" not in normalized:
        return "other"

    if NEGATION_TOKENS.intersection(normalized.split()):
        return "no_helmet"

    return "helmet"

def resolve_class_name(
    class_names: Union[Dict[int, str], Sequence[str]],
    class_id: int,
) -> str:
    """Look up a class name from a model's ``names`` mapping.

    Ultralytics exposes ``names`` as a dict for most models and as a plain
    sequence for others, so both shapes are supported.

    Args:
        class_names: The model's ``names`` attribute.
        class_id: Integer class id taken from a detection box.

    Returns:
        The class name, or an empty string when it cannot be resolved.
    """
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, ""))
    try:
        return str(class_names[class_id])
    except (IndexError, KeyError, TypeError):
        return ""


def count_detections(
    boxes: Sequence,
    class_names: Union[Dict[int, str], Sequence[str]],
) -> Tuple[int, int]:
    """Count the helmet and no-helmet detections in a single frame.

    Args:
        boxes: The ``boxes`` attribute of an Ultralytics result.
        class_names: The model's ``names`` attribute, used to label each box.

    Returns:
        A ``(helmet_count, no_helmet_count)`` tuple. Boxes whose class is
        unrelated to helmets are ignored.
    """
    helmet_count = 0
    no_helmet_count = 0

    for box in boxes:
        label = resolve_class_name(class_names, int(box.cls[0]))
        category = classify_label(label)
        if category == "helmet":
            helmet_count += 1
        elif category == "no_helmet":
            no_helmet_count += 1

    return helmet_count, no_helmet_count


def resolve_source(source: str) -> Union[int, str]:
    """Convert a user-supplied source into a value ``cv2.VideoCapture`` accepts.

    Args:
        source: Either a camera index such as ``"0"`` or a video file path.

    Returns:
        The integer camera index, or the path unchanged.
    """
    if source.isdigit():
        return int(source)
    return source


def compliance_status(
    helmet_count: int,
    no_helmet_count: int,
) -> Tuple[str, Tuple[int, int, int]]:
    """Derive the compliance verdict for a frame.

    Args:
        helmet_count: Number of helmet detections in the frame.
        no_helmet_count: Number of no-helmet detections in the frame.

    Returns:
        The status text and the BGR colour it should be rendered in.
    """
    if no_helmet_count > 0:
        return "VIOLATION", RED
    if helmet_count > 0:
        return "COMPLIANT", GREEN
    return "NO DETECTIONS", GREY


def draw_overlay(
    frame: Frame,
    fps: float,
    helmet_count: int,
    no_helmet_count: int,
) -> Frame:
    """Draw the FPS readout and compliance counters onto a frame.

    A translucent banner is laid down first so the HUD stays legible over both
    bright and dark video content.

    Args:
        frame: The annotated frame to draw on. Modified in place.
        fps: Rolling average frames per second.
        helmet_count: Number of helmet detections in this frame.
        no_helmet_count: Number of no-helmet detections in this frame.

    Returns:
        The same frame, with the overlay applied.
    """
    status_text, status_color = compliance_status(helmet_count, no_helmet_count)
    hud_lines = [
        (f"FPS: {fps:.1f}", WHITE),
        (f"Helmets: {helmet_count}", GREEN),
        (f"No Helmet: {no_helmet_count}", RED),
        (status_text, status_color),
    ]

    frame_width = frame.shape[1]
    banner_width = int(frame_width * BANNER_WIDTH_RATIO)
    banner_height = BANNER_LINE_HEIGHT * len(hud_lines) + BANNER_PADDING

    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (banner_width, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(banner, BANNER_ALPHA, frame, 1.0 - BANNER_ALPHA, 0, frame)

    for index, (text, color) in enumerate(hud_lines):
        baseline_y = BANNER_PADDING // 2 + BANNER_LINE_HEIGHT * index + 20
        cv2.putText(
            frame,
            text,
            (10, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def run(source: str, model_path: str, confidence: float) -> int:
    """Run the live detection loop against a webcam or video file.

    Args:
        source: Camera index such as ``"0"``, or a path to a video file.
        model_path: Path to the YOLO weights to load.
        confidence: Minimum confidence for a detection to be kept.

    Returns:
        0 when the loop finishes normally, 1 when the source cannot be opened.
    """
    model = YOLO(model_path)
    capture = cv2.VideoCapture(resolve_source(source))

    if not capture.isOpened():
        sys.stderr.write(f"Error: could not open video source '{source}'.\n")
        return 1

    frame_durations: Deque[float] = deque(maxlen=FPS_WINDOW)
    previous_time = time.perf_counter()
    failed_reads = 0

    print("Running inference. Press 'q' to quit.")

    try:
        while True:
            frame_read, frame = capture.read()
            if not frame_read or frame is None:
                failed_reads += 1
                if failed_reads >= MAX_CONSECUTIVE_READ_FAILURES:
                    sys.stderr.write("Video stream ended or frame could not be read.\n")
                    break
                continue

            failed_reads = 0

            results = model(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()
            helmet_count, no_helmet_count = count_detections(results[0].boxes, model.names)

            current_time = time.perf_counter()
            frame_durations.append(1.0 / max(current_time - previous_time, 1e-6))
            previous_time = current_time
            average_fps = sum(frame_durations) / len(frame_durations)

            draw_overlay(annotated, average_fps, helmet_count, no_helmet_count)
            cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user.\n")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: print the run configuration and start the detection loop.

    Args:
        argv: Optional argument vector forwarded to :func:`parse_args`.

    Returns:
        The process exit code.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("⛑️  Safety Helmet Detection — Real-Time Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print()

    return run(args.source, args.model, args.conf)


if __name__ == "__main__":
    sys.exit(main())