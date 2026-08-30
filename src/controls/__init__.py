"""Детекция контрольных пунктов (розовые цифры / кружки) и старта."""

from src.controls.detect import detect_controls, detect_controls_cv_trace
from src.controls.detect_vlm import detect_controls_vlm, detect_controls_vlm_trace
from src.controls.types import (
    ControlPoint,
    CourseDetection,
    CvDetectTrace,
    StartFinish,
    VlmDetectTrace,
)
from src.controls.visualize import (
    animate_detect_controls_cv,
    animate_detect_controls_vlm,
    animate_detect_controls_vlm_refine,
    draw_controls,
)
from src.controls.vlm_yandex import list_yandex_chat_models

__all__ = [
    "ControlPoint",
    "CourseDetection",
    "CvDetectTrace",
    "StartFinish",
    "VlmDetectTrace",
    "animate_detect_controls_cv",
    "animate_detect_controls_vlm",
    "animate_detect_controls_vlm_refine",
    "detect_controls",
    "detect_controls_cv_trace",
    "detect_controls_vlm",
    "detect_controls_vlm_trace",
    "draw_controls",
    "list_yandex_chat_models",
]
