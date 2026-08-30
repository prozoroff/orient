"""Albumentations-пайплайны: геометрия синхронно для image+mask."""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(cfg: dict[str, Any], image_size: int = 512) -> A.Compose:
    augs: list[Any] = []
    if cfg.get("horizontal_flip", 0):
        augs.append(A.HorizontalFlip(p=float(cfg["horizontal_flip"])))
    if cfg.get("vertical_flip", 0):
        augs.append(A.VerticalFlip(p=float(cfg["vertical_flip"])))
    if cfg.get("random_rotate90", 0):
        augs.append(A.RandomRotate90(p=float(cfg["random_rotate90"])))

    ssr = cfg.get("shift_scale_rotate") or {}
    if ssr:
        augs.append(
            A.Affine(
                translate_percent={
                    "x": (-float(ssr.get("shift_limit", 0.06)), float(ssr.get("shift_limit", 0.06))),
                    "y": (-float(ssr.get("shift_limit", 0.06)), float(ssr.get("shift_limit", 0.06))),
                },
                scale=(
                    1.0 - float(ssr.get("scale_limit", 0.1)),
                    1.0 + float(ssr.get("scale_limit", 0.1)),
                ),
                rotate=(-float(ssr.get("rotate_limit", 15)), float(ssr.get("rotate_limit", 15))),
                cval=255,
                cval_mask=0,
                p=float(ssr.get("p", 0.4)),
            )
        )

    bc = cfg.get("brightness_contrast") or {}
    if bc:
        augs.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(bc.get("brightness_limit", 0.1)),
                contrast_limit=float(bc.get("contrast_limit", 0.1)),
                p=float(bc.get("p", 0.3)),
            )
        )

    hsv = cfg.get("hue_saturation") or {}
    if hsv:
        augs.append(
            A.HueSaturationValue(
                hue_shift_limit=int(hsv.get("hue_shift_limit", 5)),
                sat_shift_limit=int(hsv.get("sat_shift_limit", 10)),
                val_shift_limit=int(hsv.get("val_shift_limit", 10)),
                p=float(hsv.get("p", 0.2)),
            )
        )

    # Размытие / JPEG — сближает чистый OCAD-рендер с фото карты на телефоне.
    blur = cfg.get("blur") or {}
    if blur:
        blur_p = float(blur.get("p", 0.35))
        gauss_limit = blur.get("gaussian_limit", [3, 7])
        motion_limit = int(blur.get("motion_limit", 7))
        augs.append(
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=tuple(int(x) for x in gauss_limit),
                        p=1.0,
                    ),
                    A.MotionBlur(blur_limit=motion_limit, p=1.0),
                ],
                p=blur_p,
            )
        )

    jpeg = cfg.get("jpeg_compression") or {}
    if jpeg:
        quality = jpeg.get("quality_range", [55, 95])
        augs.append(
            A.ImageCompression(
                quality_range=(int(quality[0]), int(quality[1])),
                compression_type="jpeg",
                p=float(jpeg.get("p", 0.3)),
            )
        )

    augs.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(augs)


def build_eval_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
