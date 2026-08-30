"""Фабрика моделей сегментации."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SegFormerWrapper(nn.Module):
    """HuggingFace SegFormer → логиты [B,C,H,W] в исходном разрешении."""

    def __init__(self, pretrained_name: str, num_classes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Параметры официальных MiT-конфигураций. Это позволяет создать
            # архитектуру для полного checkpoint без обращения к Hugging Face.
            variant = pretrained_name.rsplit("/", 1)[-1].lower().replace("_", "-")
            depths = {
                "mit-b0": [2, 2, 2, 2],
                "mit-b1": [2, 2, 2, 2],
                "mit-b2": [3, 4, 6, 3],
                "mit-b3": [3, 4, 18, 3],
                "mit-b4": [3, 8, 27, 3],
                "mit-b5": [3, 6, 40, 3],
            }.get(variant)
            if depths is None:
                raise ValueError(f"Unknown offline SegFormer variant: {variant}")
            small = variant == "mit-b0"
            config = SegformerConfig(
                num_labels=num_classes,
                depths=depths,
                hidden_sizes=[32, 64, 160, 256] if small else [64, 128, 320, 512],
                decoder_hidden_size=256 if small else 768,
                id2label={i: str(i) for i in range(num_classes)},
                label2id={str(i): i for i in range(num_classes)},
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B,C,h/4,w/4]
        return nn.functional.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def build_model(cfg: dict[str, Any], *, pretrained: bool = True) -> nn.Module:
    mcfg = cfg["model"]
    arch = mcfg["architecture"].lower()
    num_classes = int(mcfg["classes"])
    in_ch = int(mcfg.get("in_channels", 3))
    encoder = mcfg.get("encoder_name", "efficientnet-b4")
    weights = mcfg.get("encoder_weights", "imagenet")

    if arch in {"unet", "unetplusplus", "deeplabv3plus", "fpn", "upernet"}:
        import segmentation_models_pytorch as smp

        builders = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
        }
        if arch == "upernet":
            # SMP не всегда имеет UPerNet — fallback на UnetPlusPlus
            cls = getattr(smp, "UPerNet", None) or smp.UnetPlusPlus
        else:
            cls = builders[arch]
        return cls(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_ch,
            classes=num_classes,
        )

    if arch == "segformer":
        name = mcfg.get("pretrained_name") or f"nvidia/{encoder.replace('_', '-')}"
        return SegFormerWrapper(name, num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown architecture: {arch}")


def split_param_groups(
    model: nn.Module,
    lr: float,
    encoder_lr_factor: float = 0.1,
    weight_decay: float = 1e-4,
) -> list[dict[str, Any]]:
    """Меньший LR для энкодера / backbone."""
    encoder_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            name.startswith("encoder")
            or name.startswith("backbone")
            or name.startswith("model.segformer")
            or ".encoder." in name
            or ".backbone." in name
        ):
            encoder_params.append(p)
        else:
            other_params.append(p)

    # Если разделение не сработало — один group
    if not encoder_params or not other_params:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

    return [
        {"params": encoder_params, "lr": lr * encoder_lr_factor, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]
