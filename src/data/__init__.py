from src.data.labels import channels_to_index, colorize_label, overlay_prediction


def __getattr__(name: str):
    """Не загружать training-зависимости во время production-инференса."""
    if name in {"OrienteeringSegDataset", "create_dataloaders", "load_index"}:
        from src.data.dataset import OrienteeringSegDataset, create_dataloaders, load_index

        return {
            "OrienteeringSegDataset": OrienteeringSegDataset,
            "create_dataloaders": create_dataloaders,
            "load_index": load_index,
        }[name]
    raise AttributeError(name)

__all__ = [
    "OrienteeringSegDataset",
    "create_dataloaders",
    "load_index",
    "channels_to_index",
    "colorize_label",
    "overlay_prediction",
]
