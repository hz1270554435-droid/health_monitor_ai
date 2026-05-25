from __future__ import annotations


def build_small_cnn(num_classes: int = 2):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for model training") from exc

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )


def build_ds_cnn(num_classes: int = 2):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for model training") from exc

    def depthwise_block(in_channels: int, out_channels: int, stride: int = 1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        depthwise_block(16, 24, stride=2),
        depthwise_block(24, 32, stride=2),
        depthwise_block(32, 48, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(48, num_classes),
    )


MODEL_REGISTRY = {
    "small_cnn": build_small_cnn,
    "ds_cnn": build_ds_cnn,
}


def build_model(name: str, num_classes: int = 2):
    model_name = str(name).strip().lower()
    try:
        builder = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported audio model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from exc
    return builder(num_classes=num_classes)
