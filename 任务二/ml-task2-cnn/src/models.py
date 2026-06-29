"""可配置 CNN：支援按參數切換卷積核、通道、層數、Dropout、BatchNorm。"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

PRESET_MODEL_CONFIGS: dict[str, dict] = {
    # 模型1：基礎 CNN
    "base_cnn": {
        "channels": [32, 64, 128],
        "stage_depths": [1, 1, 1],
        "kernel_size": 3,
        "use_batchnorm": False,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
    # 模型2：加寬 CNN（深度不變，只增加通道）
    "wide_cnn": {
        "channels": [64, 128, 256],
        "stage_depths": [1, 1, 1],
        "kernel_size": 3,
        "use_batchnorm": False,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
    # 模型3：加深 CNN（通道接近 base，增加每個 stage 的卷積層）
    "deep_cnn": {
        "channels": [32, 64, 128],
        "stage_depths": [2, 2, 2],
        "kernel_size": 3,
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
}

MODEL_ALIASES: dict[str, str] = {
    # 兼容舊名稱，避免既有指令與 checkpoint 失效
    "simple_cnn": "base_cnn",
    "vgg_small": "wide_cnn",
    "resnet_small": "deep_cnn",
}


class ConfigurableCNN(nn.Module):
    """可配置 CNN 主體。"""

    def __init__(
        self,
        num_classes: int = 10,
        channels: list[int] | tuple[int, ...] = (32, 64, 128),
        stage_depths: list[int] | tuple[int, ...] = (1, 1, 1),
        kernel_size: int = 3,
        use_batchnorm: bool = False,
        use_dropout: bool = True,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        if len(channels) != len(stage_depths):
            raise ValueError("channels 與 stage_depths 長度必須一致")
        if any(c <= 0 for c in channels):
            raise ValueError("channels 需為正整數")
        if any(d <= 0 for d in stage_depths):
            raise ValueError("stage_depths 需為正整數")
        if kernel_size <= 0:
            raise ValueError("kernel_size 需大於 0")
        if not 0 <= dropout_p < 1:
            raise ValueError("dropout_p 需在 [0, 1) 範圍內")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels, depth in zip(channels, stage_depths):
            for _ in range(depth):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_p))
        classifier_layers.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        return self.classifier(x)


def normalize_model_name(name: str) -> str:
    """將舊名稱映射到新名稱。"""
    return MODEL_ALIASES.get(name, name)


def get_model_config(name: str, config_overrides: dict | None = None) -> dict:
    """取得模型有效配置（預設 + 覆蓋）。"""
    normalized = normalize_model_name(name)
    if normalized not in PRESET_MODEL_CONFIGS:
        available = list(PRESET_MODEL_CONFIGS.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(f"未知模型: {name}，可選: {available}")
    config = deepcopy(PRESET_MODEL_CONFIGS[normalized])
    if config_overrides:
        for key, value in config_overrides.items():
            if value is not None:
                config[key] = value
    return config


def build_model_from_config(config: dict, num_classes: int = 10) -> nn.Module:
    """由完整配置建立模型。"""
    return ConfigurableCNN(
        num_classes=num_classes,
        channels=config["channels"],
        stage_depths=config["stage_depths"],
        kernel_size=config["kernel_size"],
        use_batchnorm=config["use_batchnorm"],
        use_dropout=config["use_dropout"],
        dropout_p=config["dropout_p"],
    )


def build_model(name: str, num_classes: int = 10, config_overrides: dict | None = None) -> nn.Module:
    """根據名稱與可選覆蓋參數建立模型。"""
    config = get_model_config(name, config_overrides=config_overrides)
    return build_model_from_config(config, num_classes=num_classes)
