"""训练带全局曝光条件的四通道 HDR 色调映射网络。

网络输入为 ``[log1p(HDR_R), log1p(HDR_G), log1p(HDR_B), key]``：

* RGB 保留原始 HDR 的动态范围，不提前乘以 key，避免削弱高光层次；
* ``key`` 是根据整张 HDR 图计算的 Reinhard 全局曝光条件；
* 监督目标是同一张原始 HDR 图生成的双边局部 tone mapping 结果。

该脚本使用独立的 TensorBoard 日志和 checkpoint，避免覆盖旧的三通道模型。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ai_tonemap import (
    DEVICE,
    PATCH_SIZE,
    DoubleConv,
    delta_ab_loss,
    gradient_loss,
    images_for_tensorboard,
    reinhard_local_tonemap,
    bilateral_tonemap_target,
    read_exr,
    reinhard_exposure_key,
    require_finite,
)


TRAIN_LIST_TXT = "./train"
BATCH_SIZE = 1
PATCHES_PER_IMAGE = 16
EPOCHS = 61
LEARNING_RATE = 1e-4

LAMBDA_L1 = 0.1
LAMBDA_GRAD = 0.7
LAMBDA_DAB = 0.2

CHECKPOINT_DIR = "./checkpoint"
TM4_CKPT = f"{CHECKPOINT_DIR}/tm_conditioned_best.pth"
TENSORBOARD_LOG_DIR = "./runs/tonemap_conditioned"

BILATERAL_SIGMA_SPACE = 16.0
BILATERAL_SIGMA_COLOR = 0.35
BILATERAL_DETAIL_STRENGTH = 0.70


class ConditionedToneMapUNet(nn.Module):
    """接收三通道 HDR 和一个全局曝光 key 的色调映射 U-Net。

    RGB 通道在网络内部执行 ``log1p``，以压缩 HDR 动态范围；第四个
    通道是常数曝光条件，不执行 ``log1p``，从而避免改变 key 的含义。
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.down1 = DoubleConv(4, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将四通道条件输入转换为 ``[0, 1]`` 范围的 SDR RGB 输出。"""
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected input shape (N, 4, H, W), got {tuple(x.shape)}")
        hdr = torch.log1p(x[:, :3].clamp_min(0.0))
        key = x[:, 3:4]
        features = torch.cat([hdr, key], dim=1)
        d1 = self.down1(features)
        d2 = self.down2(self.pool1(d1))
        bottleneck = self.bottleneck(self.pool2(d2))
        c2 = self.conv2(torch.cat([self.up2(bottleneck), d2], dim=1))
        c1 = self.conv1(torch.cat([self.up1(c2), d1], dim=1))
        return torch.sigmoid(self.out_conv(c1))


def make_conditioned_input(hdr: np.ndarray, key: float) -> np.ndarray:
    """拼接原始 HDR RGB 和归一化的全局曝光条件通道。

    key 可能跨越多个数量级，直接输入会让网络同时处理极大的数值范围。
    这里使用 ``log1p(key)`` 作为条件值，并在训练、评估和部署中保持一致。
    """
    key_value = np.log1p(max(float(key), 0.0))
    key_map = np.full(hdr.shape[:2] + (1,), key_value, dtype=np.float32)
    return np.concatenate([hdr.astype(np.float32), key_map], axis=-1)


class ConditionedTMPatchDataset(torch.utils.data.Dataset):
    """随机生成四通道 HDR patch 及其双边局部监督目标。"""

    def __init__(self, data_source: str, patches_per_image: int = PATCHES_PER_IMAGE):
        self.patches_per_image = patches_per_image
        source = Path(data_source)
        if source.is_dir():
            self.paths = [str(path) for path in sorted(source.rglob("*.exr"))]
        elif source.is_file():
            with source.open("r", encoding="utf-8") as file:
                self.paths = [line.strip() for line in file if line.strip()]
        else:
            raise FileNotFoundError(f"Training data source not found: {data_source}")
        if not self.paths:
            raise ValueError(f"No EXR images found in {data_source}")

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取一张 HDR 图，按整图 key 构造条件输入并随机裁剪 patch。"""
        path = self.paths[idx % len(self.paths)]
        hdr = read_exr(path)
        '''
        reference = bilateral_tonemap_target(
            hdr,
            sigma_space=BILATERAL_SIGMA_SPACE,
            sigma_color=BILATERAL_SIGMA_COLOR,
            detail_strength=BILATERAL_DETAIL_STRENGTH,
        )
        '''
        reference= reinhard_local_tonemap(hdr)
        key = reinhard_exposure_key(hdr)
        conditioned = make_conditioned_input(hdr, key)
        if not np.isfinite(conditioned).all() or not np.isfinite(reference).all():
            raise ValueError(f"Training sample contains NaN or Inf: {path}")

        height, width = hdr.shape[:2]
        if height < PATCH_SIZE or width < PATCH_SIZE:
            raise ValueError(f"Image must be at least {PATCH_SIZE}x{PATCH_SIZE}, got {width}x{height}")
        y0 = random.randint(0, height - PATCH_SIZE)
        x0 = random.randint(0, width - PATCH_SIZE)
        crop = (slice(y0, y0 + PATCH_SIZE), slice(x0, x0 + PATCH_SIZE))

        conditioned_tensor = torch.from_numpy(conditioned[crop]).permute(2, 0, 1).float()
        reference_tensor = torch.from_numpy(reference[crop]).permute(2, 0, 1).float()
        return conditioned_tensor, reference_tensor


def train_conditioned_tonemap() -> None:
    """训练四通道条件网络并保存最终和最佳 checkpoint。"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    model = ConditionedToneMapUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    dataset = ConditionedTMPatchDataset(TRAIN_LIST_TXT)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    best_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for conditioned, reference in tqdm(loader, desc=f"epoch {epoch:03d}"):
            conditioned = conditioned.to(DEVICE)
            reference = reference.to(DEVICE)
            require_finite(conditioned, "conditioned_input")
            require_finite(reference, "reference")

            prediction = model(conditioned)
            l1_loss = F.l1_loss(prediction, reference)
            grad_loss = gradient_loss(prediction, reference)
            dab_loss = delta_ab_loss(prediction, reference) if LAMBDA_DAB else prediction.new_zeros(())
            loss = LAMBDA_L1 * l1_loss + LAMBDA_GRAD * grad_loss + LAMBDA_DAB * dab_loss
            require_finite(loss, "loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/loss_l1_step", l1_loss.item(), global_step)
            writer.add_scalar("train/loss_gradient_step", grad_loss.item(), global_step)
            writer.add_scalar("train/loss_delta_ab_step", dab_loss.item(), global_step)
            global_step += 1

        average_loss = total_loss / len(loader)
        writer.add_scalar("train/loss_epoch", average_loss, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_images("images/reference", images_for_tensorboard(reference[:1]), epoch)
        writer.add_images("images/prediction", images_for_tensorboard(prediction[:1]), epoch)
        print(
            f"[ToneMap-Conditioned] epoch {epoch:03d}, "
            f"loss {average_loss:.6f}, l1 {l1_loss.item():.6f}, "
            f"gradient {grad_loss.item():.6f}"
        )

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), TM4_CKPT)
            print(f"Conditioned checkpoint saved to {TM4_CKPT}")

    writer.close()
    print(f"Conditioned training finished, best loss: {best_loss:.6f}")


if __name__ == "__main__":
    train_conditioned_tonemap()
