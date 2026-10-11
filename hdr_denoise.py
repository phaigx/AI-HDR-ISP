"""训练用于 HDR 图像去噪的残差 U-Net。

训练时在线模拟光子散粒噪声和读出噪声，网络不直接预测干净图像，
而是预测噪声残差，推理时通过 ``noisy - predicted_noise`` 得到结果。
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import imageio.v2 as imageio


# 数据源可以是 EXR 目录，也可以是包含 EXR 路径的文本文件。
TRAIN_LIST_TXT = "./train"
PATCH_SIZE = 256
BATCH_SIZE = 1
EPOCHS_DENOISE = 100
# 光子数越大，散粒噪声相对越弱；读出噪声与信号无关。
PHOTON_SCALE_MIN = 500.0
PHOTON_SCALE_MAX = 3000.0
READ_NOISE_STD_MIN = 0.001
READ_NOISE_STD_MAX = 0.01
CHARBONNIER_EPS = 1e-3
RECONSTRUCTION_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DENOISE_CKPT = "./checkpoint/denoise_best.pth"
TENSORBOARD_LOG_DIR = "./runs/denoise"


def read_exr_cv2(path: str) -> np.ndarray:
    """读取三通道 EXR，并统一转换为 RGB 的 float32 数组。"""
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#保持原始位宽和颜色类型
    if image is None:
        raise FileNotFoundError(f"Unable to read EXR image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel EXR image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

def read_exr(path):
    img = imageio.imread(path, format='EXR-FI')
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        # 灰度图扩展成 3 通道
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] >= 3:
        # 只取前 3 个通道，丢弃 alpha 或其他通道
        img = img[..., :3]
    return img

def write_exr(path: str, image: np.ndarray) -> None:
    """将 RGB float32 数组转换为 BGR 后写回 EXR 文件。"""
    bgr = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        raise IOError(f"Unable to write EXR image: {path}")


def add_poisson_gaussian_noise(clean_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """添加信号相关的散粒噪声和信号无关的读出噪声。

    返回 ``(noisy_image, noise)``，其中噪声标签严格定义为 noisy-clean，
    因此可以直接监督残差网络；输入信号先截断为非负值以满足泊松分布要求。
    """
    signal = np.maximum(clean_image, 0.0).astype(np.float32)#将图像中所有负值像素截断为 0，确保信号非负
    photon_scale = random.uniform(PHOTON_SCALE_MIN, PHOTON_SCALE_MAX)#随机采样光子数缩放因子，模拟不同曝光条件下的噪声强度。
    read_noise_std = random.uniform(READ_NOISE_STD_MIN, READ_NOISE_STD_MAX)#随机采样读出噪声标准差，模拟不同传感器的读出噪声水平

    expected_photons = signal * photon_scale#根据缩放因子将信号转换为期望的光子数，作为泊松分布的参数
    shot_sample = np.random.poisson(expected_photons).astype(np.float32)#根据期望光子数采样泊松分布，模拟散粒噪声的随机性
    shot_noise = shot_sample / photon_scale - signal#将采样得到的光子数转换回原始信号范围，并计算散粒噪声，即采样值减去原始信号
    read_noise = np.random.normal(0.0, read_noise_std, signal.shape).astype(np.float32)#根据采样的读出噪声标准差生成高斯噪声，模拟信号无关的读出噪声

    noisy_image = np.maximum(signal + shot_noise + read_noise, 0.0)#黑电平之后的数据一般会裁剪成非负值，因此将最终的噪声图像裁剪为非负范围。
    return noisy_image.astype(np.float32), (noisy_image - signal).astype(np.float32)


def denoise_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    """计算残差损失，并用图像域重建项约束最终去噪结果。
    Charbonnier损失是平滑的 L1 近似，对少量较大的异常噪声更稳健；
    重建项权重较小，用于避免网络只拟合噪声而忽略整体图像质量。
    """
    residual = predicted_noise - target_noise
    charbonnier = torch.sqrt(residual.square() + CHARBONNIER_EPS**2).mean()#计算 Charbonnier 损失，作为主要的残差约束
    denoised = noisy - predicted_noise
    reconstruction = F.l1_loss(denoised, clean)
    return charbonnier + RECONSTRUCTION_LOSS_WEIGHT * reconstruction


def hdr_for_tensorboard(hdr: torch.Tensor) -> torch.Tensor:
    """将线性 HDR 压缩到可视化范围，不改变训练输入。"""
    display = torch.log1p(hdr.detach().clamp_min(0.0))
    scale = torch.quantile(display.flatten(1), 0.99, dim=1).clamp_min(1e-6)
    return (display / scale[:, None, None, None]).clamp(0.0, 1.0)


class DoubleConv(nn.Module):
    """由两层卷积组成的基础特征提取模块。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DenoiseUNetRes(nn.Module):
    """预测噪声分量的残差去噪 U-Net。"""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
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
        # 跳跃连接保留局部纹理，输出与输入同尺寸的噪声图。
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bn = self.bottleneck(self.pool2(d2))
        c2 = self.conv2(torch.cat([self.up2(bn), d2], dim=1))
        c1 = self.conv1(torch.cat([self.up1(c2), d1], dim=1))
        return self.out_conv(c1)


def _random_patch(image: np.ndarray) -> np.ndarray:
    """随机裁剪固定大小图像块；过小图像先用镜像边界补齐。"""
    height, width = image.shape[:2]
    if height < PATCH_SIZE or width < PATCH_SIZE:
        image = cv2.copyMakeBorder(
            image, 0, max(0, PATCH_SIZE - height), 0, max(0, PATCH_SIZE - width),
            cv2.BORDER_REFLECT_101,
        )#镜像边界补齐图像，使其至少为 PATCH_SIZE 大小
        height, width = image.shape[:2]
    y0 = random.randint(0, height - PATCH_SIZE)
    x0 = random.randint(0, width - PATCH_SIZE)
    return image[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]


class DenoisePatchDataset(torch.utils.data.Dataset):
    """读取 HDR 图像并在线生成 noisy、clean、noise 三元组。"""
    def __init__(self, data_source: str, patches_per_image: int = 16):
        source = Path(data_source)
        self.patches_per_image = patches_per_image
        if source.is_dir():
            self.paths = [str(path) for path in sorted(source.rglob("*.exr"))]#递归搜索目录下所有 EXR 文件，并按路径排序
        elif source.is_file():
            with source.open("r", encoding="utf-8") as file:
                self.paths = [line.strip() for line in file if line.strip()]#读取文本文件中的每一行，去除空行和首尾空白字符，得到 EXR 文件路径列表
        else:
            raise FileNotFoundError(f"Training data source not found: {data_source}")
        if not self.paths:
            raise ValueError(f"No EXR images found in {data_source}")

    def __len__(self) -> int:
        return len(self.paths)*self.patches_per_image

    def __getitem__(self, idx: int):
        # 噪声动态采样，避免提前生成和存储大量训练样本。
        path_idx = idx % len(self.paths)
        clean_patch = _random_patch(read_exr(self.paths[path_idx]))
        noisy_patch, noise = add_poisson_gaussian_noise(clean_patch)
        to_tensor = lambda image: torch.from_numpy(image).permute(2, 0, 1).float()#将 NumPy 图像从 (H, W, C) 转换为 PyTorch 张量 (C, H, W) 并转换为 float 类型
        return to_tensor(noisy_patch), to_tensor(clean_patch), to_tensor(noise)


def train_denoise() -> None:
    """执行残差去噪训练并保存模型权重。"""
    os.makedirs("./checkpoint", exist_ok=True)
    denoise_net = DenoiseUNetRes().to(DEVICE)
    optimizer = torch.optim.Adam(denoise_net.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    dataset = DenoisePatchDataset(TRAIN_LIST_TXT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    global_step = 0
    for epoch in range(EPOCHS_DENOISE):
        denoise_net.train()
        total_loss = 0.0
        best_loss = float("inf")
        for noisy, clean, gt_noise in tqdm(loader):#含噪声图像、干净图像和噪声标签的批次
            # 模型只预测噪声；干净图像通过损失函数参与监督，不直接作为输入。
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            gt_noise = gt_noise.to(DEVICE)
            predicted_noise = denoise_net(noisy)
            loss = denoise_loss(predicted_noise, gt_noise, noisy, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1
        average_loss = total_loss / len(loader)
        writer.add_scalar("train/loss_epoch", average_loss, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_images("images/noisy_log", hdr_for_tensorboard(noisy[:1]), epoch)
        writer.add_images("images/clean_log", hdr_for_tensorboard(clean[:1]), epoch)
        writer.add_images("images/denoised_log", hdr_for_tensorboard(noisy[:1] - predicted_noise[:1]), epoch)
        print(f"[Denoise-Res] epoch {epoch:03d}, noise_mse_loss {average_loss:.6f}")
        '''
        if epoch % 20 == 0:#
            torch.save(denoise_net.state_dict(), f"./checkpoint/denoise_epoch{epoch:03d}.pth")
            print(f"Denoise net checkpoint saved to ./checkpoint/denoise_epoch{epoch:03d}.pth")
        '''
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(denoise_net.state_dict(), DENOISE_CKPT)
            print(f"Denoise net checkpoint saved to {DENOISE_CKPT}")
    writer.close()
    print(f"Denoise net saved to {DENOISE_CKPT},best_loss {best_loss:.6f}")


if __name__ == "__main__":
    train_denoise()
