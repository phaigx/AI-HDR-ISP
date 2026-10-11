"""训练色调映射 U-Net。

使用干净 HDR 图像生成 Reinhard 局部色调映射结果作为监督目标，
让网络学习从 HDR RGB 数据直接预测 [0, 1] 范围内的 SDR RGB 图像。
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
from tqdm import tqdm


# 数据源可以是包含 EXR 文件的目录，也可以是每行一个路径的文本文件。
TRAIN_LIST_TXT = "./train"
PATCH_SIZE = 256
BATCH_SIZE = 1
EPOCHS_TM = 61
# 三项损失分别约束像素值、边缘结构和 Lab 空间中的色度差异。
LAMBDA_L1 = 0.1
LAMBDA_GRAD = 0.7
LAMBDA_DAB = 0.2
# 线性 sRGB/Rec.709 RGB 到相对亮度的转换系数。
# 0.299/0.587/0.114 是面向 Rec.601 非线性 RGB 的传统近似，不适合直接用于线性 HDR。
LINEAR_RGB_LUMA = (0.2126, 0.7152, 0.0722)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TM_CKPT = "./checkpoint/tm_best.pth"
TENSORBOARD_LOG_DIR = "./runs/tonemap"


def read_exr_cv2(path: str) -> np.ndarray:
    """读取 EXR 并将 OpenCV 的 BGR 通道顺序转换为 RGB。"""
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Unable to read EXR image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel EXR image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
def read_exr(path):
    import imageio.v2 as imageio

    img = imageio.imread(path, format='EXR-FI')
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        # 灰度图扩展成 3 通道
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] >= 3:
        # 只取前 3 个通道，丢弃 alpha 或其他通道
        img = img[..., :3]
    else:
        raise ValueError(f"Expected a 3-channel EXR image: {path}")
    if not np.isfinite(img).all():
        raise ValueError(f"EXR image contains NaN or Inf values: {path}")
    return img


def log_average_luminance(hdr_rgb: np.ndarray) -> float:
    """计算 Reinhard 算法使用的对数平均亮度。"""
    luminance = (
        LINEAR_RGB_LUMA[0] * hdr_rgb[..., 0]
        + LINEAR_RGB_LUMA[1] * hdr_rgb[..., 1]
        + LINEAR_RGB_LUMA[2] * hdr_rgb[..., 2]
    )
    return float(np.exp(np.mean(np.log(np.maximum(luminance, 0.0) + 1e-8))))


def reinhard_exposure_key(hdr_rgb: np.ndarray) -> float:
    """计算 Reinhard 映射使用的全局曝光 key。"""
    hdr_rgb = np.maximum(np.asarray(hdr_rgb, dtype=np.float32), 0.0)
    return 0.18 / (log_average_luminance(hdr_rgb) + 1e-8)


def reinhard_local_tonemap(hdr_rgb: np.ndarray) -> np.ndarray:
    """生成稳定的 Reinhard 参考图，并保持高亮区域的颜色和细节。

    先将 HDR RGB 的负值截断为 0，再用对数平均亮度计算曝光 key。之后
    只对亮度应用经典的 ``L / (1 + L)`` 映射，最后使用原始 RGB 与
    亮度的比例恢复颜色。这样高亮像素不会因为某个颜色通道较小，或因为
    局部高斯分母过大而被独立压成黑色。

    该函数保留原名称以兼容已有训练、评估和导入代码；目标实际使用
    全局 Reinhard 映射，而不是会在强对比边缘产生黑色光晕的局部分母。
    """
    hdr_rgb = np.maximum(np.asarray(hdr_rgb, dtype=np.float32), 0.0)
    luminance = (
        LINEAR_RGB_LUMA[0] * hdr_rgb[..., 0]
        + LINEAR_RGB_LUMA[1] * hdr_rgb[..., 1]
        + LINEAR_RGB_LUMA[2] * hdr_rgb[..., 2]
    )
    luminance = np.maximum(luminance, 0.0)
    key = reinhard_exposure_key(hdr_rgb)
    scaled_luminance = luminance * key
    mapped_luminance = scaled_luminance / (1.0 + scaled_luminance)
    chroma = hdr_rgb / np.maximum(luminance[..., None], 1e-8)
    reference = chroma * mapped_luminance[..., None]
    return np.clip(reference, 0.0, 1.0).astype(np.float32)


def log_hdr_tonemap(hdr_rgb: np.ndarray) -> np.ndarray:
    """用 log HDR 映射生成保留颜色和细节的 [0, 1] 参考图。

    映射与 ``hdr_for_tensorboard`` 保持一致：先截断负值并执行
    ``log1p``，再使用整张图的 99% 分位数作为曝光尺度。使用整张图
    计算尺度后再裁剪 patch，可避免同一张图的不同 patch 亮度不一致。
    """
    import numpy as np

    log_image = np.log1p(np.maximum(hdr_rgb, 0.0))
    scale = max(float(np.percentile(log_image, 99.0)), 1e-6)
    return np.clip(log_image / scale, 0.0, 1.0).astype(np.float32)


def bilateral_tonemap_target(
    hdr_rgb: np.ndarray,
    sigma_space: float = 16.0,
    sigma_color: float = 0.35,
    detail_strength: float = 0.70,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> np.ndarray:
    """使用对数亮度双边分解生成局部 tone mapping 训练目标。

    双边滤波只平滑对数亮度的低频 base，不跨越明显边缘传播亮度。
    映射时压缩 base 并保留受控比例的 detail，从而比全局 Reinhard
    保留更多局部高光层次，同时降低普通高斯局部分母造成 halo 的风险。
    target 在整张 HDR 图上计算，再由数据集裁剪 patch，保证同一张图的
    所有 patch 使用一致的局部映射结果。
    """
    if sigma_space <= 0.0 or sigma_color <= 0.0:
        raise ValueError("sigma_space and sigma_color must be positive")
    if not 0.0 <= detail_strength <= 1.0:
        raise ValueError("detail_strength must be in [0, 1]")
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError("percentile range must satisfy 0 <= low < high <= 100")

    hdr_rgb = np.maximum(np.asarray(hdr_rgb, dtype=np.float32), 0.0)
    luminance = (
        LINEAR_RGB_LUMA[0] * hdr_rgb[..., 0]
        + LINEAR_RGB_LUMA[1] * hdr_rgb[..., 1]
        + LINEAR_RGB_LUMA[2] * hdr_rgb[..., 2]
    )
    luminance = np.maximum(luminance, 1e-8)

    log_luminance = np.log1p(luminance).astype(np.float32)
    base = cv2.bilateralFilter(
        log_luminance,
        d=0,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
    detail = log_luminance - base

    base_low, base_high = np.percentile(
        base,
        [percentile_low, percentile_high],
    )
    base_range = max(float(base_high - base_low), 1e-6)
    mapped_log_luminance = (
        (base - float(base_low)) / base_range
        + detail_strength * detail / base_range
    )
    mapped_luminance = np.clip(mapped_log_luminance, 0.0, 1.0)

    chroma = hdr_rgb / luminance[..., None]
    target = chroma * mapped_luminance[..., None]
    return np.clip(target, 0.0, 1.0).astype(np.float32)


def require_finite(tensor: torch.Tensor, name: str) -> None:
    """Fail at the first non-finite training value with useful range details."""
    if not torch.isfinite(tensor).all():
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel():
            details = f"finite range [{finite.min().item():.6g}, {finite.max().item():.6g}]"
        else:
            details = "no finite values"
        raise FloatingPointError(f"{name} contains NaN or Inf ({details})")


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """比较预测图与目标图的水平、垂直 Sobel 梯度。"""
    sobel_x = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = pred.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    channels = pred.shape[1]
    kernel_x = sobel_x.expand(channels, 1, 3, 3)
    kernel_y = sobel_y.expand(channels, 1, 3, 3)
    pred_gx = F.conv2d(pred, kernel_x, padding=1, groups=channels)
    pred_gy = F.conv2d(pred, kernel_y, padding=1, groups=channels)
    target_gx = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_gy = F.conv2d(target, kernel_y, padding=1, groups=channels)
    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] sRGB tensors to CIE Lab without loading Kornia."""
    rgb = rgb.clamp(0.0, 1.0)
    threshold = 0.04045
    rgb_linear = torch.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, rgb_linear)
    xyz = xyz / rgb.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xyz_root = torch.where(
        xyz > epsilon,
        xyz.clamp_min(0.0).pow(1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0,
    )
    fx, fy, fz = xyz_root.unbind(dim=1)
    return torch.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)],
        dim=1,
    )


def delta_ab_loss(pred_sdr: torch.Tensor, ref_sdr: torch.Tensor) -> torch.Tensor:
    """在 Lab 空间比较 a/b 色度通道，并归一化到与像素损失相近的尺度。

    Lab 的 a/b 通道通常以几十为单位，而 L1 图像损失通常小于 1。
    如果直接相加，色度项会压过亮度和结构项，模型容易收敛到低饱和度、
    灰蒙蒙的折中结果。
    """
    # rgb_to_lab 接收的是 sRGB；模型和 reference 仍保持线性域，
    # 因此只在计算色度损失时临时编码，不改变模型输出定义。
    pred_lab = rgb_to_lab(linear_to_srgb_tensor(pred_sdr))
    ref_lab = rgb_to_lab(linear_to_srgb_tensor(ref_sdr))
    ab_distance = torch.sqrt((pred_lab[:, 1:] - ref_lab[:, 1:]).square().sum(dim=1) + 1e-8)
    return torch.mean(ab_distance) / 100.0


def linear_to_srgb_tensor(linear: torch.Tensor) -> torch.Tensor:
    """将线性 RGB 张量编码为 sRGB，仅用于显示或图像保存。

    训练模型仍在保持线性的 ``[0, 1]`` 域中计算损失；该函数不能用于
    生成训练目标，否则会把 gamma 编码混入模型的输出定义。
    """
    linear = linear.clamp(0.0, 1.0)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )


def hdr_for_tensorboard(hdr: torch.Tensor) -> torch.Tensor:
    """将线性 HDR 压缩并编码为 sRGB TensorBoard 图像，不改变训练输入。"""
    display = torch.nan_to_num(
        hdr.detach(),
        nan=0.0,
        posinf=torch.finfo(hdr.dtype).max,
        neginf=0.0,
    ).clamp_min(0.0)
    display = torch.log1p(display)
    scale = torch.quantile(display.flatten(1), 0.99, dim=1).clamp_min(1e-6)
    display = linear_to_srgb_tensor(display / scale[:, None, None, None])
    return torch.nan_to_num(
        display,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def images_for_tensorboard(images: torch.Tensor) -> torch.Tensor:
    """将线性训练图像编码为 sRGB 后安全写入 TensorBoard。"""
    images = linear_to_srgb_tensor(images.detach())
    return torch.nan_to_num(
        images,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


class DoubleConv(nn.Module):
    """U-Net 中重复使用的两层 3x3 卷积加 ReLU 模块。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ToneMapUNet(nn.Module):
    """轻量级两级编码器-解码器色调映射网络。"""
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
        # 线性 HDR 的动态范围可能跨越多个数量级。先压缩到 log 域，
        # 避免少数极亮像素主导卷积响应；该操作会随 ONNX 一起导出，
        # 从而保证 TensorRT 推理与训练使用完全相同的输入变换。
        x = torch.log1p(x.clamp_min(0.0))
        # 跳跃连接保留浅层纹理，瓶颈层负责聚合更大的空间上下文。
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bn = self.bottleneck(self.pool2(d2))
        c2 = self.conv2(torch.cat([self.up2(bn), d2], dim=1))
        c1 = self.conv1(torch.cat([self.up1(c2), d1], dim=1))
        return torch.sigmoid(self.out_conv(c1))


class TMPatchDataset(torch.utils.data.Dataset):
    """随机裁剪 HDR 图像块，并在线生成 Reinhard 参考图。"""
    def __init__(self, data_source: str,patches_per_image: int = 16):
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
        return len(self.paths)*self.patches_per_image

    def __getitem__(self, idx: int):
        # 每次访问随机位置，使有限的 HDR 图像能够产生更多训练样本。
        path_idx = idx % len(self.paths)
        hdr = read_exr(self.paths[path_idx])
        reference = reinhard_local_tonemap(hdr)
        key = reinhard_exposure_key(hdr)
        hdr = hdr * key
        if not np.isfinite(reference).all():
            raise ValueError(f"Reinhard target contains NaN or Inf: {self.paths[path_idx]}")
        height, width = hdr.shape[:2]
        if height < PATCH_SIZE or width < PATCH_SIZE:
            raise ValueError(f"Image must be at least {PATCH_SIZE}x{PATCH_SIZE}, got {width}x{height}")
        y0 = random.randint(0, height - PATCH_SIZE)
        x0 = random.randint(0, width - PATCH_SIZE)
        crop = (slice(y0, y0 + PATCH_SIZE), slice(x0, x0 + PATCH_SIZE))
        # NumPy 图像是 (H, W, C)，网络输入统一为 PyTorch 的 (C, H, W)。
        to_tensor = lambda image: torch.from_numpy(image[crop]).permute(2, 0, 1).float()
        return to_tensor(hdr), to_tensor(reference)


def train_tone_map() -> None:
    """执行色调映射训练并保存模型权重。"""
    os.makedirs("./checkpoint", exist_ok=True)
    tm_net = ToneMapUNet().to(DEVICE)
    optimizer = torch.optim.Adam(tm_net.parameters(), lr=1e-4)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    dataset = TMPatchDataset(TRAIN_LIST_TXT)
    # EXR/ImageIO/OpenCV 使用原生库，在 Linux fork 出来的 worker 中容易触发段错误。
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    global_step = 0
    for epoch in range(EPOCHS_TM):
        tm_net.train()
        total_loss = 0.0
        for hdr_in, ref_sdr in tqdm(loader):
            # ref_sdr 是 Reinhard 映射结果，作为监督目标；hdr_in 是线性 HDR 输入。
            hdr_in, ref_sdr = hdr_in.to(DEVICE), ref_sdr.to(DEVICE)
            require_finite(hdr_in, "hdr_in")
            require_finite(ref_sdr, "ref_sdr")
            pred_sdr = tm_net(hdr_in)
            require_finite(pred_sdr, "pred_sdr")
            l1_loss = F.l1_loss(pred_sdr, ref_sdr)
            grad_loss = gradient_loss(pred_sdr, ref_sdr)
            dab_loss = delta_ab_loss(pred_sdr, ref_sdr)
            loss = (
                LAMBDA_L1 * l1_loss
                + LAMBDA_GRAD * grad_loss
                + LAMBDA_DAB * dab_loss
            )
            require_finite(l1_loss, "l1_loss")
            require_finite(grad_loss, "grad_loss")
            require_finite(dab_loss, "dab_loss")
            require_finite(loss, "loss")
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in tm_net.named_parameters():
                if parameter.grad is not None:
                    require_finite(parameter.grad, f"gradient {name}")
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
        print(
            f"[ToneMap-Train] epoch {epoch:03d}, tm_loss {average_loss:.6f}, "
            f"l1 {l1_loss.item():.6f}, gradient {grad_loss.item():.6f}, "
            f"delta_ab {dab_loss.item():.6f}"
        )
        writer.add_images("images/hdr_input_log", hdr_for_tensorboard(hdr_in[:1]), epoch)
        writer.add_images("images/reference", images_for_tensorboard(ref_sdr[:1]), epoch)
        writer.add_images("images/prediction", images_for_tensorboard(pred_sdr[:1]), epoch)
        if epoch % 20 == 0:
            torch.save(tm_net.state_dict(), f"./checkpoint/tm_epoch{epoch:03d}.pth")
            print(f"TM net checkpoint saved to ./checkpoint/tm_epoch{epoch:03d}.pth")
    torch.save(tm_net.state_dict(), TM_CKPT)
    writer.close()
    print(f"TM net saved to {TM_CKPT}")


if __name__ == "__main__":
    train_tone_map()
