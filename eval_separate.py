"""模式 1：分别评估去噪模型和色调映射模型。

去噪评估使用 ``noisy HDR -> denoised HDR``，色调映射评估使用干净 HDR
作为输入。两项评估相互独立，便于区分两个模型各自的误差来源。
"""

import math
import os
import random
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

from hdr_denoise import DenoiseUNetRes, read_exr as read_denoise_exr


# 模型权重、验证列表和运行设备配置集中放在这里，便于修改实验设置。
DENOISE_CKPT = "./checkpoint/denoise_best.pth"
TM_CKPT = "./checkpoint/tm_best.pth"
TM4_CKPT = "./checkpoint/tm_conditioned_best.pth"
VAL_DATA_SOURCE = "./test"
PNG_OUTPUT_DIR = "./eval_outputs/denoise"
TONEMAP_PNG_OUTPUT_DIR = "./eval_outputs/tonemap"
TONEMAP4_PNG_OUTPUT_DIR = "./eval_outputs/tonemap_conditioned"
# 评估阶段独立控制噪声强度：光子数越小，散粒噪声越强；读出噪声标准差越大，噪声越强。
PHOTON_SCALE_MIN = 50.0
PHOTON_SCALE_MAX = 300.0
READ_NOISE_STD_MIN = 0.01
READ_NOISE_STD_MAX = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_psnr(image: np.ndarray, reference: np.ndarray, max_val: float = 1.0) -> float:
    """根据均方误差计算 PSNR。"""
    mse = max(float(np.mean((image - reference) ** 2)), 1e-10)
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def compute_hdr_psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """在共同曝光尺度下计算线性 HDR 的 PSNR，避免极亮离群值抬高结果。"""
    scale = float(np.percentile(np.maximum(reference, 0.0), 99.0))
    scale = max(scale, 1e-6)
    return compute_psnr(image / scale, reference / scale, max_val=1.0)


def add_poisson_gaussian_noise(clean_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按本文件的评估参数添加泊松散粒噪声和高斯读出噪声。"""
    signal = np.maximum(clean_image, 0.0).astype(np.float32)
    photon_scale = random.uniform(PHOTON_SCALE_MIN, PHOTON_SCALE_MAX)
    read_noise_std = random.uniform(READ_NOISE_STD_MIN, READ_NOISE_STD_MAX)

    expected_photons = signal * photon_scale
    shot_sample = np.random.poisson(expected_photons).astype(np.float32)
    shot_noise = shot_sample / photon_scale - signal
    read_noise = np.random.normal(0.0, read_noise_std, signal.shape).astype(np.float32)

    noisy_image = np.maximum(signal + shot_noise + read_noise, 0.0)
    return noisy_image.astype(np.float32), (noisy_image - signal).astype(np.float32)


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """将 [0, 1] 范围的线性 RGB 转换为非线性 sRGB。"""
    linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
    return np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )


def save_hdr_png(path: str, image: np.ndarray, exposure_scale: float) -> None:
    """曝光缩放后将线性 HDR RGB 转成 sRGB，并保存为 8-bit PNG。"""
    display_linear = np.maximum(image, 0.0) / max(exposure_scale, 1e-6)
    srgb = linear_to_srgb(display_linear)
    png_rgb = np.round(srgb * 255.0).astype(np.uint8)
    png_bgr = cv2.cvtColor(png_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, png_bgr):
        raise IOError(f"Unable to write PNG image: {path}")


def save_tonemap_png(
    path: str,
    image: np.ndarray,
    apply_gamma: bool,
) -> None:
    """将 ``[0, 1]`` 色调映射图保存为 8-bit PNG。

    Reinhard 参考图和模型预测结果都定义在线性 RGB 的 ``[0, 1]`` 范围内。
    当 ``apply_gamma`` 为 ``True`` 时，先执行 sRGB 分段编码后再量化，
    得到适合普通显示器直接查看的图像；当其为 ``False`` 时，直接把
    线性 RGB 值量化为 8-bit，便于对比 gamma 校正对视觉亮度的影响。
    两种模式都不会修改输入数组。

    Args:
        path: 输出 PNG 文件路径。
        image: ``(H, W, 3)`` 的线性 RGB 图像，期望范围为 ``[0, 1]``。
        apply_gamma: 是否执行线性 RGB 到 sRGB 的 gamma 编码。
    """
    clipped = np.clip(image, 0.0, 1.0)
    output = linear_to_srgb(clipped) if apply_gamma else clipped
    png_rgb = np.round(output * 255.0).astype(np.uint8)
    png_bgr = cv2.cvtColor(png_rgb, cv2.COLOR_RGB2BGR)
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(path, png_bgr):
        raise IOError(f"Unable to write tonemap PNG image: {path}")


def validate_tonemap_image(image: np.ndarray, name: str) -> None:
    """检查色调映射图是否为有效的三通道 ``[0, 1]`` 图像。

    评估阶段直接使用线性浮点结果计算指标，并将同一结果保存为 PNG。
    因此在保存前显式检查有限值、通道数和范围，可以区分模型产生的异常
    输出与 PNG gamma 编码造成的正常亮度差异。
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(
            f"{name} must be in [0, 1], got range "
            f"[{float(image.min()):.6g}, {float(image.max()):.6g}]"
        )


def compute_delta_e(image: torch.Tensor, reference: torch.Tensor) -> float:
    """计算 Lab 空间 a/b 色度平面的平均欧氏距离。

    输入为单张 ``(C, H, W)`` RGB 张量；这里只关注色度差异，
    不把 L 通道的亮度误差计入该指标。
    """
    from ai_tonemap import linear_to_srgb_tensor, rgb_to_lab

    pred_lab = rgb_to_lab(linear_to_srgb_tensor(image.unsqueeze(0)))
    ref_lab = rgb_to_lab(linear_to_srgb_tensor(reference.unsqueeze(0)))
    ab_distance = (pred_lab[:, 1:] - ref_lab[:, 1:]).square().sum(dim=1).sqrt()
    return float(ab_distance.mean().cpu())


def _read_validation_paths() -> list[str]:
    """读取验证集目录中的 EXR，或读取旧格式的路径列表文件。"""
    source = Path(VAL_DATA_SOURCE)
    if source.is_dir():
        paths = [str(path) for path in sorted(source.rglob("*.exr"))]#递归搜索目录下所有 EXR 文件，按路径排序
    elif source.is_file():
        with source.open("r", encoding="utf-8") as file:
            paths = [line.strip() for line in file if line.strip()]
    else:
        raise FileNotFoundError(f"Validation data source not found: {VAL_DATA_SOURCE}")
    if not paths:
        raise ValueError(f"No EXR images found in {VAL_DATA_SOURCE}")
    return paths


def eval_denoise() -> None:
    """评估 HDR 域去噪质量并打印平均 PSNR。"""
    os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
    model = DenoiseUNetRes().to(DEVICE)
    model.load_state_dict(torch.load(DENOISE_CKPT, map_location=DEVICE))
    model.eval()
    scores = []
    scores_ori=[]
    for path in _read_validation_paths():
        # 每张干净图像都重新生成一次随机噪声，模拟实际采集过程。
        clean = read_denoise_exr(path)
        noisy, _ = add_poisson_gaussian_noise(clean)
        input_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            denoised = input_tensor - model(input_tensor)
        denoised_np = denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
        noisy_scale = float(np.percentile(np.maximum(noisy, 0.0), 99.0))
        denoised_scale = float(np.percentile(np.maximum(denoised_np, 0.0), 99.0))
        exposure_scale = max(noisy_scale, denoised_scale, 1e-6)
        output_stem = Path(path).stem
        save_hdr_png(
            str(Path(PNG_OUTPUT_DIR) / f"{output_stem}_noisy.png"),
            noisy,
            exposure_scale,
        )
        save_hdr_png(
            str(Path(PNG_OUTPUT_DIR) / f"{output_stem}_denoised.png"),
            denoised_np,
            exposure_scale,
        )
        scores.append(compute_hdr_psnr(denoised_np, clean))
        scores_ori.append(compute_hdr_psnr(noisy, clean))
    print("==== Denoise separate eval ====")
    print(f"Average PSNR(HDR normalized domain): {np.mean(scores):.3f}\n")
    print(f"Average PSNR(Original normalized domain): {np.mean(scores_ori):.3f}\n")


def eval_tonemap_clean() -> None:
    """在干净 HDR 输入上评估并保存 SDR 色调映射结果。

    对每张测试图生成训练时同源的 Reinhard 参考图，并保存参考图和
    模型预测图的两种版本：

    * ``*_linear.png``：不执行 gamma 校正，直接量化线性 RGB；
    * ``*_srgb.png``：执行 sRGB gamma 编码后再量化，适合普通显示。

    文件统一写入 ``TONEMAP_PNG_OUTPUT_DIR``，同时继续计算并打印 PSNR
    和色度 Delta-E 指标。指标在未 gamma 的线性 ``[0, 1]`` 数据上计算，
    不受 PNG 显示编码影响。
    """
    from ai_tonemap import (
        ToneMapUNet,
        read_exr as read_tonemap_exr,
        reinhard_exposure_key,
        reinhard_local_tonemap,
    )

    os.makedirs(TONEMAP_PNG_OUTPUT_DIR, exist_ok=True)
    model = ToneMapUNet().to(DEVICE)
    model.load_state_dict(torch.load(TM_CKPT, map_location=DEVICE))
    model.eval()
    psnr_scores = []
    delta_scores = []
    for path in _read_validation_paths():
        # reinhard_local_tonemap 是训练时使用的同源参考；
        # 这里使用更新后的稳定 Reinhard 亮度映射，不再使用旧的局部分母。
        # ToneMapUNet 内部会执行与训练时相同的 log1p 输入压缩。
        clean_hdr = read_tonemap_exr(path)
        reference_np = reinhard_local_tonemap(clean_hdr)
        print(f"Reinhard reference for {path}: min={reference_np.min():.6g}, max={reference_np.max():.6g}")
        validate_tonemap_image(reference_np, f"Reinhard reference for {path}")
        exposure_key = reinhard_exposure_key(clean_hdr)
        normalized_hdr = clean_hdr * exposure_key
        input_tensor = torch.from_numpy(normalized_hdr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            prediction = model(input_tensor)
        prediction_np = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print(f"ToneMap prediction for {path}: min={prediction_np.min():.6g}, max={prediction_np.max():.6g}")
        validate_tonemap_image(prediction_np, f"ToneMap prediction for {path}")
        reference_tensor = torch.from_numpy(reference_np).permute(2, 0, 1).float().to(DEVICE)
        output_stem = Path(path).stem
        output_dir = Path(TONEMAP_PNG_OUTPUT_DIR)
        save_tonemap_png(
            str(output_dir / f"{output_stem}_reference_linear.png"),
            reference_np,
            apply_gamma=False,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_reference_srgb.png"),
            reference_np,
            apply_gamma=True,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_prediction_linear.png"),
            prediction_np,
            apply_gamma=False,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_prediction_srgb.png"),
            prediction_np,
            apply_gamma=True,
        )
        psnr_scores.append(compute_psnr(prediction_np, reference_np))
        delta_scores.append(compute_delta_e(prediction.squeeze(0), reference_tensor))
    print("==== ToneMap separate eval (clean HDR input) ====")
    print(f"Average PSNR(SDR): {np.mean(psnr_scores):.3f}")
    print(f"Average Delta-E (ab): {np.mean(delta_scores):.3f}\n")


def eval_tonemap_conditioned() -> None:
    """评估四通道全局曝光条件色调映射模型并保存 PNG 结果。

    四通道输入由原始 HDR RGB 和 ``log1p(Reinhard key)`` 组成，与
    ``train_tonemap_conditioned.py`` 的训练输入完全一致。参考图仍由
    原始 HDR 生成，指标在未 gamma 的线性 ``[0, 1]`` 数据上计算。
    """
    from ai_tonemap import (
        read_exr as read_tonemap_exr,
        reinhard_exposure_key,
        reinhard_local_tonemap,
    )
    from train_tonemap_conditioned import (
        ConditionedToneMapUNet,
        make_conditioned_input,
    )

    os.makedirs(TONEMAP4_PNG_OUTPUT_DIR, exist_ok=True)
    model = ConditionedToneMapUNet().to(DEVICE)
    model.load_state_dict(torch.load(TM4_CKPT, map_location=DEVICE))
    model.eval()
    psnr_scores = []
    delta_scores = []

    for path in _read_validation_paths():
        clean_hdr = read_tonemap_exr(path)
        reference_np = reinhard_local_tonemap(clean_hdr)
        validate_tonemap_image(reference_np, f"Conditioned Reinhard reference for {path}")

        key = reinhard_exposure_key(clean_hdr)
        model_input = make_conditioned_input(clean_hdr, key)
        input_tensor = (
            torch.from_numpy(model_input)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )
        with torch.no_grad():
            prediction = model(input_tensor)
        prediction_np = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        validate_tonemap_image(prediction_np, f"Conditioned ToneMap prediction for {path}")

        output_dir = Path(TONEMAP4_PNG_OUTPUT_DIR)
        output_stem = Path(path).stem
        save_tonemap_png(
            str(output_dir / f"{output_stem}_reference_linear.png"),
            reference_np,
            apply_gamma=False,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_reference_srgb.png"),
            reference_np,
            apply_gamma=True,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_prediction_linear.png"),
            prediction_np,
            apply_gamma=False,
        )
        save_tonemap_png(
            str(output_dir / f"{output_stem}_prediction_srgb.png"),
            prediction_np,
            apply_gamma=True,
        )

        reference_tensor = (
            torch.from_numpy(reference_np)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )
        psnr_scores.append(compute_psnr(prediction_np, reference_np))
        delta_scores.append(compute_delta_e(prediction.squeeze(0), reference_tensor))

    print("==== Conditioned ToneMap separate eval (clean HDR input) ====")
    print(f"Average PSNR(SDR): {np.mean(psnr_scores):.3f}")
    print(f"Average Delta-E (ab): {np.mean(delta_scores):.3f}\n")


if __name__ == "__main__":
    eval_denoise()
    #eval_tonemap_conditioned()