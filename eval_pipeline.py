"""模式 2：评估完整的 ``noisy HDR -> denoise -> tone mapping`` 流程。

该脚本串联两个已经训练好的网络，测量噪声存在时最终 SDR 输出相对
Reinhard 参考图的质量，因此比单独评估更接近实际使用场景。
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch

from ai_tonemap import ToneMapUNet, reinhard_exposure_key, reinhard_local_tonemap
from eval_separate import (
    _read_validation_paths,
    compute_delta_e,
    compute_psnr,
    validate_tonemap_image,
)
from hdr_denoise import DenoiseUNetRes, add_poisson_gaussian_noise, read_exr


# 两个阶段分别加载各自的 checkpoint；推理阶段不更新任何参数。
DENOISE_CKPT = "./checkpoint/denoise_best.pth"
TM_CKPT = "./checkpoint/tm_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_full_pipeline() -> None:
    """加载两个模型，执行完整推理链并打印平均指标。"""
    denoise_net = DenoiseUNetRes().to(DEVICE)
    denoise_net.load_state_dict(torch.load(DENOISE_CKPT, map_location=DEVICE))
    tm_net = ToneMapUNet().to(DEVICE)
    tm_net.load_state_dict(torch.load(TM_CKPT, map_location=DEVICE))
    denoise_net.eval()
    tm_net.eval()
    psnr_scores = []
    delta_scores = []
    for path in _read_validation_paths():
        # clean_hdr 只用于生成噪声输入和参考答案，不会直接送入网络。
        clean_hdr = read_exr(path)
        noisy_hdr, _ = add_poisson_gaussian_noise(clean_hdr)
        # 与训练阶段使用完全相同的稳定 Reinhard 参考生成逻辑。
        reference_np = reinhard_local_tonemap(clean_hdr)
        validate_tonemap_image(reference_np, f"Reinhard reference for {path}")
        input_tensor = torch.from_numpy(noisy_hdr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            # 去噪网络输出的是噪声残差，因此从 noisy_hdr 中减去它，
            # 再把恢复出的 HDR 图像交给色调映射网络。
            denoised_hdr = input_tensor - denoise_net(input_tensor)
            # 评估目标由 clean_hdr 生成，因此使用同一张 clean_hdr 的 key，
            # 避免去噪残差改变全局亮度后引入额外的目标不一致。
            exposure_key = reinhard_exposure_key(clean_hdr)
            tonemap_input = denoised_hdr * exposure_key
            prediction = tm_net(tonemap_input)
        prediction_np = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        validate_tonemap_image(prediction_np, f"ToneMap prediction for {path}")
        reference_tensor = torch.from_numpy(reference_np).permute(2, 0, 1).float().to(DEVICE)
        # prediction_np 是 (H, W, C) 的 NumPy 图像，预测张量仍是 (C, H, W)。
        psnr_scores.append(compute_psnr(prediction_np, reference_np))
        delta_scores.append(compute_delta_e(prediction.squeeze(0), reference_tensor))
    print("==== Full pipeline eval (noisy -> denoise -> tonemap) ====")
    print(f"Average PSNR(SDR): {np.mean(psnr_scores):.3f}")
    print(f"Average Delta-E (ab): {np.mean(delta_scores):.3f}")


if __name__ == "__main__":
    eval_full_pipeline()