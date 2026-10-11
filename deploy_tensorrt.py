"""PC 部署入口：PyTorch -> ONNX -> TensorRT engine -> TensorRT 推理。"""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "denoise": ("hdr_denoise", "DenoiseUNetRes", "./checkpoint/denoise_best.pth", 3),
    "tonemap": ("ai_tonemap", "ToneMapUNet", "./checkpoint/tm_best.pth", 3),
    "tonemap4": (
        "train_tonemap_conditioned",
        "ConditionedToneMapUNet",
        "./checkpoint/tm_conditioned_best.pth",
        4,
    ),
}


def export_model(model_name: str, output_dir: Path, height: int, width: int, opset: int) -> None:
    """加载 PyTorch 检查点并导出为带动态空间尺寸的 ONNX 模型。

    ``model_name`` 必须是 ``MODELS`` 中注册的模型名称。
    导出示例输入使用 ``(1, C, height, width)``，批次、高度和宽度会被标记
    为动态维度；通道数由模型注册信息决定。模型权重从注册表指定的检查点读取，
    并始终在 CPU 上加载，避免导出阶段不必要地占用 GPU。
    """
    import importlib

    import torch

    module_name, class_name, checkpoint_path, input_channels = MODELS[model_name]
    model_class = getattr(importlib.import_module(module_name), class_name)# 导入模型类
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.onnx"
    example = torch.randn(1, input_channels, height, width)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        opset_version=opset,# ONNX opset 版本
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )
    print(f"Exported {model_name}: {output_path}")


def command_export(args: argparse.Namespace) -> None:
    """处理 ``export`` 子命令，并将一个或全部检查点导出为 ONNX。

    ``args.model`` 为 ``all`` 时遍历 ``MODELS`` 注册表，导出去噪、三通道
    色调映射和四通道条件色调映射模型；否则只导出用户指定的模型。
    导出尺寸、输出目录和 ONNX opset 版本均来自命令行参数。
    """
    names = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for name in names:
        export_model(name, args.output_dir, args.height, args.width, args.opset)


def parse_shape(value: str) -> tuple[int, int, int, int]:
    """解析并校验 TensorRT optimization profile 的 ``N,C,H,W`` 形状。

    输入格式必须是逗号分隔的四个整数 ``N,C,H,W``。当前部署网络固定
    使用单批次、三个或四个通道，且高度和宽度必须是 4 的倍数，以满足
    网络下采样和上采样结构的尺寸要求。格式或约束不满足时抛出
    ``ArgumentTypeError``，该异常会被 argparse 转换为命令行错误。

    Args:
        value: 命令行中的形状字符串，例如 ``"1,3,512,512"`` 或
            ``"1,4,512,512"``。

    Returns:
        校验后的四元组 ``(batch, channels, height, width)``。
    """
    import argparse

    shape = tuple(int(part) for part in value.split(","))
    if len(shape) != 4 or shape[0] != 1 or shape[1] not in (3, 4):
        raise argparse.ArgumentTypeError("shape must be N,C,H,W with N=1 and C in {3,4}")
    if shape[2] % 4 or shape[3] % 4:
        raise argparse.ArgumentTypeError("height and width must be divisible by 4")
    return shape


def command_build(args: argparse.Namespace) -> None:
    """读取 ONNX、创建 TensorRT optimization profile 并序列化 engine。

    函数先创建 TensorRT 网络和 ONNX 解析器，再根据最小、优化和最大
    形状建立 optimization profile。workspace 大小由 ``workspace_gib``
    控制，生成的序列化 engine 会写入 ``args.engine``；目标目录不存在
    时自动创建。

    如果 ONNX 解析失败，会收集 TensorRT 提供的全部错误信息后抛出；
    如果 engine 构建失败，则显式报告失败，而不会生成一个伪成功文件。
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)# 显式批次维度
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)# 网络标志位
    network = builder.create_network(network_flags)# 创建 TensorRT 网络
    parser = trt.OnnxParser(network, logger)# 创建 ONNX 解析器
    if not parser.parse(args.onnx.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Unable to parse ONNX:\n{errors}")

    config = builder.create_builder_config()# 创建 TensorRT 构建配置
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30))# 设置 workspace 大小
    profile = builder.create_optimization_profile()# 创建 TensorRT 优化配置文件 
    profile.set_shape("input", args.min_shape, args.opt_shape, args.max_shape)# 设置输入层形状
    config.add_optimization_profile(profile)# 添加优化配置文件

    serialized = builder.build_serialized_network(network, config)# 构建序列化 engine
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {args.onnx}")
    args.engine.parent.mkdir(parents=True, exist_ok=True)# 创建目标目录
    args.engine.write_bytes(serialized)# 写入序列化 engine 到文件
    print(f"Built TensorRT engine: {args.engine}")


def read_linear_exr(path: Path):
    """读取线性 HDR EXR，并统一返回 RGB ``float32`` 数组。

    灰度 EXR 会复制为三个通道；包含三个以上通道的图像只保留前三个
    通道（通常对应 RGB）。函数不会进行曝光、gamma 或数值范围压缩，
    因而保留原始线性 HDR 数值。读取失败、通道数不足或图像维度不符合
    预期时抛出异常，不返回静默的默认图像。

    Args:
        path: 待读取的 EXR 文件路径。

    Returns:
        形状为 ``(H, W, 3)``、类型为 ``float32`` 的 RGB 数组。
    """
    import imageio.v2 as imageio
    import numpy as np

    try:
        image = imageio.imread(path, format="EXR-FI")
    except Exception as error:
        raise IOError(f"Unable to read EXR image: {path}") from error
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Expected a 3-channel EXR image: {path}")
    return image


def add_poisson_gaussian_noise(image):
    """按照去噪模型训练时的参数，模拟真实传感器产生的 HDR 噪声。

    这里的输入 ``image`` 是干净的线性 RGB HDR 图像，形状为 ``(H, W, 3)``。
    函数返回同样形状的非负 ``float32`` 图像，供残差去噪网络作为 noisy
    输入。噪声由两部分组成：

    1. 光子散粒噪声（Poisson noise）：光子到达是随机事件，信号越强，
       期望光子数越大，噪声的绝对幅度通常也越大。
    2. 读出噪声（Gaussian noise）：传感器读出电路引入的近似信号无关噪声。

    ``photon_scale`` 和 ``read_noise_std`` 每次调用都随机采样，保持与
    ``hdr_denoise.py`` 训练阶段的 ``add_poisson_gaussian_noise`` 一致。
    因此同一张干净图像每次推理得到的 noisy 输入可能不同。
    """
    import random
    import numpy as np

    # 光子缩放因子：把线性 HDR 值换算为期望光子数。
    # 数值越大，泊松噪声相对越弱；范围必须与训练阶段保持一致。
    photon_scale = random.uniform(500.0, 3000.0)

    # 读出噪声的标准差。该噪声与信号强度无关，按像素独立采样。
    read_noise_std = random.uniform(0.001, 0.01)

    # 泊松分布的参数不能为负，因此将可能存在的负值（如黑电平误差）
    # 截断为 0。后续所有计算统一使用 float32，避免类型不一致。
    signal = np.maximum(image, 0.0).astype(np.float32)

    # 将线性信号转换为每个像素的期望光子数，并进行泊松采样。
    # 采样结果仍处于“光子数”单位，需要在下一步除以 photon_scale。
    expected_photons = signal * photon_scale
    shot_sample = np.random.poisson(expected_photons).astype(np.float32)

    # 将泊松采样结果还原到原始 HDR 信号单位，得到散粒噪声残差：
    # shot_sample / photon_scale = signal + shot_noise。
    shot_noise = shot_sample / photon_scale - signal

    # 生成与图像同形状的零均值高斯读出噪声。
    read_noise = np.random.normal(0.0, read_noise_std, signal.shape).astype(np.float32)

    # 将干净信号、散粒噪声和读出噪声相加。
    # 传感器输出通常不会低于黑电平，因此最终结果再次截断为非负值。
    noisy_image = np.maximum(signal + shot_noise + read_noise, 0.0)
    return noisy_image.astype(np.float32)


def save_linear_exr(path: Path, image) -> None:
    """将线性 RGB HDR 数组以 ``float32`` 写入 EXR 文件。

    该函数不做曝光缩放、gamma 编码、裁剪或通道交换，因此适合保存
    去噪后的线性 HDR 结果。输出目录会自动创建；写入失败会转换为包含
    目标路径的 ``IOError``，方便区分文件系统或图像编码问题。

    Args:
        path: 输出 EXR 文件路径。
        image: 形状为 ``(H, W, 3)`` 的线性 RGB 图像。
    """
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.imwrite(path, np.asarray(image, dtype=np.float32), format="EXR-FI")
    except Exception as error:
        raise IOError(f"Unable to write linear EXR image: {path}") from error


def _linear_to_srgb(linear):
    """将线性 RGB 值按 IEC sRGB 分段公式转换为非线性 sRGB 值。

    输入会先限制在 ``[0, 1]``，返回值也位于同一范围；该函数只做颜色
    编码，不负责 HDR 曝光缩放或 8 位量化。低于 sRGB 分段阈值的值使用
    线性公式，其余值使用幂函数公式。

    Args:
        linear: 任意可广播的线性 RGB 数组。

    Returns:
        与输入形状相同的非线性 sRGB 数组。
    """
    import numpy as np

    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def save_hdr_srgb_png(path: Path, image) -> None:
    """将线性 HDR 通过 99% 分位数曝光缩放后保存为 sRGB PNG。

    曝光尺度由所有像素的非负值第 99 百分位决定，用于抑制少量极亮
    离群值对整体显示的影响。写 PNG 前将 RGB 转为 OpenCV 使用的 BGR
    顺序并量化为 8 位；原始线性 HDR 数值不会被此函数修改。该文件
    主要用于观察 HDR 输入或去噪结果，不应当作为线性数据再次参与推理。

    Args:
        path: 输出 PNG 文件路径。
        image: 线性 RGB HDR 图像，通常形状为 ``(H, W, 3)``。
    """
    import numpy as np
    import cv2

    linear = np.maximum(image, 0.0)
    scale = max(float(np.percentile(linear, 99.0)), 1e-6)
    srgb = _linear_to_srgb(linear / scale)
    bgr8 = cv2.cvtColor(np.round(srgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr8):
        raise IOError(f"Unable to write sRGB PNG image: {path}")


def save_sdr_srgb_png(path: Path, image) -> None:
    """将模型输出的 ``[0, 1]`` 线性 RGB 编码为非线性 sRGB PNG。

    与 HDR 保存函数不同，这里不再进行曝光估计，因为色调映射模型的
    输出已经定义在 ``[0, 1]``。函数只执行 sRGB 编码、RGB 到 BGR 的
    通道转换和 8 位量化。写入失败时抛出 ``IOError``。

    Args:
        path: 输出 PNG 文件路径。
        image: 色调映射模型输出的 ``[0, 1]`` 线性 RGB 图像。
    """
    import cv2
    import numpy as np

    srgb = _linear_to_srgb(image)
    bgr8 = cv2.cvtColor(np.round(srgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr8):
        raise IOError(f"Unable to write sRGB PNG image: {path}")


def pad_to_multiple(image, multiple: int = 4):
    """用边缘反射填充图像，使高度和宽度成为指定倍数。

    返回填充后的图像以及原始 ``(height, width)``，调用方可在推理完成后
    按原始尺寸裁掉填充区域。反射填充不会引入固定颜色边框；当输入本来
    已经满足倍数要求时不会额外填充。

    Args:
        image: 至少包含高度和宽度维度的图像数组。
        multiple: 高度和宽度需要对齐到的正整数倍数。

    Returns:
        ``(padded_image, (original_height, original_width))``。
    """
    import cv2

    height, width = image.shape[:2]
    padded_height = (height + multiple - 1) // multiple * multiple# 计算填充后的高度，使其为 multiple 的倍数
    padded_width = (width + multiple - 1) // multiple * multiple
    padded = cv2.copyMakeBorder(
        image,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REFLECT_101,
    )
    return padded, (height, width)


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    """计算一维滑窗的起始位置，确保末端区域也被完整覆盖。

    常规步进无法整除剩余长度时，函数额外加入 ``length - tile_size``
    作为最后一个起点，使最后一个 tile 紧贴边缘，避免图像末端未参与
    推理。输入长度不超过 tile 尺寸时只返回起点 0。
    """
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def infer_tiled(runner, image, tile_size: int, overlap: int):
    """使用重叠 tile 推理大图，并对重叠区域的输出取平均。

    当输入尺寸小于 tile 尺寸时，会先反射填充到一个完整 tile；所有
    tile 都由 ``runner`` 执行，结果按覆盖次数累加后除以权重图，从而
    减少 tile 边界处的接缝。输入 tile 尺寸必须是 4 的倍数，重叠量
    必须满足 ``0 <= overlap < tile_size``。返回结果保持填充后的尺寸，
    原始尺寸由调用方负责裁剪。

    Args:
        runner: 接受 HWC RGB 数组并返回同尺寸 HWC 数组的推理对象。
        image: 待处理的 HWC 图像。
        tile_size: 每个推理块的边长。
        overlap: 相邻推理块在水平和垂直方向上的重叠像素数。

    Returns:
        对所有重叠 tile 做平均后的 HWC 输出数组。
    """
    import cv2
    import numpy as np

    if tile_size % 4 != 0:
        raise ValueError("tile-size must be divisible by 4")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile-overlap must be in [0, tile-size)")

    height, width = image.shape[:2]
    work_height = max(height, tile_size)
    work_width = max(width, tile_size)
    padded = cv2.copyMakeBorder(
        image,
        0,
        work_height - height,
        0,
        work_width - width,
        cv2.BORDER_REFLECT_101,
    )
    step = tile_size - overlap
    output = None
    weights = np.zeros((work_height, work_width, 1), dtype=np.float32)
    for y0 in _tile_starts(work_height, tile_size, step):
        for x0 in _tile_starts(work_width, tile_size, step):
            tile = runner(padded[y0:y0 + tile_size, x0:x0 + tile_size])
            if output is None:
                output = np.zeros((work_height, work_width, tile.shape[2]), dtype=np.float32)
            output[y0:y0 + tile_size, x0:x0 + tile_size] += tile
            weights[y0:y0 + tile_size, x0:x0 + tile_size] += 1.0
    return output / np.maximum(weights, 1.0)


class TensorRTRunner:
    def __init__(self, engine_path: Path):
        """加载 TensorRT engine，并准备可复用的执行上下文。

        优先使用 ``cuda-python`` 直接管理 CUDA 内存；如果该包不可用，
        则回退到带 CUDA 的 PyTorch 作为内存分配和 stream 后端。随后从
        engine 的 I/O 张量中识别唯一输入和输出名称。        engine 无法反序列化或没有可用 CUDA 后端时会立即抛出异常。
        初始化阶段还会从 engine 的 I/O 张量中找到唯一输入和输出，
        后续每次调用都复用同一个 execution context，避免重复加载 engine。

        Args:
            engine_path: 已构建的 TensorRT engine 文件路径。

        Raises:
            ImportError: TensorRT 或可用的 CUDA 后端依赖不存在时抛出。
            RuntimeError: engine 无法反序列化或 CUDA 设备不可用时抛出。
        """
        import tensorrt as trt

        self.trt = trt
        try:
            from cuda import cudart
        except ImportError:
            try:
                import torch
            except ImportError as error:
                raise ImportError(
                    "需要 cuda-python，或安装带 CUDA 的 PyTorch 作为 TensorRT 推理后端"
                ) from error
            if not torch.cuda.is_available():
                raise RuntimeError("cuda-python 不可用，且 PyTorch CUDA 设备不可用")
            self.backend = "torch"
            self.torch = torch
        else:
            self.backend = "cuda-python"
            self.cudart = cudart
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))# 关闭 TensorRT 日志
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())# 反序列化 engine
        if self.engine is None:
            raise RuntimeError(f"Unable to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()# 创建执行上下文
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]# 获取 engine 的 I/O 张量名称
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)# 找到输入张量名称
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)

    def __call__(self, image):
        """对一张 HWC、RGB、float32 图像执行一次 TensorRT 推理。

        TensorRT 输入格式为 ``NCHW``，因此这里负责增加 batch 维度并转换
        通道顺序。动态输出空间尺寸若未被 TensorRT 解析，则按当前全卷积
        网络的尺寸保持特性回退推断；最终返回不含 batch 维度的 HWC 数组。
        实际执行后端由初始化时的依赖探测结果决定：优先走 PyTorch CUDA
        tensor 路径，否则走 cuda-python 显式内存路径。

        Args:
            image: 形状为 ``(H, W, 3)`` 的连续或非连续 RGB ``float32`` 数组。

        Returns:
            形状为 ``(H, W, 3)`` 的 NumPy 输出数组。

        Raises:
            RuntimeError: 输入形状不在 engine 的 optimization profile 内，
                或 TensorRT 异步执行失败时抛出。
        """
        import numpy as np

        input_tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)# 转换为 NCHW 格式并增加 batch 维度
        if not self.context.set_input_shape(self.input_name, input_tensor.shape):
            raise RuntimeError(f"Input shape {tuple(input_tensor.shape)} is outside the TensorRT optimization profile")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dimension < 0 for dimension in output_shape):
            # 这两个全卷积网络保持 N、H、W 不变，TensorRT 10 某些构建结果
            # 在 Python API 中仍会返回未解析的动态输出空间维度。
            output_shape = (input_tensor.shape[0], 3, input_tensor.shape[2], input_tensor.shape[3])
        output_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.output_name))# 获取输出张量的 dtype
        if self.backend == "torch":
            return self._run_with_torch(input_tensor, output_shape, output_dtype)
        return self._run_with_cuda_python(input_tensor, output_shape, output_dtype)

    def _run_with_torch(self, input_tensor, output_shape, output_dtype):
        """使用 PyTorch CUDA tensor 和当前 CUDA stream 执行 engine。

        输入数组会复制到 CUDA，输出 tensor 的 dtype 按 TensorRT engine
        的输出类型创建。函数通过 tensor data pointer 将两块显存绑定到
        execution context，提交异步推理后同步当前 stream，最后将结果
        拷贝回 CPU 并从 NCHW 转回 HWC。
        """
        import numpy as np

        input_device = self.torch.from_numpy(input_tensor).cuda()
        torch_dtype = self.torch.from_numpy(np.empty((), dtype=output_dtype)).dtype
        output_device = self.torch.empty(output_shape, dtype=torch_dtype, device="cuda")
        stream = self.torch.cuda.current_stream()# 获取当前 CUDA stream
        self.context.set_tensor_address(self.input_name, input_device.data_ptr())# 设置输入张量的地址为 input_device 的 data pointer
        self.context.set_tensor_address(self.output_name, output_device.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()# 同步当前 CUDA stream
        return output_device.cpu().numpy()[0].transpose(1, 2, 0)# 将输出 tensor 转换为 HWC 格式并返回

    def _run_with_cuda_python(self, input_tensor, output_shape, output_dtype):
        """使用 cuda-python 显式分配、拷贝和释放 CUDA 内存执行 engine。

        输入和输出缓冲区以及 CUDA stream 都在本函数内管理，并在
        ``finally`` 中释放，确保 TensorRT 抛出异常时不会遗留设备资源。
        数据传输、推理和回传均加入同一个 stream，因此同步 stream 后
        才读取主机端输出数组。
        """
        import numpy as np

        output_tensor = np.empty(output_shape, dtype=output_dtype)
        check, stream = self.cudart.cudaStreamCreate()
        if check != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {check}")
        input_device = None
        output_device = None
        try:
            check, input_device = self.cudart.cudaMalloc(input_tensor.nbytes)
            if check != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc input failed: {check}")
            check, output_device = self.cudart.cudaMalloc(output_tensor.nbytes)
            if check != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc output failed: {check}")
            self.context.set_tensor_address(self.input_name, int(input_device))
            self.context.set_tensor_address(self.output_name, int(output_device))
            kind = self.cudart.cudaMemcpyKind
            self.cudart.cudaMemcpyAsync(input_device, input_tensor.ctypes.data, input_tensor.nbytes, kind.cudaMemcpyHostToDevice, stream)
            if not self.context.execute_async_v3(stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self.cudart.cudaMemcpyAsync(output_tensor.ctypes.data, output_device, output_tensor.nbytes, kind.cudaMemcpyDeviceToHost, stream)
            check = self.cudart.cudaStreamSynchronize(stream)[0]
            if check != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamSynchronize failed: {check}")
        finally:
            if input_device is not None:
                self.cudart.cudaFree(input_device)
            if output_device is not None:
                self.cudart.cudaFree(output_device)
            self.cudart.cudaStreamDestroy(stream)
        return output_tensor[0].transpose(1, 2, 0)


def command_infer(args: argparse.Namespace) -> None:
    """处理 ``infer`` 子命令，执行去噪或色调映射推理并保存结果。

    ``denoise`` 阶段先模拟训练分布中的噪声，再预测并减去噪声残差，
    输出线性 EXR 和两个可视化 PNG；``tonemap`` 阶段直接输出 sRGB PNG。
    原始图像经过分块推理后会裁回未填充尺寸。输出扩展名会根据阶段
    严格校验：去噪输出必须是 ``.exr``，色调映射输出必须是 ``.png``。

    Args:
        args: 由 ``build_parser`` 生成的 ``infer`` 子命令参数对象。

    Raises:
        ValueError: 输出扩展名与推理阶段不匹配时抛出。
    """
    image = read_linear_exr(args.input)
    if args.stage == "denoise":
        noisy_image = add_poisson_gaussian_noise(image)
        image, original_size = pad_to_multiple(noisy_image)
        runner = TensorRTRunner(args.denoise_engine)
        predicted_noise = infer_tiled(runner, image, args.tile_size, args.tile_overlap)
        image = image - predicted_noise
    elif args.stage == "tonemap":
        from ai_tonemap import reinhard_exposure_key

        image = image * reinhard_exposure_key(image)
        image, original_size = pad_to_multiple(image)
        runner = TensorRTRunner(args.tonemap_engine)
        image = infer_tiled(runner, image, args.tile_size, args.tile_overlap)
    else:
        from train_tonemap_conditioned import make_conditioned_input
        from ai_tonemap import reinhard_exposure_key

        key = reinhard_exposure_key(image)
        image = make_conditioned_input(image, key)
        image, original_size = pad_to_multiple(image)
        runner = TensorRTRunner(args.tonemap4_engine)
        image = infer_tiled(runner, image, args.tile_size, args.tile_overlap)
    height, width = original_size
    image = image[:height, :width]
    if args.stage == "denoise":
        if args.output.suffix.lower() != ".exr":
            raise ValueError("denoise output must use the .exr extension")
        noisy_image = noisy_image[:height, :width]
        save_linear_exr(args.output, image)
        noisy_srgb_output = args.noisy_srgb_output or args.output.with_name(f"{args.output.stem}_noisy_srgb.png")
        srgb_output = args.srgb_output or args.output.with_name(f"{args.output.stem}_srgb.png")
        save_hdr_srgb_png(noisy_srgb_output, noisy_image)
        save_hdr_srgb_png(srgb_output, image)
        print(f"Saved linear EXR: {args.output}")
        print(f"Saved noisy sRGB PNG: {noisy_srgb_output}")
        print(f"Saved sRGB PNG: {srgb_output}")
        return
    if args.output.suffix.lower() != ".png":
        raise ValueError("tonemap output must use the .png extension")
    save_sdr_srgb_png(args.output, image)
    print(f"Saved: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """创建完整的命令行参数解析器。

    解析器提供三个互相独立的子命令：

    * ``export``：加载 PyTorch 检查点并导出 ONNX；
    * ``build``：读取 ONNX 并构建 TensorRT engine；
    * ``infer``：使用 engine 对 EXR 图像执行去噪或色调映射。

    每个子命令通过 ``set_defaults(handler=...)`` 绑定处理函数，主函数
    解析参数后只需调用统一的 ``args.handler(args)``。

    Returns:
        已配置完成、且要求必须指定一个子命令的 argparse 解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export PyTorch checkpoints to ONNX")
    export.add_argument("--model", choices=["denoise", "tonemap", "tonemap4", "all"], default="all")
    export.add_argument("--output-dir", type=Path, default=Path("./deploy/onnx"))
    export.add_argument("--height", type=int, default=256)
    export.add_argument("--width", type=int, default=256)
    export.add_argument("--opset", type=int, default=17)
    export.set_defaults(handler=command_export)

    build = subparsers.add_parser("build", help="build a TensorRT engine from ONNX")
    build.add_argument("--onnx", type=Path, required=True)
    build.add_argument("--engine", type=Path, required=True)
    build.add_argument("--min-shape", type=parse_shape, default=(1, 3, 256, 256))
    build.add_argument("--opt-shape", type=parse_shape, default=(1, 3, 512, 512))
    build.add_argument("--max-shape", type=parse_shape, default=(1, 3, 1024, 1024))
    build.add_argument("--workspace-gib", type=int, default=2)
    build.set_defaults(handler=command_build)

    infer = subparsers.add_parser("infer", help="run TensorRT inference")
    infer.add_argument("--stage", choices=["denoise", "tonemap", "tonemap4"], required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--tile-size", type=int, default=512)
    infer.add_argument("--tile-overlap", type=int, default=32)
    infer.add_argument(
        "--srgb-output",
        type=Path,
        help="denoise 阶段额外保存的非线性 sRGB PNG 路径",
    )
    infer.add_argument(
        "--noisy-srgb-output",
        type=Path,
        help="denoise 阶段保存去噪前 noisy 输入的非线性 sRGB PNG 路径",
    )
    infer.add_argument("--denoise-engine", type=Path, default=Path("./deploy/engines/denoise.engine"))
    infer.add_argument("--tonemap-engine", type=Path, default=Path("./deploy/engines/tonemap.engine"))
    infer.add_argument("--tonemap4-engine", type=Path, default=Path("./deploy/engines/tonemap4.engine"))
    infer.set_defaults(handler=command_infer)
    return parser


def main() -> None:
    """运行部署脚本的命令行入口。

    函数从 ``sys.argv`` 解析用户选择的子命令及其参数，再调用解析器
    写入参数对象中的 ``handler``。具体的模型导出、engine 构建和推理
    逻辑分别由对应的命令处理函数负责。
    """
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
