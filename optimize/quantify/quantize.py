# -*- coding: utf-8 -*-
from onnxruntime.quantization import quantize_dynamic, QuantType
from loguru import logger
import fire


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """
    采用ONNX动态量化，谨慎使用
    模型量化过程主要将权重转换为INT8，在最终指标损失0.1~0.3%的基础上，提升模型推理速度，目前只能用于CPU服务器，GPU服务部署无法使用
    """
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    logger.info(f"quantized model saved to:{quantized_model_path}")


if __name__ == "__main__":
    quantize_onnx_model("./onnx_models/o_pf16.onnx", "./onnx_models/o_pf16_qq.onnx")
    # fire.Fire(quantize_onnx_model)

