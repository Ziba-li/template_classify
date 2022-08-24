# -*- coding: utf-8 -*-
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
from loguru import logger
import fire


def optimizer(fp16_pth, opt_fp16_path):
    """
    将转完onnx的模型直接做优化会报错，需要先进行fp16转换，然后再进行opt转换，opt转换完成后即可进行int8量化
    """

    # disable embedding layer norm optimization for better model size reduction
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False

    opt_model = optimizer.optimize_model(
        fp16_pth,
        'bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=opt_options)
    opt_model.save_model_to_file(opt_fp16_path)


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """
    采用ONNX动态量化，谨慎使用
    模型量化过程主要将权重转换为INT8，在最终指标损失0.1~0.3%的基础上，提升模型推理速度，目前只能用于CPU服务器，GPU服务部署无法使用
    """
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    logger.info(f"quantized model saved to:{quantized_model_path}")


# if __name__ == "__main__":
    # optimizer()
    # quantize_onnx_model()
    # fire.Fire(quantize_onnx_model)

