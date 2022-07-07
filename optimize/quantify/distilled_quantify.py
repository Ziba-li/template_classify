#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch
from loguru import logger
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
from transformers import BertConfig, BertForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType


class Evaluate:
    def __init__(self, config_path: str, model_path: str, model_labels: int):
        bert_config = BertConfig.from_pretrained(config_path)
        bert_config.num_labels = model_labels
        self.model = BertForSequenceClassification(config=bert_config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def export_onnx_model(self, onnx_model_path: str):
        with torch.no_grad():
            inputs = {'input_ids': torch.ones(1, 128, dtype=torch.int64),
                      'attention_mask': torch.ones(1, 128, dtype=torch.int64),
                      'token_type_ids': torch.ones(1, 128, dtype=torch.int64)}

            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(self.model,  # model being run
                              (inputs['input_ids'],  # model input (or a tuple for multiple inputs)
                               inputs['attention_mask'],
                               inputs['token_type_ids']),  # model input (or a tuple for multiple inputs)
                              onnx_model_path,  # where to save the model (can be a file or file-like object)
                              opset_version=11,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input_ids',  # the model's input names
                                           'input_mask',
                                           'segment_ids'],
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                            'input_mask': symbolic_names,
                                            'segment_ids': symbolic_names,
                                            'output': symbolic_names})
            logger.info("ONNX Model exported to {0}".format(onnx_model_path))

    @staticmethod
    def optimize(path: str, out_path: str):
        # disable embedding layer norm optimization for better model size reduction
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_embed_layer_norm = False
        opt_model = optimizer.optimize_model(
            path,
            'bert',
            num_heads=12,
            hidden_size=768,
            opt_level=99,
            only_onnxruntime=True,  # 若为false，则会报错
            optimization_options=opt_options)
        opt_model.save_model_to_file(out_path)

    @staticmethod
    def quantize_onnx_model(onnx_model_path, quantized_model_path):
        quantize_dynamic(onnx_model_path,
                         quantized_model_path,
                         weight_type=QuantType.QUInt8)

        logger.info(f"quantized model saved to:{quantized_model_path}")


def main(config: str,
         finished_distill_path: str,
         num_labels: int,
         onnx_model_path: str,
         opt_path: str,
         quant_path: str):
    """
    config: 原始学生模型配置文件
    finished_model_path： 蒸馏完成的二进制文件
    num_labels： 分类模型的标签数量
    onnx_model_path： 蒸馏后转为onnx的文件
    opt_path： opt优化
    quant_path： 量化
    """
    e = Evaluate(config_path=config,
                 model_path=finished_distill_path,
                 model_labels=num_labels)
    e.export_onnx_model(onnx_model_path)
    e.optimize(onnx_model_path, opt_path)
    e.quantize_onnx_model(opt_path, quant_path)


if __name__ == "__main__":
    main(config="../../model/hfl_rbt3_finished/config.json",
         finished_distill_path="../distill/output_root_dir/mnli_t8_TbaseST4tiny_AllSmmdH1_lr10e60_bs128/gs11269.pkl",
         num_labels=4,
         onnx_model_path="3.onnx",
         opt_path="3.opt.onnx",
         quant_path="3.opt.quant.onnx")

