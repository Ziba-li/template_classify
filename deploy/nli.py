# -*- coding: utf-8 -*-
import abc
import torch
import onnxruntime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from tqdm import tqdm
from torch.nn import functional as F
from decimal import Decimal
from scipy.special import softmax
import transformers
from custom_logging import CustomizeLogger
from config import settings
from transformers import BertForSequenceClassification, AutoTokenizer, BertTokenizerFast

config_path = Path(__file__).with_name("logging_config.json")
logger = CustomizeLogger.make_logger(config_path)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class Predict:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def single_predict(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def batch_predict(self, *args, **kwargs):
        pass

    def total_control(self, *args, **kwargs) -> Dict[str, Any]:
        data = kwargs.get("data", None)
        batch = kwargs.get("batch", None) if kwargs.get("batch", None) else 64
        token_input = kwargs.get("tokens", None)
        if (not data) or data == [""]:
            return {"categories": None, "probabilities": None}
        if isinstance(data, str):
            logger.info("单条预测：")
            if token_input:
                predict_result = self.single_predict(token_input)
            else:
                predict_result = self.single_predict(data)
        elif isinstance(data, list):
            logger.info("批量预测：")
            if token_input:
                predict_result = self.batch_predict(token_input, batch)
            else:
                predict_result = self.batch_predict(data, batch)
        else:
            logger.info(f"传入数据类型为：{type(data)}")
            raise "传输数据类型有问题！"
        return predict_result


class PytorchPredict(Predict):
    def __init__(self, model: str = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model, Truncation=True)
        self.model = BertForSequenceClassification.from_pretrained(model).to(device)

    def single_predict(self, text: str = "") -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=500, truncation=True).to(device)
        with torch.no_grad():
            logit = self.model(**inputs).logits
            scores = F.softmax(logit, dim=-1).cpu().detach().numpy()[0].tolist()
        return {"categories": np.argmax(scores),
                "probabilities": float(Decimal(max(scores)).quantize(Decimal("0.0000")))}

    def batch_predict(self, texts: List[str], batch: int = 64):
        labels, probabilities = [], []
        for i in tqdm(range(0, len(texts), batch)):
            inputs = self.tokenizer(texts[i: i + batch], return_tensors="pt", padding=True,
                                    max_length=500, truncation=True).to(device)
            with torch.no_grad():
                logit = self.model(**inputs).logits
                scores = F.softmax(logit, dim=-1).cpu().detach().numpy().tolist()
                label = list(np.argmax(scores, axis=1))
                labels.extend(label)
                probabilities.extend([max(i) for i in scores])
        assert len(labels) == len(texts) == len(probabilities)
        return {"categories": labels,
                "probabilities": [float(Decimal(i).quantize(Decimal("0.0000"))) for i in probabilities]}

    def total_control(self, data: Union[str, List[str]] = "", batch_size: int = 64):
        with torch.no_grad():
            result = super(PytorchPredict, self).total_control(data=data, batch=batch_size)
        return result


class ONNXPredict(Predict):
    def __init__(self, onnx_model_path: str):
        self.tokenizer = BertTokenizerFast(vocab_file=settings.MODEL_VOCAB_FOLDER_PATH, do_lower_case=True)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 设置图级别优化（最全面）
        self.session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=["CUDAExecutionProvider",
                                                                                              "CPUExecutionProvider"])

    def single_predict(self, input_token: transformers.tokenization_utils_base.BatchEncoding):
        logit = self.session.run(None, {
            'input_ids': np.array([input_token['input_ids']], dtype=np.int64),  # 仅支持2维list或者ndarray
            'input_mask': np.array([input_token['attention_mask']], dtype=np.int64),
            'segment_ids': np.array([input_token['token_type_ids']], dtype=np.int64)})
        probabilities = softmax(logit[0], axis=-1)
        categories = np.argmax(probabilities, axis=1).tolist()
        return {"categories": categories[0],
                "probabilities": float(Decimal(max(probabilities.tolist()[0])).quantize(Decimal("0.0000")))}

    def batch_predict(self, input_token: transformers.tokenization_utils_base.BatchEncoding, batch: int = 64):
        labels, probabilities = [], []
        for i in tqdm(range(0, len(input_token["input_ids"]), batch)):
            logit = self.session.run(None, {
                'input_ids': np.array(input_token['input_ids'][i: i + batch], dtype=np.int64),
                'input_mask': np.array(input_token['attention_mask'][i: i + batch], dtype=np.int64),
                'segment_ids': np.array(input_token['token_type_ids'][i: i + batch], dtype=np.int64)})
            batch_probabilities = softmax(logit[0], axis=-1)
            categories = list(np.argmax(batch_probabilities, axis=1))
            labels.extend(categories)
            probabilities.extend(batch_probabilities)
        assert len(labels) == len(input_token["input_ids"]) == len(probabilities)
        return {"categories": labels,
                "probabilities": [float(Decimal(max(i.tolist())).quantize(Decimal("0.0000"))) for i in probabilities]}

    def total_control(self, data: Union[List[str], str], batch_size: int = 64):
        input_tokenizer = self.tokenizer(data if data else [""], max_length=500, truncation=True,
                                         padding='max_length')
        result = super(ONNXPredict, self).total_control(data=data, batch=batch_size, tokens=input_tokenizer)
        return result


if __name__ == '__main__':
    p = ONNXPredict(onnx_model_path="../model/emo.onnx")
    # p = PytorchPredict(model="../model/emotional_cls_4")
    import pandas as pd

    txt = list(pd.read_csv("../data/test.csv", sep='\t').to_dict()['sentence'].values())
    res = p.total_control(txt)
    # res = p.total_control(["超参数[公式]主要控制soft label和hard label的loss比例，Distilled BiLSTM在实验中发现只使用soft label会得到最好的效果。
    # 个人建议让soft label占比更多一些，一方面是强迫学生更多的教师知识，另一方面实验证实soft target可以起到正则化的作用，让学生模型更稳定地收敛。超参数T主要控制
    # 预测分布的平滑程度，TinyBERT实验发现T=1更好，BERT-PKD的搜索空间则是{5, 10, 20}。因此建议在1～20之间多尝试几次，T越大越能学到teacher模型的泛化信息。比
    # 如MNIST在对2的手写图片分类时，可能给2分配0.9的置信度，3是1e-6，7是1e-9，从这个分布可以看量化意味着降低模型权重的精度。一个方法是 k 均值量化：给定模型权重矩阵
    # W，权重值为浮点数。将权重值分为 N 组。然后将 W 转换成 [1…N] 的整型矩阵，每个元素指代 N 个聚类中心之一。这样，我们就把矩阵元素从 32 位浮点数压缩成 log(N) 位
    # 的整形值。计算机架构一般只会允许降到 8 位或是 1 位。但后者十分罕见，毕竟二值化一个矩阵意味着只能还有两种不同的值，这对模型的损伤是巨大的出2和3有一定的相似度，量
    # 化意味着降低模型权重的精度。一个方法是 k 均值量化：给定模型权重矩阵 W，权重值为浮点数。将权重值分为 N 组。然后将 W 转换成 [1…N] 的整型矩阵，每个元素指代 N 个聚
    # 类中心之一。这样，我们就把矩阵元素从 32 位浮点数压缩成 log(N) 位的整形值。计算机架构一般只会允许降到 8 位或是 1 位。但后者十分罕见，毕竟二值化一个矩阵意味着只能
    # 还有两种不同的值，这对模型的损伤是巨大的。这种时候可以调大T，让概率分布更平滑，展示teacher更多的泛化能力。"])
    print(res)
