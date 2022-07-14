# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime
import pandas as pd
from tqdm import tqdm
from typing import List
from evaluate_factory import EvaluateFactory
from torch.utils.data import DataLoader
from scipy.special import softmax
from evaluate_factory import TestDataset
import fire
import torch
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class EvaluateONNX(EvaluateFactory):
    def __init__(self, onnx_model_path: str):
        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 设置图级别优化（最全面）
        self.session = onnxruntime.InferenceSession(onnx_model_path, self.sess_options,
                                                    providers=["CUDAExecutionProvider"])

    def transform_text(self, texts: TestDataset, batch_size=128) -> List[int]:
        total = []
        data_loader = DataLoader(texts, batch_size=128, shuffle=False)
        for batch in tqdm(data_loader):
            inputs = {'input_ids': batch['input_ids'].squeeze(1).cpu().numpy(),
                      'attention_mask': batch['attention_mask'].squeeze(1).cpu().numpy(),
                      'token_type_ids': batch['token_type_ids'].squeeze(1).cpu().numpy()}
            logit = self.session.run(None, {
                'input_ids': inputs['input_ids'],
                'input_mask': inputs['attention_mask'],
                'segment_ids': inputs['token_type_ids']
            })
            total.append(logit[0])
        total_merge = np.concatenate(total, axis=0)
        probabilities = softmax(total_merge, axis=-1)
        categories = np.argmax(probabilities, axis=1).tolist()
        return categories


def main(onnx_model_path: str, test_file: str) -> None:
    eo = EvaluateONNX(onnx_model_path)
    acc = eo.evaluate_predict_result(test_file)
    print(onnx_model_path, acc)


if __name__ == '__main__':
    main(onnx_model_path="../../model/emo.onnx", test_file="../../data/dev.csv")
    # main(onnx_model_path="quantify/3.opt.onnx", test_file="../data/dev.csv")
    # fire.Fire(main)





























