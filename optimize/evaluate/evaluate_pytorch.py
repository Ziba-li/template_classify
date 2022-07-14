# -*- coding: utf-8 -*-
import numpy as np
import torch

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from evaluate_factory import EvaluateFactory
from transformers import BertForSequenceClassification
import fire
from torch.nn import functional as F
from textpruner import inference_time

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class EvaluatePytorch(EvaluateFactory):
    def __init__(self, model: str):
        self.model = BertForSequenceClassification.from_pretrained(model).to(device)

    def transform_text(self, texts, batch_size=128):
        categories = []
        data_loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for inputs in tqdm(data_loader):
                inputs_token = {"input_ids": inputs["input_ids"].squeeze(1),
                                "token_type_ids": inputs["token_type_ids"].squeeze(1),
                                "attention_mask": inputs["attention_mask"].squeeze(1)}
                logit = self.model(**inputs_token).logits
                scores = F.softmax(logit, dim=-1).cpu().detach().numpy().tolist()
                category = np.argmax(scores, axis=1).tolist()
                categories.extend(category)
        return categories

    def evaluate_performance_and_accuracy(self, test_file,
                                          performance=True, accuracy=True, batch_size=128):
        if accuracy:
            acc = self.evaluate_predict_result(test_file, batch_size)
        else:
            acc = None
        if performance:
            dummy_inputs = [torch.randint(low=0, high=10000, size=(1, 128))]
            inference_time(self.model, dummy_inputs)
        return f"accuracy：{acc}"


def main(pytorch_model_path: str, test_file: str) -> None:
    eo = EvaluatePytorch(pytorch_model_path)
    inference_info = eo.evaluate_performance_and_accuracy(test_file, accuracy=True, performance=True, batch_size=32)
    print(inference_info)


if __name__ == '__main__':
    # main(pytorch_model_path="../model/emotional_cls_4", test_file="../data/dev.csv")
    # main(pytorch_model_path="../model_cut/pruned_models/pruned_H8.0F2048n_iters8", test_file="../data/dev.csv")
    main(pytorch_model_path="../../model_cut/pruned_models/pruned_H6.0F1536n_iters16", test_file="../../data/dev.csv")
    # main(pytorch_model_path="../model/hfl_rbt3_finished", test_file="../data/dev.csv")
