# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from typing import List,Tuple, Union
from torch.utils.data import DataLoader
from evaluate_factory import EvaluateFactory, TestDataset
from transformers import BertForSequenceClassification
from torch.nn import functional as F
from textpruner import inference_time
import fire

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class EvaluatePytorch(EvaluateFactory):
    def __init__(self, model: str):
        self.model = BertForSequenceClassification.from_pretrained(model).to(device)

    def transform_text(self, texts, batch_size=128) -> List[int]:
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


def main(pytorch_model_path: str, test_file: str, batch_size=32) -> None:
    eo = EvaluatePytorch(pytorch_model_path)
    inference_info = eo.evaluate_performance_and_accuracy(test_file, model=eo.model,
                                                          dummy_inputs=TestDataset.text_2_id("你好"),
                                                          inference_time_function=inference_time,
                                                          accuracy=False, performance=True,
                                                          batch_size=batch_size)
    print(inference_info)


if __name__ == '__main__':
    # main(pytorch_model_path="../../model/cls_4_distill_rbt3", test_file="../../data/dev.csv")
    # main(pytorch_model_path="../../model/pruner_cls_4_distill_rbt3", test_file="../../data/dev.csv")
    # main(pytorch_model_path="../pruner/pruned_models/pruned_H8.0F2048", test_file="../../data/dev.csv")
    main(pytorch_model_path="../../model/emotional_cls_4", test_file="../../data/dev.csv")


























