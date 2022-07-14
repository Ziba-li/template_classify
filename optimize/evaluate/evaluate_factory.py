# -*- coding: utf-8 -*-
import abc
import torch
import pandas as pd
from typing import List
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, Dataset

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, data: List[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id(line)

    @staticmethod
    def text_2_id(text: str):
        tokenizer = BertTokenizerFast(vocab_file="../../deploy/vocab.txt", do_lower_case=True)
        return tokenizer(text, max_length=128, truncation=True,
                         padding='max_length', return_tensors='pt').to(device)


class EvaluateFactory:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def transform_text(self, *args, **kwargs):
        pass

    def evaluate_predict_result(self, test_file: str, batch_size=32) -> float:
        with open(test_file, 'r', encoding='utf-8') as f1:
            data = pd.read_table(f1)
        data_dict = data.to_dict(orient='list')
        sentences = TestDataset(data_dict['sentence'])
        predicted_result = self.transform_text(sentences, batch_size)
        true_counter = 0
        assert len(data_dict['label']) == len(predicted_result)
        for original, predict in zip(data_dict['label'], predicted_result):
            if original == predict:
                true_counter += 1
        accuracy_number = true_counter / len(predicted_result) * 100
        return accuracy_number
