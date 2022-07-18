# -*- coding: utf-8 -*-
import abc
import torch
import pandas as pd
from typing import List, Tuple, Union
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import Dataset
from textpruner import inference_time
from onnx_performance_time import InfluenceONNXTime as Iot
from transformers.tokenization_utils_base import BatchEncoding
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
    def text_2_id(text: str or List[str]):
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

    def evaluate_performance_and_accuracy(self, test_file: str, dummy_inputs: BatchEncoding,
                                          model: Union[BertForSequenceClassification, None],
                                          inference_time_function: Union[inference_time, Iot.calculate_inference_time],
                                          performance: bool = True, accuracy: bool = True,
                                          batch_size=128) -> Tuple[Union[str, None], Union[str, None], Union[str, None]]:

        if accuracy:
            acc = self.evaluate_predict_result(test_file, batch_size)
        else:
            acc = None
        if performance:
            mean, std = inference_time_function(model, dummy_inputs)
        else:
            mean, std = None, None

        return f"accuracy：{acc}", f"Mean inference time: {mean:.2f}ms", f"Standard deviation: {std:.2f}ms"
