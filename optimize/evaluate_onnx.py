# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime
import pandas as pd
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Dataset
from scipy.special import softmax
from transformers import BertTokenizerFast
from deploy.config import settings
import fire


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
        tokenizer = BertTokenizerFast(vocab_file=settings.MODEL_VOCAB_FOLDER_PATH, do_lower_case=True)
        return tokenizer(text, max_length=128, truncation=True,
                         padding='max_length', return_tensors='pt')


class EvaluateONNX:
    def __init__(self, onnx_model_path: str):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 设置图级别优化（最全面）
        self.session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=["CUDAExecutionProvider",
                                                                                              "CPUExecutionProvider"])

    def transform_text(self, texts: TestDataset) -> List[int]:
        total = []
        data_loader = DataLoader(texts, batch_size=128, shuffle=False)
        for batch in tqdm(data_loader):
            inputs = {'input_ids': batch['input_ids'].squeeze(1).numpy(),
                      'attention_mask': batch['attention_mask'].squeeze(1).numpy(),
                      'token_type_ids': batch['token_type_ids'].squeeze(1).numpy()}
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

    def evaluate_predict_result(self, test_file: str) -> float:
        with open(test_file, 'r', encoding='utf-8') as f1:
            data = pd.read_table(f1)
        data_dict = data.to_dict(orient='list')
        sentences = TestDataset(data_dict['sentence'])
        predicted_result = self.transform_text(sentences)
        true_counter = 0
        assert len(data_dict['label']) == len(predicted_result)
        for original, predict in zip(data_dict['label'], predicted_result):
            if original == predict:
                true_counter += 1
        accuracy_number = true_counter / len(predicted_result) * 100
        return accuracy_number


def main(onnx_model_path: str, test_file: str) -> None:
    eo = EvaluateONNX(onnx_model_path)
    acc = eo.evaluate_predict_result(test_file)
    print(onnx_model_path, acc)


if __name__ == '__main__':
    main(onnx_model_path="../model/emo.onnx", test_file="../data/dev.csv")
    # main(onnx_model_path="quantify/3.opt.onnx", test_file="../data/dev.csv")
    # fire.Fire(main)





























