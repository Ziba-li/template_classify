# -*- coding: utf-8 -*-
import torch
from collections.abc import Mapping
from transformers.tokenization_utils_base import BatchEncoding
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from tqdm import tqdm

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class InfluenceONNXTime:
    def __init__(self, session: InferenceSession, warm_up: int = 5, repetitions: int = 10):
        self.session = session
        self.warm_up = warm_up
        self.repetitions = repetitions

    def __influence(self, test_inputs):
        if device == "cpu":
            logits = self.session.run(None, {
                'input_ids': test_inputs['input_ids'].squeeze(1).numpy(),
                'input_mask': test_inputs['attention_mask'].squeeze(1).numpy(),
                'segment_ids': test_inputs['token_type_ids'].squeeze(1).numpy()
            })
        else:
            logits = self.session.run(None, {
                'input_ids': test_inputs['input_ids'].squeeze(1).cpu().numpy(),
                'input_mask': test_inputs['attention_mask'].squeeze(1).cpu().numpy(),
                'segment_ids': test_inputs['token_type_ids'].squeeze(1).cpu().numpy()
            })
        return logits

    def calculate_inference_time(self, _: None, test_inputs: BatchEncoding):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = torch.zeros(self.repetitions)
        for _ in tqdm(range(self.warm_up), desc='cuda-warm-up'):
            if isinstance(test_inputs, Mapping):
                _ = self.__influence(test_inputs)
            else:
                raise ValueError(f"{test_inputs} must be a dict!")

        for rep in tqdm(range(self.repetitions), desc='cuda-repetitions'):
            if isinstance(test_inputs, Mapping):
                starter.record()
                _ = self.__influence(test_inputs)
                ender.record()
            else:
                raise ValueError(f"{test_inputs} must be a dict!")
            torch.cuda.synchronize()
            elapsed_time_ms = starter.elapsed_time(ender)
            timings[rep] = elapsed_time_ms
        mean = timings.sum().item() / self.repetitions
        std = timings.std(dim=-1).item()
        print(f"Mean inference time: {mean:.2f}ms")
        print(f"Standard deviation: {std:.2f}ms")
        return mean, std
