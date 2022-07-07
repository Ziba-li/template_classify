# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List, Union, Optional


class InputContent(BaseModel):
    text: Union[str, List[str]] = None


class InferenceResult(BaseModel):
    categories: Union[int, List[int]] = None
    probabilities:  Union[float, List[float]] = List[None]

