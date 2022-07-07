from pathlib import Path
from pydantic import BaseSettings, HttpUrl
from typing import Any, Dict, Optional, Union


class Settings(BaseSettings):
    # 项目基本信息
    # 项目名
    PROJECT_NAME: str = '用于XX场景的分类'
    # 项目类型 CLS
    PROJECT_TYPE: str = 'CLS'
    # 模型路径
    MODEL_VOCAB_FOLDER_PATH: str = str(Path('../deploy', "vocab.txt"))
    # 项目描述
    DESCRIPTION: str = "这是一个示例"
    # 版本
    VERSION: str = '0.1.0'
    # git 地址
    GIT_URL: Optional[HttpUrl] = None
    # 项目负责人信息
    CONTACT = {
        "name": "Deadpoolio the Amazing",
        "url": "http://x-force.example.com/contact/",
        "email": "dp@x-force.example.com",
    }

    # 接口基本信息
    # 接口标题
    TITLE: str = "对应于业务的标题"
    # PYTORCH模型文件夹路径
    PYTORCH_MODEL_PATH: str = "../model/emotional_cls_4"
    # ONNX模型路径
    ONNX_MODEL_PATH: str = "../model/emo.onnx"
    # PYTORCH路由
    PYTORCH_ROUTING: str = "/oneline/pytorch_classify"
    # ONNX路由
    ONNX_ROUTING: str = "/oneline/onnx_classify"


settings = Settings()

