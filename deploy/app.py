# -*- coding: utf-8 -*-
import fire
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from nli import PytorchPredict, ONNXPredict
from config import settings
from data_model import InputContent, InferenceResult
from custom_logging import CustomizeLogger
from fastapi.encoders import jsonable_encoder

config_path = Path(__file__).with_name("logging_config.json")
logging = CustomizeLogger.make_logger(config_path)

pytorch_app = FastAPI(debug=True, title=settings.TITLE)
onnx_app = FastAPI(debug=True, title=settings.TITLE)
app = FastAPI(debug=True, title=settings.TITLE)
tp = PytorchPredict(model=settings.PYTORCH_MODEL_PATH)
op = ONNXPredict(onnx_model_path=settings.ONNX_MODEL_PATH)


@pytorch_app.api_route(settings.PYTORCH_ROUTING, methods=["GET", "POST", "PUT"], response_model=InferenceResult)
def pytorch_predict(request: InputContent):
    text = request.text
    result = tp.total_control(text)
    logging.info(result)
    return result


@onnx_app.api_route(settings.ONNX_ROUTING, methods=["GET", "POST", "PUT"], response_model=InferenceResult)
def onnx_predict(request: InputContent):
    text = request.text
    result = op.total_control(text)
    logging.info(result)
    return result


if __name__ == '__main__':
    # fire.Fire(main)
    uvicorn.run(onnx_app, host="0.0.0.0", port=8000)
