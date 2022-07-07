import requests
import json
from config import settings


def get_response(data: str or list):
    url = f"http://127.0.0.1:8000{settings.ONNX_ROUTING}"
    payload = json.dumps({
        "text": data
    })

    response = requests.request("POST", url, data=payload)
    return response.text


