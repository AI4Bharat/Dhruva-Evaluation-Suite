from typing import List

import numpy as np
# from tqdm import tqdm
# import soundfile as sf
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

import gevent.ssl

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(debug=True)

headers = {}
# headers["Authorization"] = f"Bearer 9i2vidTyIdmWO1vpDbFJAk8trK2J5rTS"
headers["Authorization"] = f"Bearer 3GxAZcoEa5eyxf4QC6hIiz4PvI7cJheL"


def get_client():
    triton_client = http_client.InferenceServerClient(
        url="aml-asr-hi-endpoint.eastus.inference.ml.azure.com",
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context,
        concurrency=10
    )
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))
    return triton_client


def get_aws_client():
    triton_client = http_client.InferenceServerClient(
        url="15.207.122.54:8000",
        concurrency=40
    )
    health_ctx = triton_client.is_server_ready()
    print("Is server ready - {}".format(health_ctx))
    return triton_client


def get_nmt_client():
    triton_client = http_client.InferenceServerClient(
        # url="20.193.155.199:5000",
        url="indictrans--fairseq-t4.centralindia.inference.ml.azure.com",
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context,
        concurrency=10,
        # verbose=False
    )
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))
    return triton_client

# triton_client = get_client()
triton_nmt_client = get_nmt_client()
# triton_aws_client = get_aws_client()


class Input(BaseModel):
    inp: List


class ULCAASRLanguageConfig(BaseModel):
    sourceLanguage: str = "hi"


class ULCAASRTranscriptionFormatConfig(BaseModel):
    valoue: str = "transcript"


class ULCAASRConfig(BaseModel):
    language: ULCAASRLanguageConfig
    transcriptionFormat: ULCAASRTranscriptionFormatConfig
    audioFormat: str = "wav"
    samplingRate: str = "16000"
    postProcessors: str | None = None

class ULCAASRAudioContent(BaseModel):
    audioContent: list


class ULCAASR(BaseModel):
    config: ULCAASRConfig
    audio: list[ULCAASRAudioContent]


@app.get("/")
async def index():
    with open("../datasets/raw/MUCS/Hindi/test/audio/0116_003.wav", "rb") as f:
        audio_file = base64.b64encode(f.read()).decode("utf-8")

    return (200, "hi")


@app.post("/infer_e2e")
async def infer2(ip: Input):
    # triton_client = get_client()
    inp = np.array(ip.inp, dtype=np.uint8)
    input0 = http_client.InferInput("WAVPATH", inp.shape, "UINT8")
    input0.set_data_from_numpy(inp)
    output0 = http_client.InferRequestedOutput("TRANSCRIPTS")

    try:
        # response = triton_client.infer(
        response = triton_client.async_infer(
            "e2e",
            model_version="1",
            inputs=[input0],
            request_id=str(1),
            outputs=[output0],
            headers=headers
        )    
        response = response.get_result(block=True, timeout=2)
        # res = response._greenlet.get(block=True, timeout=2)
        # print("res: ", res.status_code)
        # if res.status_code >= 400:
            # raise HTTPException(status_code=res.status_code, detail="Error")

    except Exception as e:
        print("************: ", e)
        raise HTTPException(status_code=502, detail="Empty result")

    encoded_result = response.as_numpy("TRANSCRIPTS")
    res = [result.decode("utf-8") for result in encoded_result.tolist()]
    print("--------")

    return {"status": 200, "data": res}


def pad_batch(batch_data):
    batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
    # print("batch_data_lens: ", batch_data_lens)
    max_length = max(batch_data_lens)
    batch_size = len(batch_data)

    padded_zero_array = np.zeros((batch_size, max_length),dtype=np.float32)
    # print(padded_zero_array.shape)

    for idx, data in enumerate(batch_data):
        # print(data.shape)
        padded_zero_array[idx,0:batch_data_lens[idx]] = data

    return padded_zero_array, np.reshape(batch_data_lens,[-1,1])


@app.post("/infer")
async def infer(ip: ULCAASR):
    raw_audio = np.array(ip.audio[0].audioContent)
    # print(raw_audio.shape)
    o = pad_batch(raw_audio)

    input0 = http_client.InferInput("AUDIO_SIGNAL", o[0].shape, "FP32")
    input1 = http_client.InferInput("NUM_SAMPLES", o[1].shape, "INT32")
    input0.set_data_from_numpy(o[0])
    input1.set_data_from_numpy(o[1].astype('int32'))
    output0 = http_client.InferRequestedOutput('TRANSCRIPTS')

    try:
        response = triton_client.async_infer(
        # response = triton_aws_client.async_infer(
            "offline_conformer",
            model_version='1',
            inputs=[input0, input1],
            outputs=[output0],
            headers=headers
        )
        # Testing
        # response = response._greenlet.get(block=True, timeout=4)
        # print(response.status_code)
        response = response.get_result(block=True, timeout=2)
    except Exception as e:
        raise
        # print("************: ", e)
        # raise HTTPException(status_code=502, detail="AML issue")

    encoded_result = response.as_numpy("TRANSCRIPTS")
    res = [result.decode("utf-8") for result in encoded_result.tolist()]
    print("res: ", res)
    return {"status": 200, "data": res}


class ULCANMT(BaseModel):
    text: str
    source_language: str
    target_language: str


def get_string_tensor(string_value, tensor_name):
    string_obj = np.array([string_value], dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj


@app.post("/infer_nmt")
async def infer_nmt(ip: ULCANMT):
    inputs = [
        get_string_tensor(ip.text, "INPUT_TEXT"),
        get_string_tensor(ip.source_language, "INPUT_LANGUAGE_ID"),
        get_string_tensor(ip.target_language, "OUTPUT_LANGUAGE_ID")
    ]
    output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
    response = triton_nmt_client.async_infer("nmt", model_version='1', inputs=inputs, outputs=[output0], headers=headers)
    response = response.get_result(block=True, timeout=2)

    encoded_result = response.as_numpy('OUTPUT_TEXT')
    result = encoded_result.tolist()[0]
    return {"status": 200, "data": result.decode("utf-8")}
    # return {"status": 200, "data": "some text"}


#  request_body, json_size = _get_inference_request(
#             inputs=inputs,
#             request_id=request_id,
#             outputs=outputs,
#             sequence_id=sequence_id,
#             sequence_start=sequence_start,
#             sequence_end=sequence_end,
#             priority=priority,
#             timeout=timeout)

#     "v2/models/{}/versions/{}/infer".format(quote(model_name), model_version))

# self._post(
#     request_uri=request_uri,
#     request_body=request_body,
#     headers=headers,
#     query_params=query_params
# )

# headers["Inference-Header-Content-Length"] = json_size

# ssl_options=ssl_options,
# ssl_context_factory=ssl_context_factory,