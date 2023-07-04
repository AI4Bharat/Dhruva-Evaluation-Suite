import argparse
import base64
import json
import sys

sys.path.append("..")
from schema.services.request.ulca_asr_inference_request import ULCAAsrInferenceRequest
from schema.services.request.ulca_tts_inference_request import ULCATtsInferenceRequest
from schema.services.request.ulca_ner_inference_request import ULCANerInferenceRequest
from schema.services.request.ulca_translation_inference_request import (
    ULCATranslationInferenceRequest,
)
from schema.services.request.ulca_transliteration_inference_request import (
    ULCATransliterationInferenceRequest,
)
from schema.services.request.ulca_s2s_inference_request import ULCAS2SInferenceRequest

ASR_TASK = "dhruva_asr"
TTS_TASK = "dhruva_tts"
NMT_TASK = "dhruva_nmt"
TRASNLITERATION_TASK = "dhruva_transliteration"


parser = argparse.ArgumentParser(description="Process four input strings")

# Add the arguments
parser.add_argument("--task", help="ASR, NMT or TTS", required=True)
parser.add_argument("--payload", help="Payload for NMT or TTS tasks", required=False)
parser.add_argument(
    "--payload_path", help="Path for Payload Audio file for ASR Task", required=False
)
parser.add_argument(
    "--payload_meta",
    help="Any metadata for the payload to include in the file name",
    required=True,
)
# parser.add_argument('--length', help='Access Token', required=False)
parser.add_argument("--source_language", help="Source Language", required=True)
parser.add_argument("--token", help="Access Token", required=True)

# NMT Arguments
parser.add_argument("--target_language", help="Target Language", required=False)

# TTS Arguments
parser.add_argument("--gender", help="Gender", required=False)

# ASR Arguments
parser.add_argument(
    "--sampling_rate", default="16000", help="Sampling Rate", required=False
)
parser.add_argument(
    "--encoding", default="base64", help="Encoding of the file", required=False
)
parser.add_argument(
    "--format", default="wav", help="Format of the file", required=False
)


args = parser.parse_args()
if args.payload is None and args.payload_path is None:
    print("Error: either --payload or --payloadPath must be provided")

if args.task == "NMT":
    if args.target_language is None:
        print("Error: --target must be provided for NMT")

elif args.task == "TTS":
    if args.gender is None:
        print("Error: --gender must be provided for TTS")

if args.task == "S2S":
    if args.target_language is None:
        print("Error: --target must be provided for S2S")
    if args.gender is None:
        print("Error: --gender must be provided for S2S")


if args.task == "NMT":
    # text = "बहुत समय पहले की बात है। वर्धमान नाम का एक व्यापारी दक्षिण भारत में रहता था एक दिन जब वो आराम कर रहा था तब उसके मन में खयाल आया की इस दुनिया में पैसा ही सब कुछ है जितना व्यक्ति को मिलता है वो उतना ही सक्तिशाली होता है दुश्मन भी अमीर दोस्त ढूंढते हैं वृद्ध भी जवान हो जाते हैं अगर उनके पास पैसा होता है"

    # Create an argument parser

    # Assign the argument values to variables
    text = args.payload
    source_lan = args.source_language
    target_lan = args.target_language
    ACCESS_TOKEN = args.token

    lua_file_path = f"nmt_test/{args.task}_{args.payload_meta}.lua"

    body = {
        "input": [{"source": ""}],
        "config": {"language": {"sourceLanguage": "", "targetLanguage": ""}},
    }
    body["input"][0]["source"] = text
    body["config"]["language"]["sourceLanguage"] = source_lan
    body["config"]["language"]["targetLanguage"] = target_lan
    body = ULCATranslationInferenceRequest(**body).dict()
    str_body = json.dumps(body).translate(str.maketrans({'"': r"\""}))

    # Create the Lua file
    with open(lua_file_path, "w") as f:
        f.write(f'wrk.method = "POST"\n')
        f.write(f'wrk.headers["Content-Type"] = "application/json"\n')
        f.write(f'wrk.headers["Authorization"] = "{ACCESS_TOKEN}"\n')
        f.write(f'wrk.body = "{str_body}"\n')


if args.task == "TTS":
    # Create an argument parser

    # Assign the argument values to variables
    text = args.payload
    source_lan = args.source_language
    gender = args.gender
    ACCESS_TOKEN = args.token

    # Define the path to save the Lua file
    lua_file_path = f"tts_test/{args.task}_{args.payload_meta}.lua"

    body = {
        "input": [{"source": ""}],
        "config": {"language": {"sourceLanguage": ""}, "gender": ""},
    }
    body["input"][0]["source"] = text
    body["config"]["language"]["sourceLanguage"] = source_lan
    body["config"]["gender"] = gender
    body = ULCATtsInferenceRequest(**body).dict()
    str_body = json.dumps(body).translate(str.maketrans({'"': r"\""}))

    # Create the Lua file
    with open(lua_file_path, "w") as f:
        f.write(f'wrk.method = "POST"\n')
        f.write(f'wrk.headers["Content-Type"] = "application/json"\n')
        f.write(f'wrk.headers["Authorization"] = "{ACCESS_TOKEN}"\n')
        f.write(f'wrk.body = "{str_body}"\n')

if args.task == "ASR":
    # Assign the argument values to variables
    path = args.payload_path
    sampling = args.sampling_rate
    encoding = args.encoding
    format = args.format
    source = args.source_language
    ACCESS_TOKEN = args.token
    with open(path, "rb") as file:
        wav_file = file.read()
        encoded_string = base64.b64encode(wav_file)
        # Encode the file.
        encoded_string = str(encoded_string, "ascii", "ignore")
        # print(encoded_string)
    # Define the path to save the Lua file
    lua_file_path = f"{args.task.lower()}_test/{args.task}_{args.payload_meta}.lua"
    body = {
        "audio": [{"audioContent": ""}],
        "config": {
            "language": {"sourceLanguage": ""},
            "audioFormat": "",
            "encoding": "",
            "samplingRate": "",
            "postProcessors": [],
        },
    }
    body["audio"][0]["audioContent"] = encoded_string
    body["config"]["language"]["sourceLanguage"] = source
    body["config"]["audioFormat"] = format
    body["config"]["encoding"] = encoding
    body["config"]["samplingRate"] = sampling
    body = ULCAAsrInferenceRequest(**body).dict()
    str_body = json.dumps(body).translate(str.maketrans({'"': r"\""}))
    with open(lua_file_path, "w") as f:
        f.write(f'wrk.method = "POST"\n')
        f.write(f'wrk.headers["Content-Type"] = "application/json"\n')
        f.write(f'wrk.headers["Authorization"] = "{ACCESS_TOKEN}"\n')
        f.write(f'wrk.body = "{str_body}"')

if args.task == "S2S":
    # Assign the argument values to variables
    path = args.payload_path
    sampling = args.sampling_rate
    encoding = args.encoding
    format = args.format
    source = args.source_language
    target = args.target_language
    gender = args.gender
    ACCESS_TOKEN = args.token
    with open(path, "rb") as file:
        wav_file = file.read()
        encoded_string = base64.b64encode(wav_file)
        # Encode the file.
        encoded_string = str(encoded_string, "ascii", "ignore")
        # print(encoded_string)
    # Define the path to save the Lua file
    lua_file_path = f"{args.task.lower()}_test/{args.task}_{args.payload_meta}.lua"
    body = {
        "audio": [{"audioContent": ""}],
        "config": {
            "language": {"sourceLanguage": "", "targetLanguage": ""},
            "audioFormat": "",
            "gender": "",
        },
    }
    body["audio"][0]["audioContent"] = encoded_string
    body["config"]["language"]["sourceLanguage"] = source
    body["config"]["language"]["targetLanguage"] = target
    body["config"]["audioFormat"] = format
    body["config"]["gender"] = gender
    body = ULCAS2SInferenceRequest(**body).dict()
    str_body = json.dumps(body).translate(str.maketrans({'"': r"\""}))
    with open(lua_file_path, "w") as f:
        f.write(f'wrk.method = "POST"\n')
        f.write(f'wrk.headers["Content-Type"] = "application/json"\n')
        f.write(f'wrk.headers["Authorization"] = "{ACCESS_TOKEN}"\n')
        f.write(f'wrk.body = "{str_body}"')

if args.task.lower() == "transliteration":
    # Create an argument parser

    # Assign the argument values to variables
    text = args.payload
    source_lan = args.source_language
    target_lan = args.target_language
    ACCESS_TOKEN = args.token

    # Define the path to save the Lua file
    lua_file_path = f"nmt_test/{args.task}_{args.payload_meta}.lua"

    body = {
        "input": [{"source": ""}],
        "config": {"language": {"sourceLanguage": "", "targetLanguage": ""}},
    }
    body["input"][0]["source"] = text
    body["config"]["language"]["sourceLanguage"] = source_lan
    body["config"]["language"]["targetLanguage"] = target_lan
    body = ULCATransliterationInferenceRequest(**body).dict()
    str_body = json.dumps(body).translate(str.maketrans({'"': r"\""}))

    # Create the Lua file
    with open(lua_file_path, "w") as f:
        f.write(f'wrk.method = "POST"\n')
        f.write(f'wrk.headers["Content-Type"] = "application/json"\n')
        f.write(f'wrk.headers["Authorization"] = "{ACCESS_TOKEN}"\n')
        f.write(f'wrk.body = "{str_body}"\n')
