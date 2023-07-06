# Dhruva Evaluation Suite

Welcome to Dhruva Evaluation Suite! This tool can be used to perform functional testing and performance testing of Dhruva.

## Getting Started

### Prerequisites
To perform functional testing, you will need a dataset hosted on Huggingface to easily hit the models and generate the results with required metrics.

Top perform Performance testing, you will need one data element for any of the task you want to benchmark, with which we will create a payload file.

### Installation
Clone the repository, and update the submodule with the following command:

```bash
git clone --recurse-submodules https://github.com/AI4Bharat/Dhruva-Evaluation-Suite.git
```

Install all the python packages using Poetry, and initiate the Virtual Environment:

```bash
poetry install
```
```bash
poetry shell
```
Set the DHRUVA_API_KEY environment variable in your bashrc / bash_profile
```bash
export DHRUVA_API_KEY="<token>"
```

## Functional Testing

To perform Fucntional testing for a task using the Dhruva Evaluation tool, build a simple YML file as shown below (according to the task). Examples given below:

### ASR

```yml
task:
  name: "evaluation"
  type: "asr"
  metric: "wer"

model:
  type: "REST"
  url: "<DOMAIN>/services/inference/asr?serviceId=ai4bharat%2Fconformer-hi-gpu--t4"

dataset:
  - name: "MUCS"
    path: "AI4Bharat/MUCS_internal"
    source_language: "hi"
    target_language : "hi"
    input_column : "path"
    label_column : "transcript"
    split: "test"
    subset: "hi"
    language : "hi" 
```
### NMT

```yml
task:
  name: "evaluation"
  type: "nmt"
  metric: "sacrebleu"

model:
  type: "REST"
  url: "<DOMAIN>/services/inference/translation?serviceId=ai4bharat/indictrans-v2-all-gpu--t4"

dataset:
  - name: "facebook/flores"
    path: "facebook/flores"
    source_language:
      - "hi"
      - "ta"
      - "te"
      - "mr"
      - "ml"
      - "sn"
      - "pa"
      - "as"
      - "bn"
      - "or"
      - "ur"
      - "ka"
      - "gu"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
    target_language:
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "en"
      - "hi"
      - "ta"
      - "te"
      - "mr"
      - "ml"
      - "sn"
      - "pa"
      - "as"
      - "bn"
      - "or"
      - "ur"
      - "ka"
      - "gu"
    split: "devtest"
    # subset: "hi"


```  
### Transliteraton

```yml
task:
  name: "evaluation"
  type: "transliteration"
  metric: "cer"

model:
  type: "REST"
  url: "https://<domain>/services/inference/transliteration?serviceId=ai4bharat%2Findicxlit--cpu-fsv2"

dataset:
  - name: "ai4bharat/aksharantar"
    path: "/home/dhruvauser/evaluation_suite/new_eval/Dhruva-Evaluation-Suite/src/functional/dhruva_datasets/Aksharantar/hin/test"
    source_language: "en"
    target_language: "hi"
    split: "test"
    input_column: "english word"
    label_column: "native word"
    subset: "hin"

```  

Use the YML file created to perform testig, using the following command:

```bash
python3 dhruva_evaluate.py -f  <FILE NAME>.yml
```  

## Performance Testing

### REST API Performance Testing

Generate the payload for the task, in the form of a "lua" file, that you intend to test out with one of the following commands:

#### ASR

```bash
python3 payload_generator.py --payload_path "<AUDIO FILE>.wav" --source_language "<LANG CODE>" --token "<API KEY>" --task "ASR" --payload_meta "<PAYLOAD METADATA>"
```  

#### NMT

```bash
python3 payload_generator.py --payload "<SENTENCE>" --source_language "<LANG CODE>" --target_language "<LANG CODE>" --token "<API KEY>" --task "NMT" --payload_meta "<PAYLOAD METADATA>"
```  
#### TTS

```bash
python3 payload_generator.py --payload "<SENTENCE>" --source_language "<LANG CODE>" --token "<API KEY>" --task "TTS" --payload_meta "<PAYLOAD METADATA>" --gender "<GENDER>"
```  
#### S2S

```bash
python3 payload_generator.py --payload_path "<AUDIO FILE>.wav" --source_language "<LANG CODE>" --target_language "<LANG CODE>" --token "<API KEY>" --task "S2S" --payload_meta "<PAYLOAD METADATA>" --gender "<GENDER>"
```

To perform the load test, we need to build a YML file in the following format which would help us run the tests:

```yml
task:
  name: "evaluation"
  type: "asr"

model:
  type: "REST"
  url: "<DOMAIN>/services/inference/asr?serviceId=ai4bharat%2Fconformer-hi-gpu--t4"

params:
  payload_path: "ASR_check.lua"
  test_params: {
    1: {"threads": 5, "connections": 10, "rps": 10, "duration": "30s"},
    2: {"threads": 10, "connections": 20, "rps": 20, "duration": "2m"}
    }

```

This file is used to run a performance benchmark test with two different loads generated one after the other as shown above.

After the creation of the YML file, run the following command to perform the load test:

```bash
python3 rest_api_wrapper.py -f "<FILE NAME>.py>"
```
### Websocket Performance Testing

To perform websocket testing, first you would need a json file with test configuration details. Here is an example:

```json
{
    "task_sequence": [
        {
            "taskType": "asr",
            "config": {
                "language": {
                    "sourceLanguage": "hi"
                },
                "samplingRate": 16000,
                "audioFormat": "wav",
                "encoding": "base64"
            }
        },
        {
            "taskType": "translation",
            "config": {
                "language": {
                    "sourceLanguage": "hi",
                    "targetLanguage": "en"
                }
            }
        }
    ],
    "socket_url": "wss://<DOMAIN>",
    "input_filepath": "path/to/input/audio/file"
}
```

To run the actual performance test, we make use of the locust software to generate the load and get the results. To do so, run the following command:

```bash
locust -f socket_api_wrapper.py --users <NUMBER OF USERS> --spawn-rate <DESIRED SPAWN RATE> -H <URL> -c <PATH TO JSON FILE>
```

## Contributing
* Clone the Project
* Create your Feature Branch (git checkout -b feature/AmazingFeature)
* Commit your Changes (git commit -m 'Add some AmazingFeature')
* Push to the Branch (git push origin feature/AmazingFeature)
* Open a Pull Request
