# Dhruva Evaluation Suite

Welcome to Dhruva Evaluation Suite! This tool can be used to perform functional testing and performance testing of Dhruva.

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
  url: "https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat%2Fconformer-hi-gpu--t4"

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
  url: "https://api.dhruva.co/services/inference/translation?serviceId=ai4bharat/indictrans-v2-all-gpu--t4"

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
### Transilteraton

```yml
task:
  name: "evaluation"
  type: "transliteration"
  metric: "cer"

model:
  type: "REST"
  url: "https://api.dhruva.ai4bharat.org/services/inference/transliteration?serviceId=ai4bharat%2Findicxlit--cpu-fsv2"

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

To perform the load test, use the load_test_helper.py file to define all the cases and run the following command:

```bash
python3 load_test_helper.py --task "<TASK NAME>" --lua_file "<PATH TO GENERATED LUA FILE>" --url "<URL TO THE API ENDPOINT>" --result_folder_name "<PATH TO DIRECTORY>"
```
### Websocket Performance Testing

To perform websocket testing, you can make use of the locust software to generate the load and get the results. Run the following command:
```bash
locust -f locust-socket-test.py --users <NUMBER OF USERS> --spawn-rate <DESIRED SPAWN RATE> -H http://api.dhruva.ai4bharat.org
```