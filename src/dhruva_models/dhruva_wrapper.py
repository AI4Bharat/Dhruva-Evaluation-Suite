import json
import base64
import logging
import requests


ULCAFormats = {
    "automatic-speech-recognition": {
        # "headers": headers,
        "payload": {
            "audio": [
                {
                    "audioContent": "string",
                    "audioUri": "string"
                }
            ],
            "config": {
                "language": {
                    "sourceLanguage": "string"
                },
                "audioFormat": "wav",
                "encoding": "base64",
                "samplingRate": 16000,
                "postProcessors": [None]
            }
        }
    },
    "NMT": {
        # "headers": headers,
        "payload": { "text": "", "source_language": "", "target_language": "" }
    }
}


class DhruvaModel():
    def __init__(self, task: str, lang: str, url: str, **kwargs):
        self.task = task
        self.url = url
        self.payload = ULCAFormats.get(self.task)["payload"]
        self.payload["config"]["language"]["sourceLanguage"] = lang

    def _encode(self, raw_input):
        return base64.b64encode(raw_input).decode("utf-8")

    def __call__(self, batch_audio, **kwargs):        
        self.payload["audio"] = [{"audioContent": self._encode(audio)} for audio in batch_audio]
        results = requests.post(self.url, payload=json.dumps(self.payload)).json()
        return [{"generated_transcript": p["source"]} for p in results["output"]]
