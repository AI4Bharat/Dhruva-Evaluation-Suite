# Dhruva Evaluation Suite

## Functional Testing

<br>

```python
# ASR
eval = AccuracyEvaluator(
    {
        "plugins": [
            "MUCSHindiDataset",     # Dataset downloader
            "MUCSPreProcessor",     # Dataset preprocessor. Dumps into a JSONL file
            "ASRBatchE2EModel",     # Model
            "ASRBatchE2EScorer",    # Scoring script for the model
            "WERMetric"             # Metric to evaluate
        ]
    }
)

eval.eval()
```  
  
<br>

---

<br>

## Load Testing - Locust

Locust evaluator plugin calls the model with standard data and simlautes user load


```python
# ASR
eval = AccuracyEvaluator(
    {
        "plugins": [
            "MUCSHindiDataset",     # Dataset downloader
            "MUCSPreProcessor",     # Dataset preprocessor. Dumps into a JSONL file
            "ASRBatchE2EModel",     # Model
            "LocustScorer"          # Scaled scoring script
        ]
    }
)

eval.eval()
```
<br>

---

<br>

## Architecture
<br>

[![](https://mermaid.ink/img/pako:eNp1kMFqwzAMhl_F6ORC8wI5DLZ526WFQHf0Rdhaa5ZYQVYopfTd52XdDkvrk-z_-yyhMwSOBC3sBceDeXc-m3oerUPFQmocH3PPGElWpmkezNNf0gk1nXCgUlhWP969dFaf7bb26q_o722OnN0FlpT3pgRJo95i3uyGw1S0IjfQxQ-z82I7rsb4b8zF8wy_2i2ppFCu1L2GCxjWMJAMmGJd5Plb9qAHGshDW8uI8unB50vlcFLenXKAVmWiNUxjRCWXsO5_gPYD-0KXLxnSgZ8?type=png)](https://mermaid.live/edit#pako:eNp1kMFqwzAMhl_F6ORC8wI5DLZ526WFQHf0Rdhaa5ZYQVYopfTd52XdDkvrk-z_-yyhMwSOBC3sBceDeXc-m3oerUPFQmocH3PPGElWpmkezNNf0gk1nXCgUlhWP969dFaf7bb26q_o722OnN0FlpT3pgRJo95i3uyGw1S0IjfQxQ-z82I7rsb4b8zF8wy_2i2ppFCu1L2GCxjWMJAMmGJd5Plb9qAHGshDW8uI8unB50vlcFLenXKAVmWiNUxjRCWXsO5_gPYD-0KXLxnSgZ8)

<br>

---

<br>

## Contribute Plugins
Plugins are organised as folders and files
- Add in a config for the respective plugin
- Subclass PluginBase and / or task specific base classes and override the get_inputs and invoke methods
