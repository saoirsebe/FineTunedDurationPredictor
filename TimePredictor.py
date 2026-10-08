import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class TimePredictor:
    def __init__(self, model_dir="onnx_out"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(f"{model_dir}/model.onnx")
        stats = json.load(open(f"{model_dir}/target_stats.json"))
        self.mean, self.std = stats["mean"], stats["std"]
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, sentence: str) -> float:
        enc = self.tokenizer(sentence, truncation=True, max_length=128, return_tensors="np")
        inputs = {k: v for k, v in enc.items() if k in self.input_names}
        logits = self.session.run(None, inputs)[0]
        z = float(logits[0][0])
        return float(np.exp(z * self.std + self.mean))

predictor = TimePredictor()
print(predictor.predict("rearrange closet"))