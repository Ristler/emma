import sentencepiece as spm
import numpy as np
from model import model, sp





def _init__(self, model_path: str):
    self.model = self._load_model(model_path)
  
    self.conversation_turns = []
    self.sp = spm.SentencePieceProcessor(model_file=model_path)




def softmax(logits, temperature=1.0):
    logits = np.array(logits) / temperature
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()

def generate_tokens(prompt, num_tokens=150, stop_on_special=True, temperature=1.0, stream=False):
    import re
    input_ids = sp.encode(prompt, out_type=int)
    generated_ids = []
    pad_id = sp.pad_id() if hasattr(sp, 'pad_id') else 0
    unk_id = sp.unk_id() if hasattr(sp, 'unk_id') else 1
    eos_id = sp.eos_id() if hasattr(sp, 'eos_id') else -1

    def token_stream():
        for i in range(num_tokens):
            if len(input_ids) < 512:
                padded = input_ids + [pad_id] * (512 - len(input_ids))
            else:
                padded = input_ids[-512:]
            input_array = np.array([padded])
            pred = model.predict(input_array)
            idx = len(input_ids)-1 if len(input_ids) <= 512 else 511
            logits = pred[0, idx]
            probs = softmax(logits, temperature)
            next_token_id = int(np.random.choice(len(probs), p=probs))
            if stop_on_special and next_token_id in [pad_id, unk_id, eos_id]:
                print(f"Stopping generation at step {i+1} due to special token: {next_token_id}")
                break
            input_ids.append(next_token_id)
            generated_ids.append(next_token_id)
            # Decode the full generated sequence so far
            result = sp.decode(generated_ids)
            # Remove leading whitespace and punctuation only on first yield
            if i == 0:
                result = re.sub(r'^[\s\.,;:!?-]+', '', result)
            yield result

    if stream:
        return token_stream()
    else:
        for _ in token_stream():
            pass
        # Decode only the generated part
        result = sp.decode(generated_ids)
        result = re.sub(r'^[\s\.,;:!?-]+', '', result)
        return result.strip()

