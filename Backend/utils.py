from keras.preprocessing.sequence import pad_sequences
from model import sp

MAX_LEN = 512

def preprocess(text):
    print("Preprocessing text:", text)  # Debugging statement
    tokens = sp.encode(text, out_type=int)
    return pad_sequences([tokens], maxlen=MAX_LEN)