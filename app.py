from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn

class POSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=128, hidden_dim=256):
        super(POSTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x, lengths):
        embeds = self.embedding(x)
        packed_input = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(lstm_out)

# Load the model
def load_model(filename="pos_tagger.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define vocabulary mappings (Must match training script)
word2idx = {
    "A": 1, "Birds": 2, "He": 3, "John": 4, "She": 5, "The": 6, "They": 7, "We": 8, 
    "are": 9, "apple": 10, "at": 11, "barks": 12, "book": 13, "cat": 14, "dog": 15, 
    "eats": 16, "fast": 17, "fly": 18, "football": 19, "high": 20, "in": 21, 
    "jumps": 22, "mat": 23, "moon": 24, "movies": 25, "on": 26, "playing": 27, 
    "reads": 28, "rises": 29, "runs": 30, "sky": 31, "sits": 32, "sun": 33, 
    "together": 34, "watch": 35, "a": 36, "an": 37, "and": 38, "the": 39
}

tag2idx = {"ADP": 0, "ADV": 1, "CONJ": 2, "DET": 3, "NOUN": 4, "PRON": 5, "VERB": 6}
idx2tag = {i: t for t, i in tag2idx.items()}

# Define maximum sequence length (Must match training script)
max_len = 6

def predict(sentence, model, word2idx, idx2tag, max_len):
    model.eval()
    sentence_idx = [word2idx.get(word, 0) for word in sentence]  # Convert words to indices
    sentence_tensor = torch.tensor([sentence_idx + [0] * (max_len - len(sentence))], dtype=torch.long)
    sentence_length = torch.tensor([len(sentence)], dtype=torch.long)

    with torch.no_grad():
        output = model(sentence_tensor, sentence_length)
        prediction = torch.argmax(output, dim=-1)

    predicted_tags = [idx2tag[idx.item()] for idx in prediction[0][:len(sentence)]]
    return predicted_tags

@app.route("/tag", methods=["POST"])
def tag_sentence():
    try:
        data = request.get_json()
        sentence = data.get("sentence", [])
        
        if not isinstance(sentence, list) or not all(isinstance(word, str) for word in sentence):
            return jsonify({"error": "Invalid input format. Must be a list of words."}), 400

        predicted_tags = predict(sentence,model,word2idx,idx2tag,max_len)
        return jsonify({"sentence": sentence, "tags": predicted_tags})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
import os
port = int(os.environ.get("PORT", 10000))  # Render sets PORT dynamically
app.run(host="0.0.0.0", port=port)
