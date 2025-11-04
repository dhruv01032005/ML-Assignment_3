import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from collections import Counter
import requests

class NextWord(nn.Module):
    def __init__(self, context_size, vocab_size, emb_dim=32, hidden_size=1024, activation='relu'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(context_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        
        if activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.act(self.lin1(x))
        logits = self.lin2(x)
        return logits

# Streamlit UI setup
st.set_page_config(page_title="Next Word Predictor", layout="wide")
st.title("Next Word Prediction App")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Sidebar
st.sidebar.header("Model Configuration")
embedding_size = st.sidebar.selectbox("Embedding Size", [32, 64])
context_size = st.sidebar.selectbox("Context Length", [5, 7])
activation_fn = st.sidebar.selectbox("Activation Function", ["relu", "tanh"])
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
k_words = st.sidebar.slider("Number of words to predict (k)", 5, 50, 10, 1)
random_seed = st.sidebar.slider("Random Seed",40,50,42,1)

hidden_size = 1024

# Making vocabulary
file = 'https://www.gutenberg.org/files/1661/1661-0.txt'

all_text = ""

response = requests.get(file)
text = response.text

text = text.replace("\n", " ")
text = text.lower()
text = re.sub(r'[^a-zA-Z0-9 \.]', '', text)

all_text += " " + text
print(all_text[:10000])

sentences = all_text.split('.')
sentences = [s.strip() for s in sentences if s.strip() != ""]
sentences = [s.split() for s in sentences]

all_words = []
for sentence in sentences:
    for word in sentence:
        all_words.append(word)

word_counts = Counter(all_words)

vocab = list(word_counts.keys())
vocab_size = len(vocab)

stoi = {}
itos = {}

stoi['.'] = 0
itos[0] = '.'

for i, word in enumerate(vocab):
    stoi[word] = i + 1
    itos[i + 1] = word

stoi['<UNK>'] = len(stoi)
itos[len(itos)] = '<UNK>'

# Load model and weights
model_name = f"emb_{embedding_size}_con_{context_size}_act_{activation_fn}.pth"
model_path = os.path.join(".", model_name)

loaded_model = NextWord(
    context_size=context_size,
    vocab_size=len(stoi),
    emb_dim=embedding_size,
    hidden_size=hidden_size,
    activation=activation_fn
).to(device)

state_dict = torch.load(model_path, map_location=device)
loaded_model.load_state_dict(state_dict)
loaded_model.eval()

# Prediction function
def predict_next(words, num_words=10, temperature=1.0, context_size=5, random_seed = 42):
    torch.manual_seed(random_seed)
    context = [0] * context_size
    generated = []
    for w in words:
        context.append(stoi.get(w, stoi["<UNK>"]))

    for _ in range(num_words):
        x = torch.tensor(context[-context_size:]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = loaded_model(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        context.append(next_idx)
        generated.append(next_idx)

    ans = words + [itos.get(i, "<UNK>") for i in generated]
    return " ".join(ans)

# User Input
st.subheader("Enter your text prompt")
user_text = st.text_input("Input Text", "sherlock holmes")

if st.button("Predict Next Words"):
    user_text = user_text.lower()
    words = user_text.strip().split()
    output = predict_next(words, num_words=k_words, temperature=temperature, context_size=context_size, random_seed=random_seed)
    st.markdown("### Generated Text:")
    st.success(output)

st.markdown("---")
st.caption("Change model hyperparameters from the sidebar to explore different saved models.")