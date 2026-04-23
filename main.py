from fastapi import FastAPI
import json
import torch
from torch import nn
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

MODEL_PATH = './model_jokes_weights.pth'
VOCAB_INFO_PATH = './vocab_info.json'

class AnekdotesRNN(nn.Module):
    def __init__(self, num_tokens: int, pad_idx: int, emb_size: int = 256, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size, padding_idx = pad_idx)

        self.rnn = nn.LSTM(
            input_size = emb_size,
            hidden_size = hidden_size,
            num_layers = 3,
            dropout = 0.3,
            batch_first = True
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_tokens)
        )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.output(x)
        return x

def get_vocab_info(vocab_info_path: str):
    data = {}
    with open(vocab_info_path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    sos_idx = data['sos']
    eos_idx = data['eos']
    pad_idx = data['pad']
    num_tokens = data['vocab_len']
    vocab_idxs = data['vocab_idxs']

    return sos_idx, eos_idx, pad_idx, num_tokens, vocab_idxs

def load_model(model_path: str, pad_idx: int, num_tokens: int, device: torch.device):
    model = AnekdotesRNN(num_tokens, pad_idx)
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict)
    return model

def pick_by_distributions(logits):
    return torch.distributions.Categorical(logits = logits).sample()

def encode(text: str):
    char_idxs = []
    for char in text:
        if char in vocab_idxs:
            char_idxs.append(vocab_idxs[char])
        else:
            char_idxs.append(pad_idx)
    return [sos_idx] + char_idxs + [eos_idx]

def idx2char(tok_idx):
    return idx_to_char[tok_idx]

@torch.inference_mode()
def get_continuation(model, prefix: str = '', max_len: int = 1000, count: int = 10, temperature: float = 0.3):
    x = torch.LongTensor([encode(prefix)[:-1]] * count).to(device)
    model.eval()

    finished = torch.zeros(count, dtype = torch.bool, device = device)
    results = [[] for _ in range(count)]

    logits = model(x)[:, -1, :]
    outs = pick_by_distributions(logits / temperature).unsqueeze(1)

    for i in range(max_len):
        x = torch.cat([x, outs], dim = 1)

        for j in range(count):
            if not finished[j]:
                tok = outs[j, 0].item()
                if tok == eos_idx:
                    finished[j] = True
                else:
                    results[j].append(tok)
        if finished.all():
            break
        logits = model(x)[:, -1, :]
        outs = pick_by_distributions(logits / temperature).unsqueeze(1)

    lines = []
    for i, toks in enumerate(results):
        line = prefix + " " + ''.join([idx2char(tok) for tok in toks])
        lines.append(line)
    return lines

def idx_char_map():
    return {v: k for k, v in vocab_idxs.items()}

sos_idx, eos_idx, pad_idx, num_tokens, vocab_idxs = get_vocab_info(VOCAB_INFO_PATH)
idx_to_char = idx_char_map()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

jokes_model = load_model(MODEL_PATH, pad_idx, num_tokens, device)
jokes_model.eval()
app = FastAPI(title = 'Генератор анекдотов')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)
@app.get('/')
async def hello_world():
    return {
        'message': 'Hello, World!'
    }

@app.get('/joke')
async def get_joke(prompt: str):
    return {
        'jokes': get_continuation(jokes_model, prompt)
    }

@app.get('/jokes')
async def get_jokes(prompt:str, count: int = 5, temperature: float = 0.3):
    count = max(1, min(count, 20))
    temperature = max(0.1, min(temperature, 1.0))
    return {
        'jokes': get_continuation(jokes_model, prompt,
                                  count = count, temperature = temperature)
    }

@app.get('/random_joke')
async def random_joke():
    prefixes = ['Приходит как-то', 'Штирлиц', 'Два генерала', 'Жена говорит мужу', 'Доктор']
    prefix = random.choice(prefixes)
    jokes = get_continuation(jokes_model, prefix, count = 1, temperature = 0.3)
    return {'joke': jokes[0]}

@app.get('/ui', response_class = HTMLResponse)
async def ui():
    with open('joke_chat.html', 'r', encoding = 'utf-8') as f:
        return f.read()

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'device': str(device),
        'vocab_size': num_tokens
    }