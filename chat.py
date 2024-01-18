import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class QADataset(Dataset):
    def __init__(self, qa_pairs, vocab):
        self.qa_pairs = qa_pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        question_tokens = [self.vocab[token] for token in qa_pair["question"].split()]
        answer_tokens = [self.vocab[token] for token in qa_pair["answer"].split()]
        return torch.tensor(question_tokens), torch.tensor(answer_tokens)

def collate_fn(batch):
    questions, answers = zip(*batch)
    padded_questions = pad_sequence(questions, batch_first=True, padding_value=0)
    padded_answers = pad_sequence(answers, batch_first=True, padding_value=0)
    return padded_questions, padded_answers

# Define your dummy dataset with question-answer pairs
qa_pairs = [
    {"question": "What is your name?", "answer": "My name is Ayyuce"},
    {"question": "How are you?", "answer": "I am doing well"},
    {"question": "Where do you live?", "answer": "I live in the virtual world"},
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris"},
    {"question": "How old are you?", "answer": "I don't have an age, I'm a machine learning model"},
]

vocab = {word: idx for idx, word in enumerate(set(" ".join([pair["question"] + " " + pair["answer"] for pair in qa_pairs]).split()))}
dataset = QADataset(qa_pairs, vocab)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

ntoken = len(vocab)
ninp = 256
nhead = 8
nhid = 512
nlayers = 6
model = TransformerModel(ntoken, ninp, nhead, nhid, nlayers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        questions, answers = batch
        output = model(questions)

        # Calculate loss for each sequence in the batch
        loss = 0
        for i in range(output.size(1)):
            loss += criterion(output[:, i, :], answers[:, i])

        # Average the loss over the sequences in the batch
        loss /= output.size(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


test_input_question = "What is your name?"
test_input_tokens = [vocab[token] for token in test_input_question.split()]
test_input = torch.tensor(test_input_tokens).unsqueeze(0)
generated_output = model(test_input)

# Get the index of the word with the highest probability at each position
generated_indices = torch.argmax(generated_output, dim=-1).squeeze().tolist()

# Filter out padding tokens (if any)
generated_indices = [idx for idx in generated_indices if idx != 0]

# Reverse the vocab dictionary to map indices to words
idx_to_word = {idx: word for word, idx in vocab.items()}

# Convert indices back to words
generated_answer = " ".join([idx_to_word[idx] for idx in generated_indices])

print("Generated Answer:", generated_answer)
