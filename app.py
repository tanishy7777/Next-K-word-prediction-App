import os
import time

import streamlit as st

st.write("Next Word Prediction")
st.write("Choose the model configuration")
cont_size = st.radio(
    'Context Size',
    ("5", "10"))
emb_dim = st.radio(
    'Embedding Dim',
    ("64", "128"))
activation = st.radio(
    'Activation Function',
    ("sine", "relu"))

start_text = st.text_input("Enter Seed text")
start = st.button("Start")

if start and start_text:
    print("started execution")



    st.write(f"You are in {cont_size} {emb_dim} {activation}")
    # name = fr'next_word_model_emb{emb_dim}_cont{cont_size}_{activation}'
    name = fr'next_word_model_emb64_cont5_relu'

    model_path = weights_path = None

    for i in os.listdir('./Models'):
        if i.startswith(name):
            weights_path = i

    print(weights_path)


    from datasets import load_dataset
    import torch
    from torch.utils.data import Dataset, DataLoader
    import re

    st.write("Downloading Dataset")

    ds = load_dataset("microsoft/orca-math-word-problems-200k")


    def tokenizer(text):
        tokens = [i for i in re.split(r'(\s|[^a-zA-Z])', text) if i and not i.isspace()]
        return tokens


    st.write("Building Vocab")

    def build_vocab(dataset):

        vocab = {'EOQ'}
        count = 0
        for string in dataset['train']['question']:
            if not string:
                continue

            tokens = tokenizer(string.lower())
            vocab.update(tokens)

        vocab = sorted(vocab)
        token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        return token_to_idx, idx_to_token


    token_to_idx, idx_to_token = build_vocab(ds)
    vocab_size = len(token_to_idx)
    print("Vocabulary size:", vocab_size)



    class TokenizedDataset(Dataset):
        def __init__(self, text_data, token_to_idx, context_size=5):
            self.samples = []
            self.context_size = context_size
            self.token_to_idx = token_to_idx

            counter = 0

            for text in text_data:
                if not text:
                    continue

                counter += 1
                if counter % 10000 == 0:
                    print(f'{counter*100/len(text_data)}% processed')
                    st.write(f'{counter*100/len(text_data)}% processed')

                tokens = tokenizer(text.strip().lower())
                indexed_tokens = [self.token_to_idx.get(token, self.token_to_idx['EOQ']) for token in tokens]
                indexed_tokens = [self.token_to_idx['EOQ']] * self.context_size + indexed_tokens + [self.token_to_idx['EOQ']]

                for i in range(context_size - 1, len(indexed_tokens) - 1):
                    context = indexed_tokens[i - context_size + 1: i + 1]
                    target = indexed_tokens[i + 1]
                    self.samples.append((context, target))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            context, target = self.samples[idx]
            return torch.tensor(context), torch.tensor(target)

    context_size = int(cont_size)
    emb_dim = int(emb_dim)
    activation = activation

    st.write("Making Dataloader")

    train_texts = ds['train']['question']
    tokenized_dataset = TokenizedDataset(train_texts, token_to_idx, context_size=context_size)
    train_loader = DataLoader(tokenized_dataset, batch_size=4096, shuffle=True)



    for context, target in train_loader:
        print("Context:", context)
        print("Target:", target)
        break

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    device = get_default_device()
    print("Device: ",device)

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    import torch
    import torch.nn as nn

    vocab_size = len(token_to_idx)
    num_epochs = 15
    learning_rate = 0.001


    class NextWord(nn.Module):
        def __init__(self, context_size, vocab_size=len(token_to_idx), emb_dim=64, hidden_size=256, activation='sine'):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(context_size * emb_dim, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            self.lin3 = nn.Linear(hidden_size, hidden_size)
            self.lin4 = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(0.3)
            self.activation = activation

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)

            if self.activation == 'sine':
                x = torch.sin(self.lin1(x))
                x = torch.sin(self.lin2(x))
                x = torch.sin(self.lin3(x))
            elif self.activation == 'relu':
                x = nn.functional.relu(self.lin1(x))
                x = nn.functional.relu(self.lin2(x))
                x = nn.functional.relu(self.lin3(x))
            x = self.lin4(x)
            return x

    model = NextWord(context_size, vocab_size, emb_dim, hidden_size=512, activation=activation)
    to_device(model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for param_name, param in model.named_parameters():
        print(param_name, param.shape)


    def reconstruct_text(tokens):
        result = []
        print(tokens)

        for i, token in enumerate(tokens):
            result.append(token)

            if i < len(tokens) - 1:
                if token.isalpha() and tokens[i + 1].isalpha():
                    result.append(" ")
                elif token.isalpha() and tokens[i + 1].isdigit():
                    result.append(" ")
                elif token.isdigit() and tokens[i + 1].isalpha():
                    result.append(" ")
                elif token in [',','.','!','?',':','%','&'] and tokens[i + 1].isalnum():
                    result.append(" ")

        return ''.join(result)

    checkpoint = torch.load('./Models/'+weights_path, weights_only=False, map_location=torch.device('cpu'))
    # model.load_state_dict(torch.load("./Models/next_word_model_emb64_cont5_relu_hidden512.pth"))

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    import torch

    def stream_data(str):
        for word in str.split(" "):
            yield word + " "
            time.sleep(0.09)

    def generate_text(model, start_text, token_to_idx, idx_to_token, context_size=5, max_length=50):
        model.eval()
        tokens = tokenizer(start_text.lower())
        input_indices = [token_to_idx.get(token, token_to_idx['EOQ']) for token in tokens]

        if len(input_indices) < context_size:
            input_indices = [token_to_idx['EOQ']] * (context_size - len(input_indices)) + input_indices

        context = input_indices[-context_size:]
        generated_tokens = tokens


        with torch.no_grad():

            for _ in range(max_length):
                context_tensor = torch.tensor(context, device=device).unsqueeze(0)

                output_logits = model(context_tensor)
                predicted_token_idx = torch.distributions.categorical.Categorical(logits=output_logits).sample().item()

                predicted_token = idx_to_token.get(predicted_token_idx, '<unk>')
                generated_tokens.append(predicted_token)

                context = context[1:] + [predicted_token_idx]

                if predicted_token == 'EOQ':
                    generated_tokens.pop()
                    break

        return reconstruct_text(generated_tokens)

    # start_text = "Ramesh has 5 pencils"
    generated_text = generate_text(model, start_text, token_to_idx, idx_to_token, context_size=int(cont_size), max_length=50)
    print(generated_text)
    st.write(generated_text)
    st.sidebar.write_stream(stream_data(generated_text))


