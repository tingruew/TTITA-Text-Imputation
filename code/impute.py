import pandas as pd
import torchtext
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
import gzip, json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rotary_embedding_torch import RotaryEmbedding


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def convert_to_tensor(inputs, key, element):
    if isinstance(element, list):
        return torch.tensor(element, dtype=torch.float32)
    elif inputs[key] == 'cat':
        return torch.tensor([element], dtype=torch.int)
    else:
        return torch.tensor([element], dtype=torch.float32)
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        See https://github.com/bzhangGo/rmsnorm
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = torch.nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    
class FeedForward(torch.nn.Module):
    def __init__(self, model_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(model_dim, ff_dim)
        self.fc2 = torch.nn.Linear(ff_dim, model_dim)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        return self.fc2(self.silu(self.fc1(x)))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, model_dim, heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % heads == 0, "model dimension must be divisible by the number of heads"
        
        self.model_dim = model_dim
        self.heads = heads
        self.k = model_dim // heads
        
        self.q_linear = torch.nn.Linear(model_dim, model_dim)
        self.k_linear = torch.nn.Linear(model_dim, model_dim)
        self.v_linear = torch.nn.Linear(model_dim, model_dim)
        self.all = torch.nn.Linear(model_dim, model_dim)

        self.rotary = RotaryEmbedding(dim=32)
        
    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, V)
        return output
        
    def separate_heads(self, x):
        seq_length, batch_size, _ = x.size()
        return x.transpose(0, 1).view(batch_size, seq_length, self.heads, self.k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim).transpose(0,1)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.separate_heads(self.q_linear(Q))
        K = self.separate_heads(self.k_linear(K))
        V = self.separate_heads(self.v_linear(V))
    
        Q = self.rotary.rotate_queries_or_keys(Q)
        K = self.rotary.rotate_queries_or_keys(K)

        output = self.attention(Q, K, V, mask)
        output = self.all(self.combine_heads(output))

        return output
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, model_dim, heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, heads)
        self.cross_attention = MultiHeadAttention(model_dim, heads)
        self.feed_forward = FeedForward(model_dim, ff_dim)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.norm3 = RMSNorm(model_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask):
        x_norm = self.norm1(x)
        output = self.self_attention(x_norm, x_norm, x_norm, tgt_mask)
        x = self.dropout(output) + x

        x_norm = self.norm2(x)
        output = self.cross_attention(x_norm, enc_output, enc_output)
        x = self.dropout(output) + x

        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = self.dropout(ff_output) + x
        return x

class Transformer(torch.nn.Module):
    def __init__(self, inputs, output, num_size, cat_size, text_size, 
                    embed_dim, heads, vocab_size, maps):
        super(Transformer, self).__init__()

        self.inputs = inputs
        self.output = output
        self.module = torch.nn.ModuleList([])
        embed = 0
        for col in self.inputs:
            if self.inputs[col] == "int":
                self.module = self.module.append(torch.nn.Linear(1, num_size))
                embed += num_size
            elif self.inputs[col] == "cat":
                self.module = self.module.append(torch.nn.ModuleList([torch.nn.Embedding(len(maps[col]), cat_size),
                                                                            torch.nn.Linear(cat_size, cat_size),
                                                                            torch.nn.Flatten(1)]))
                embed += cat_size
            elif self.inputs[col] == "text":
                self.module = self.module.append(torch.nn.Identity())
                embed += text_size

        self.decoder = torch.nn.ModuleList([DecoderLayer(embed, heads, embed_dim, 0.1) for _ in range(6)])

        self.dropout = torch.nn.Dropout(0.1)
        self.dropout_last = torch.nn.Dropout(0.1)
        self.embedding = torch.nn.Embedding(vocab_size, embed)
        self.fc = torch.nn.Linear(embed, vocab_size)
        self.embed = embed

    def generate_mask(self, tgt):
        tgt = tgt.transpose(0,1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=DEVICE), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def forward(self, data_dict, infer=False):
        
        input_tokens = data_dict.pop(self.output+'_input')
        x_list = []
        
        tgt_mask = self.generate_mask(input_tokens)

        for layer, xi in zip(self.module, data_dict):
            if type(layer) == torch.nn.modules.container.ModuleList:
                out = data_dict[xi]
                for module in layer:
                    out = module(out)
            else: 
                out = layer(data_dict[xi])
            
            x_list.append(out)

        if not infer:
            x = torch.cat(x_list, dim=1).view(-1, self.embed).unsqueeze(0)
        else:
            x = torch.cat([x.view(-1) for x in x_list]).view(1, self.embed).unsqueeze(1)
            
        embeddings = self.embedding(input_tokens.long())
        embeddings = self.dropout(embeddings)
        
        for dec_layer in self.decoder:
            embeddings = dec_layer(embeddings, x, tgt_mask)
            
        embeddings = self.dropout_last(embeddings)
        output = self.fc(embeddings)
        return output
    
class Imputer:
    def __init__(self,
                 input_columns,
                 output_column,
                 numerical_size=100, 
                 categorical_size=10, 
                 textual_size=2**7, 
                 batch_size=128,
                 embed_dim=2048,
                 heads=2,
                 vocab_size=20000):

        self.input = dict(sorted(input_columns.items()))
        self.output = output_column
        self.num_size = numerical_size
        self.cat_size = categorical_size
        self.text_size = textual_size
        self.embed_dim = embed_dim
        self.heads = heads
        self._vocab_size = vocab_size
        self.batch_size = batch_size
        self._scalers = {}
        self._means = {}
        self._maps = {}
        self._vectorizers = {}
        self._tokenizer = None
        self._output_vectorizer = None
        self._max = 1000
        self._data = None


    def _convert(self, value, table):
        try:
            return table[value]
        except:
            return 0
    
    def _generator(self, strings):
        for string in strings:
            yield self._tokenizer(string)

    def _calibrate(self, df):

        print("Calibrating...")
        self._data = df
        self._train = pd.DataFrame()
        self._train[self.output] = self._data[self.output].fillna('')
        self._tokenizer = torchtext.data.get_tokenizer("basic_english")
        special_symbols = ['<unk>', '<pad>', '[start]', '[end]']

        v = torchtext.vocab.build_vocab_from_iterator(self._generator(self._train[self.output]),
                                                      min_freq=1,
                                                      specials=special_symbols,
                                                      special_first=True)
        v.set_default_index(UNK_IDX)
        self._output_vectorizer = v
        self._vocab_size = len(v)
        
        for col in self.input:

            if self.input[col] == 'text':
                vectorizer = HashingVectorizer(n_features=self.text_size,ngram_range=(1,5),analyzer='char',dtype=np.float32)
                self._train[col] = self._data[col].fillna('')
                vectorizer.fit(self._train[col])
                self._vectorizers[col] = vectorizer
                
            elif self.input[col] =='int':
                self._means[col] = self._data[col].mean()
                numbers = self._data[col].astype(float).fillna(self._means[col])
                scaler = StandardScaler()
                self._train[col] = scaler.fit_transform(numbers.values.reshape(-1, 1))
                self._scalers[col] = scaler

            elif self.input[col] == 'cat':
                table = self._data[col].value_counts().reset_index()[col].to_dict()
                table = {y: x+1 for x, y in table.items()}
                table[np.nan] = 0
                self._maps[col] = table
                self._train[col] = self._data[col].apply(lambda x: self._convert(x, table))

            else:
                raise TypeError("Types of column specified is not one of ['numeric', 'categorical', 'textual']")
    
    def _text_transform(self, text):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(self._output_vectorizer(self._tokenizer(text))),
                          torch.tensor([EOS_IDX])))
        
    def _custom_collate(self, original_batch):
        
        batch = []
        for sample in original_batch:
            batch.append(self._text_transform(sample.pop(self.output).strip()))

        padded = torch.nn.utils.rnn.pad_sequence(batch, padding_value=PAD_IDX)
        
        data_dict = {}
        data_dict[self.output] = padded[1:,:]
        data_dict[self.output+'_input'] = padded[:-1,:]

        for dictionary in original_batch:
            for key, value in dictionary.items():
                value_tensor = convert_to_tensor(self.input, key, value).unsqueeze(0)
                
                if key not in data_dict:
                    data_dict[key] = value_tensor
                else:
                    data_dict[key] = torch.cat((data_dict[key], value_tensor), dim=0)
        return data_dict

    def fit(self,
            train_df: pd.DataFrame,
            validation_split=0.1,
            learning_rate=0.0005,
            epochs=10):

        self._calibrate(train_df)
        
        dataset = Table(self._train, self.input, self.output, self._vectorizers)

        train, val = torch.utils.data.random_split(dataset, [1-validation_split, validation_split], generator=torch.Generator().manual_seed(42))
        train = DataLoader(train, batch_size=self.batch_size, shuffle=True, collate_fn=self._custom_collate)
        val = DataLoader(val, batch_size=self.batch_size, collate_fn=self._custom_collate)
        
        self.train(train, val, learning_rate, epochs)

    def train(self, train, val, learning_rate, epochs):
            
        # Instantiate the model
        model = Transformer(self.input, self.output, self.num_size, self.cat_size, self.text_size, 
                   self.embed_dim, self.heads, self._vocab_size, self._maps)
        
        model.to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_vloss = 1e6
        print("Training...")
        for _ in range(epochs):
            print("EPOCH", _)
            model.train()
            losses = 0
            for i, data_dict in enumerate(train):
                
                target = data_dict.pop(self.output).type(torch.LongTensor)
                target = target.to(DEVICE)
                for key, v in data_dict.items():
                    data_dict[key] = data_dict[key].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(data_dict)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))
                losses += (loss.item())
                loss.backward()
                optimizer.step()

            losses = losses/len(train.dataset)
            print("Training Loss:", losses)

            model.eval()
            losses = 0
            with torch.no_grad():
                for i, data_dict in enumerate(val):
                
                    target = data_dict.pop(self.output).type(torch.LongTensor)
                    target = target.to(DEVICE)
                    for key, v in data_dict.items():
                        data_dict[key] = data_dict[key].to(DEVICE)

                    outputs = model(data_dict)
                    loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))
                    losses += (loss.item())
                losses = losses/len(val.dataset)
                print("Validation Loss:", losses)
                if losses < best_vloss:
                    torch.save(model, 'model.pt')
                    best_vloss = losses
    
    def predict(self, df):
        
        model = torch.load('model.pt')
        model.eval()
        predictions = []

        # Process other inputs
        df.reset_index(drop=True, inplace=True)
        new_df = pd.DataFrame()

        for col in self.input:
            if self.input[col] == 'text':
                new_df[col] = self._vectorizers[col].transform(df[col].fillna("")).todense().tolist()
            elif self.input[col] == 'cat':
                new_df[col] = df[col].apply(lambda x: self._convert(x, self._maps[col]))
            else:
                numbers = df[col].astype(float).fillna(self._means[col])
                new_df[col] = self._scalers[col].transform(numbers.values.reshape(-1, 1)).flatten()
                
        with torch.no_grad():  
            for idx in tqdm(range(new_df.shape[0])):

                row = new_df.iloc[idx].to_dict()
                row = {key: convert_to_tensor(self.input, key, value) for key, value in row.items()}

                for key, v in row.items():
                    row[key] = row[key].to(DEVICE)

                y_input = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
                
                row[self.output+'_input'] = y_input
                
                for _ in range(self._max):       
                    
                    pred = model(row, infer=True).squeeze(1)
                    next_word = torch.argmax(pred, dim=1)[_]
                    next_word = next_word.item()
                    y_input = torch.cat([y_input,torch.ones(1, 1).fill_(next_word).to(DEVICE)], dim=0)
                    
                    row[self.output+'_input'] = y_input

                    if next_word == EOS_IDX:
                        break

                y_input = y_input.flatten()
                y_input = " ".join(self._output_vectorizer.lookup_tokens(list(y_input.cpu().numpy()))).replace("[start]", "").replace("[end]", "")
                predictions.append(y_input)
            
        return predictions

class Table(Dataset):
    
    def __init__(self, df, input_columns, output_column, vectorizers):
        
        self.data = df.reset_index(drop=True)
        self._input = input_columns
        self._output = output_column
        self.columns = pd.DataFrame()

        for col in input_columns:
            if input_columns[col] == 'text':
                self.columns[col] = vectorizers[col].transform(self.data[col]).todense().tolist()
            else:
                self.columns[col] = self.data[col]

        self.columns[output_column] = self.data[output_column]
        self.columns.reset_index(drop=True, inplace=True)

    def __len__(self):
        
        return self.columns.shape[0]

    def __getitem__(self, idx):
            
        return self.columns.iloc[idx].to_dict()
