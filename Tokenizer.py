import pandas as pd
import torch


class Tokenizer:
    def __init__(self, label_file: str):
        self.labels_df = pd.read_csv(label_file)
        self.idx2char, self.char2idx = self.make_dictionary(self.labels_df)
        self.pad_token = self.char2idx['<pad>']
        self.sos_token = self.char2idx['<sos>']
        self.eos_token = self.char2idx['<eos>']
        
    def make_dictionary(self, df):
        idx2char, char2idx = dict(), dict()
        ids = df['id']
        labels = df['char']
        for id_, label in zip(ids, labels):
            idx2char[id_] = label
            char2idx[label] = id_
        return idx2char, char2idx
    
    def encode(self, sentence):
        tokens = list()
        for char in sentence:
            if char in self.char2idx:
                token = self.char2idx[char]
                tokens.append(token)
        return tokens
    
    def decode(self, codes):
        if torch.is_tensor(codes):
            codes = list(codes.cpu().detach().squeeze().numpy().astype('int'))
        sentence = ""
        for token in codes:
            char = self.idx2char[token]
            sentence += char
            if char == "<eos>":
                break
        return sentence
    
    def __len__(self):
        return len(self.idx2char)

    