import torch
from torch.utils.data import Dataset, DataLoader
from openspeech.data.sampler import SmartBatchingSampler, RandomSampler
import random


class TextDataset(Dataset):
    def __init__(self, transcripts: list, tokenizer, mode="BERT"):
        super(TextDataset, self).__init__()
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.mode = mode
        self.mask_id = 2000
        
    def maskInputs(self, inputs, mask=True):
        # 문장의 변형된 위치를 저장할 리스트.
        labels = [0 for _ in range(len(inputs))]

        for i, token in enumerate(inputs[1:-1]):
            prob = random.random()
            # 전체 문장의 15% 중에서
            if prob < 0.15:
                labels[i] = inputs[i]
                
                prob /= 0.15
                # 80%는 <Mask> 토큰을,
                if prob < 0.8:
                    inputs[i] = self.mask_id
                # 10%는 다른 토큰으로 바꾸고,
                elif prob < 0.9:
                    inputs[i] = random.randrange(5, len(self.tokenizer)-1)
                # 10%는 바꾸지 않는다.
                else:
                    pass

        return inputs, labels
        
    def __getitem__(self, idx):
        transcript = self.tokenizer.encode(self.transcripts[idx])
        inputs = [self.tokenizer.sos_token] + transcript            
        if self.mode == "GPT":
            outputs = transcript + [self.tokenizer.eos_token]
        elif self.mode == "BERT":
            inputs.append(self.tokenizer.eos_token)
            inputs, outputs = self.maskInputs(inputs, mask=True)
            
        return torch.LongTensor(inputs), torch.LongTensor(outputs)
        
    def __len__(self):
        return len(self.transcripts)
    
    
def _collate_fn(batch, pad_id: int= 0):
    def seq_length_(p):
        return len(p[0])
    
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    seq_lengths = [len(s[0]) for s in batch]
    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = max_seq_sample.size(0)
    
    batch_size = len(batch)
    inputs = torch.zeros(batch_size, max_seq_size).fill_(pad_id).long()
    targets = torch.zeros(batch_size, max_seq_size).fill_(pad_id).long()
    
    for x in range(batch_size):
        sample = batch[x]
        input_var = sample[0]
        target = sample[1]
        inputs[x].narrow(0, 0, len(input_var)).copy_(torch.LongTensor(input_var))
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
        
    seq_lengths = torch.IntTensor(seq_lengths)
    
    return inputs, seq_lengths, targets
    
class TextDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, num_workers: int, **kwargs,):
        super(TextDataLoader, self).__init__(dataset=dataset, num_workers=num_workers, **kwargs,)
        self.collate_fn = _collate_fn
    
class DataModule:
    def __init__(self, configs, tokenizer):
        self.configs = configs
        self.tokenizer = tokenizer
        self.train_transcripts, self.valid_transcripts = self.parse_manifest()
        self.train_dataset = TextDataset(
            transcripts=self.train_transcripts, 
            tokenizer=tokenizer, 
            mode=self.configs.model_name,
        )
        self.valid_dataset = TextDataset(
            transcripts=self.valid_transcripts, 
            tokenizer=tokenizer,
            mode=self.configs.model_name,
        )
        
        if self.configs.sample_mode == "smart":
            sampler = SmartBatchingSampler
        elif self.configs.sample_mode == "random":
            sampler = RandomSampler
        
        self.train_sampler = sampler(data_source=self.train_dataset, batch_size=self.configs.batch_size)
        self.valid_sampler = sampler(data_source=self.valid_dataset, batch_size=self.configs.batch_size)
        
    def parse_manifest(self):
        transcripts = []
        with open(self.configs.manifest_file, "r") as f:
            for line in f.readlines():
                transcripts.append(line.replace("\n", ""))
        random.seed(1)
        random.shuffle(transcripts)
        train_num = int(len(transcripts) * self.configs.train_ratio)
        train_transcripts = transcripts[:train_num]
        valid_transcripts = transcripts[train_num:]
        return train_transcripts, valid_transcripts
    
    def get_dl(self, mode):
        if mode == "train":
            return TextDataLoader(
                dataset=self.train_dataset,
                num_workers=self.configs.num_workers,
                batch_sampler=self.train_sampler,
            )
        
        elif mode == "valid":
            return TextDataLoader(
                dataset=self.valid_dataset, 
                num_workers=self.configs.num_workers,
                batch_sampler=self.valid_sampler,
            )
        