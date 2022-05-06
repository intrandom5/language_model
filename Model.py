import torch
import torch.nn as nn
from torch.optim import Adam
import random
from typing import Optional
from Module import LayerNorm, BertBlock, PositionalEncoding, get_attn_pad_mask, get_attn_subsequent_mask

    
class Transformer_LM(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        max_length: int=128, 
        d_model: int=512, 
        d_ff: int=2048, 
        num_heads: int=4, 
        num_layers: int=3, 
        model: str="bert",
        sos_id: int=1,
    ):
        super(Transformer_LM, self).__init__()
        self.model = model.lower()
        self.max_length = max_length
        self.sos_id = sos_id
        self.token_embedding = nn.Embedding(num_classes, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.bert_blocks = nn.ModuleList([
            BertBlock(
                dim = d_model, 
                d_ff = d_ff, 
                num_heads = num_heads, 
                dropout_p = 0.3
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.Tanh(),
            nn.Linear(d_model, num_classes, bias=False),
        )
        
    def get_mask(self, inputs, input_lengths):
        mask = get_attn_pad_mask(
            inputs, input_lengths, inputs.size(1)
        )
        if self.model == "gpt":
            subsequent_mask = get_attn_subsequent_mask(inputs)
            mask = torch.gt((mask + subsequent_mask), 0)
        return mask
        
    
    def forward_step(self, input_var, input_lengths):
        mask = self.get_mask(input_var, input_lengths)
        
        token_embed = self.token_embedding(input_var)
        position_embed = self.position_embedding(input_var.size(1))
        outputs = token_embed + position_embed
        
        for block in self.bert_blocks:
            outputs = block(inputs=outputs, mask=mask)
            
        step_outputs = self.fc(outputs).log_softmax(dim=-1)
        
        return step_outputs
        

    
    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        logits = list()
        
        step_outputs = self.forward_step(input_var=inputs, input_lengths=input_lengths)

        for di in range(step_outputs.size(1)):
            step_output = step_outputs[:, di, :]
            input_var = torch.argmax(step_output, dim=1)
            logits.append(input_var)
        
        return step_outputs, torch.stack(logits, dim=1)
    