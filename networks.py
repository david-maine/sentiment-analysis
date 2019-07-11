import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_d, embed_d, hidden_d, out_d):
        super().__init__()

        self.embedding = nn.Embedding(input_d, embed_d)
        self.rnn = nn.RNN(embed_d, hidden_d)
        self.fc = nn.Linear(hidden_d, out_d)
    
    def forward(self, text):

        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))