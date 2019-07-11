#%%
import torch
from torchtext import data
from torchtext import datasets
import random

import torch.nn as nn


# Data Preperation
#%%
SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.FloatType)

#%%
train, test = datasets.IMDB.splits(TEXT, LABEL)
train, validate = train.split(random_state = random.seed(SEED))

#%%
VOCAB_SIZE = 25_000

TEXT.build_vocab(train, max_size = VOCAB_SIZE)
LABEL.build_vocab(train)

#%%
torch.cuda.is_available()

#%%

BATCH_SIZE = 64

device = torch.device('cuda')

#%%
train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train, validate, test),
    batch_size = BATCH_SIZE,
    device = device
)

# build the model
#%% RNN Class
class Network(nn.Module):
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

#%% Instantiate

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = Network(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


#%%
