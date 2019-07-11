#%%
import torch
from torchtext import data
from torchtext import datasets
import random


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

#%%
