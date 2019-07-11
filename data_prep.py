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

# include the lengths to allow for packed padded sequences
TEXT = data.Field(tokenize = 'spacy', include_lengths= True)
LABEL = data.LabelField(dtype = torch.float)

#%% Load the dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, validate_data = train_data.split(random_state = random.seed(SEED))

#%%
VOCAB_SIZE = 25_000

# include pre trained vectors
TEXT.build_vocab(
    train_data, 
    max_size = VOCAB_SIZE,
    vectors = "glove.6B.100d",
    unk_init = torch.Tensor.normal_
    )

LABEL.build_vocab(train_data)

#%% build the ierators
BATCH_SIZE = 64
# device = torch.device('cuda')
device = torch.device('cpu')

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train_data, validate_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)