#%%
import torch
from torchtext import data
from torchtext import datasets
import random

import torch.nn as nn

import torch.optim as optim
import time


# Data Preperation
#%%
SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.FloatType)

#%%
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, validate_data = train_data.split(random_state = random.seed(SEED))

#%%
VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size = VOCAB_SIZE)
LABEL.build_vocab(train_data)

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


#%% Train the model
optimiser = optim.SGD(model.parameters(), lr = 1e-3)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)

#%%
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

#%%
def train(model, iterator, optimiser, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimiser.zero_grad()
                
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimiser.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#%%
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#%%
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#%% run the training
N_EPOCHS = 5
LABEL.build_vocab(train,)

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iter, optimiser, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

#%%
