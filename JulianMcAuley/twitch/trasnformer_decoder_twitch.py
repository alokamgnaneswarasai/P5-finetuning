# %% [markdown]
# ## Load the data

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class SequentialRecommendationDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.max_seq_length = 0
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                user_id = int(parts[0])
                sequence = list(map(int, parts[1:]))
                self.data.append((user_id, sequence))
                self.max_seq_length = max(self.max_seq_length, len(sequence))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id, sequence = self.data[idx]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return user_id, input_seq, target_seq
    
def collate_fn(batch):
    user_ids, input_seqs, target_seqs = zip(*batch)
    input_seqs = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return user_ids, input_seqs, target_seqs
    
dataset = SequentialRecommendationDataset('sequential_recommendation_data.txt')

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)



# %%
# print the first batch
for user_ids, input_seqs, target_seqs in dataloader:
    print(user_ids)
    print(len(input_seqs))
    print(target_seqs)
    break

# %%
# define the rnn model

class RNNModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.embedding(x)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out
    
# Hyperparameters
# input size should be the number of items in the dataset + 1 (for padding) , so calculate it from the .txt file
input_size = 0
with open('sequential_recommendation_data.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        sequence = list(map(int, parts[1:]))
        input_size = max(input_size, max(sequence))
input_size += 1

print(f'vocab size: {input_size}')
hidden_size = 128
num_layers = 1
num_classes = input_size
num_epochs = 5
learning_rate = 0.005

# Initialize the model, loss function, and optimizer and train the model on gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, num_layers, num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(dataloader)
for epoch in range(num_epochs):
    for i, (user_ids, input_seqs, target_seqs) in enumerate(dataloader):
        
        model.train()
        input_seqs, target_seqs = input_seqs.to(device), target_seqs.to(device)
        outputs = model(input_seqs)
        # print(outputs.shape, target_seqs.shape)
        loss = criterion(outputs.view(-1, num_classes), target_seqs.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    if loss.item() < 0.8:
        break
# Save the model checkpoint
torch.save(model.state_dict(), 'rnnmodel.ckpt')

    

# %% [markdown]
# # Inference
# 
# In this section, we will perform inference on our trained model. The goal is to predict the next sequence of items based on a given input item. This is a common scenario in recommendation systems where we want to predict what items a user might interact with next, based on their past interactions.
# 
# The process will work as follows:
# 
# 1. We start by feeding the model an input item.
# 2. The model will generate a prediction for the next item.
# 3. We then take the model's prediction and use it as the new input, repeating the process.
# 4. This is done iteratively, up to 5 times, to generate a sequence of recommended items.
# 
# This method of using the model's own predictions as input for subsequent predictions is known as autoregression.
# 
# Let's see how this works in practice.

# %%
# Now lets do inference where i will give the model one item and it will predict the next sequence upto 5 items(feed the output of the model as input to the model again)

# Load the model checkpoint
model = RNNModel(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('rnnmodel.ckpt'))
model.eval()

# Inference

# Choose a random item from the dataset
import random
item = random.randint(1, input_size-1)
print('Starting item:', item)

# Initialize the input sequence with the chosen item
input_seq = torch.tensor([[item]]).to(device)

# Generate the next 5 items in the sequence
with torch.no_grad():
    for _ in range(5):
        output = model(input_seq)
        _, predicted = torch.max(output[:, -1, :], 1)
        input_seq = torch.cat((input_seq, predicted.unsqueeze(1)), dim=1)
        
print('Generated sequence:', input_seq.squeeze().tolist())

torch.save(model.state_dict(), 'rnnmodel.ckpt')

# %% [markdown]
# ## Now create the transformer decoder architecture and train it 

# %%
# create the transformer deocder model which takes the inout sequence one by one and predicts the next item in the sequence and train the model on gpu use teacher forcing technique

class TransformerDecoder(nn.Module):
    def __init__(self, num_items, embed_size, num_layers, num_heads, hidden_dim):
        super(TransformerDecoder, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embed_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(embed_size, num_items)
        
    def forward(self, input_seqs):
        embeddings = self.item_embedding(input_seqs)
        output = self.transformer_decoder(embeddings, embeddings)
        output = self.fc(output)
        return output
    
# Hyperparameters
num_items = 0
with open('sequential_recommendation_data.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        sequence = list(map(int, parts[1:]))
        num_items = max(num_items, max(sequence))
        
num_items += 1
embed_size = 128
num_layers = 1
num_heads = 2
hidden_dim = 256
num_epochs = 5
learning_rate = 0.005

# Initialize the model, loss function, and optimizer and train the model on gpu
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder(num_items, embed_size, num_layers, num_heads, hidden_dim)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(dataloader)

for epoch in range(num_epochs):
    for i, (user_ids, input_seqs, target_seqs) in enumerate(dataloader):
        
        model.train()
        input_seqs, target_seqs = input_seqs.to(device), target_seqs.to(device)
        outputs = model(input_seqs)
        loss = criterion(outputs.view(-1, num_items), target_seqs.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    if loss.item() < 0.8:
        break
# Save the model checkpoint
torch.save(model.state_dict(), 'transformermodel.ckpt')


# %%
# Now lets do inference where i will give the model one item and it will predict the next sequence upto 5 items(feed the output of the model as input to the model again)

# Load the model checkpoint
model = TransformerDecoder(num_items, embed_size, num_layers, num_heads, hidden_dim).to(device)
model.load_state_dict(torch.load('transformermodel.ckpt'))
model.eval()

# Inference

# Choose a random item from the dataset
import random
item = random.randint(1, num_items-1)
print('Starting item:', item)

# Initialize the input sequence with the chosen item
input_seq = torch.tensor([[item]]).to(device)

# Generate the next 5 items in the sequence
with torch.no_grad():
    for _ in range(10):
        output = model(input_seq)
        _, predicted = torch.max(output[:, -1, :], 1)
        input_seq = torch.cat((input_seq, predicted.unsqueeze(1)), dim=1)
        
print('Generated sequence:', input_seq.squeeze().tolist())

#save the model checkpoint
torch.save(model.state_dict(), 'transformermodel.ckpt')




# %% [markdown]
# # Model Evaluation: NDCG Score Calculation
# 
# In this section, we will evaluate the performance of our trained models - the Transformer model and the Recurrent Neural Network (RNN) model. The evaluation metric we will use is the Normalized Discounted Cumulative Gain (NDCG) score.
# 
# NDCG is a popular metric for evaluating recommendation systems, as it takes into account both the relevance of the recommended items and their ranking. Higher NDCG scores indicate better performance of the model.
# 
# ## Transformer Model Evaluation
# 
# First, we will calculate the NDCG score for the Transformer model. We will use the test dataset to generate recommendations and then compare these recommendations with the actual items the users interacted with.
# 
# ## RNN Model Evaluation
# 
# Next, we will calculate the NDCG score for the RNN model, following the same process as with the Transformer model. 
# 
# By comparing the NDCG scores of the two models, we can determine which model performs better at recommending items that are relevant to the users.

# %% [markdown]
# # NDCG Score Calculation
# 
# Normalized Discounted Cumulative Gain (NDCG) is a popular metric used for evaluating the performance of recommendation systems. It measures the performance of a recommendation system based on the relevance of recommended items and their ranking.
# 
# The formula for DCG (Discounted Cumulative Gain) is:
# 
# DCG@k = Σ (2^relevance[i] - 1) / log2(i + 1) for i in range(1, k+1)
# 
# Where:
# - `relevance[i]` is the relevance of the item at position `i` in the recommended list. In the context of recommendation systems, this could be the rating of the item.
# - `log2(i + 1)` is a discount factor that reduces the contribution of items as their position in the list increases. The `+1` in the log and range functions is to account for the fact that list positions start at 1, not 0.
# 
# The formula for NDCG is:
# 
# NDCG@k = DCG@k / IDCG@k
# 
# Where:
# - `DCG@k` is the DCG score for the recommended list.
# - `IDCG@k` is the DCG score for the ideal list (a perfectly ranked list).
# 
# The NDCG score is a value between 0 and 1. A score of 1 means that the recommended list is perfectly ranked, while a score of 0 means the opposite.

# %%
#load rnn and transformer model

transformer_model = TransformerDecoder(num_items, embed_size, num_layers, num_heads, hidden_dim).to(device)
transformer_model.load_state_dict(torch.load('transformermodel.ckpt'))
transformer_model.eval()

rnn_model = RNNModel(input_size, hidden_size, num_layers, num_classes).to(device)
rnn_model.load_state_dict(torch.load('rnnmodel.ckpt'))
rnn_model.eval()


# %%
#Load both the models and compare the results of ndcg@5 and ndcg@10 for both the models by calculating dcg and idcg and then calculating ndcg

# Load the dataset
dataset = SequentialRecommendationDataset('sequential_recommendation_data.txt')

# calculate NDCG for the RNN model and Transformer model

def calculate_ndcg(model, dataset, device, k):
    dcg = 0
    idcg = 0
    for user_ids, input_seqs, target_seqs in dataloader:
        input_seqs = input_seqs.to(device)
        with torch.no_grad():
            output = model(input_seqs)
            for i in range(len(user_ids)):
                user_id = user_ids[i]
                target_seq = target_seqs[i]
                predicted_seq = output[i].argmax(dim=1)
                dcg += calculate_dcg(target_seq, predicted_seq, k)
                idcg += calculate_dcg(target_seq, target_seq, k)
    return dcg / idcg

def calculate_dcg(target_seq, predicted_seq, k):
    target_seq = target_seq.cpu().numpy()
    predicted_seq = predicted_seq.cpu().numpy()
    dcg = 0
    for i in range(min(k, len(target_seq))):
        item = predicted_seq[i]
        if item in target_seq:
            rank = np.where(target_seq == item)[0][0]
            dcg += 1 / np.log2(rank + 2)
    return dcg

import numpy as np
k = 5
ndcg_rnn = calculate_ndcg(rnn_model, dataset, device, k)
ndcg_transformer = calculate_ndcg(transformer_model, dataset, device, k)

print(f'NDCG@{k} for RNN model: {ndcg_rnn}')
print(f'NDCG@{k} for Transformer model: {ndcg_transformer}')

k = 10
ndcg_rnn = calculate_ndcg(rnn_model, dataset, device, k)
ndcg_transformer = calculate_ndcg(transformer_model, dataset, device, k)

print(f'NDCG@{k} for RNN model: {ndcg_rnn}')
print(f'NDCG@{k} for Transformer model: {ndcg_transformer}')


# %%
print("Hello")

# %%



