import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im_size = 448

# Configure training/optimization
batch_size = 64
print_freq = 100
pickle_file = 'data/data.pkl'

# Configure models
hidden_size = 512

train_folder = 'data/train2014'
valid_folder = 'data/val2014'
test_folder = 'data/test2015'
qa_json = 'data/FM-CH-QA.json'
