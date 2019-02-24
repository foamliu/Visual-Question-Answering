import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure models
im_size = 448
hidden_size = 512

# Configure training/optimization
print_freq = 100
batch_size = 64
teacher_forcing_ratio = 0.5
clip = 50.0

PAD_token = 0
EOS_token = 1
SOS_token = 2

train_folder = 'data/train2014'
valid_folder = 'data/val2014'
test_folder = 'data/test2015'

qa_json = 'data/FM-CH-QA.json'
pickle_file = 'data/data.pkl'
