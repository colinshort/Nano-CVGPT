import torch
from model import CVGPT

# hyperparameters
batch_size = 32 # number of independent sequences processed in paralled
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embed = 32 # embedding size
n_head = 4 # number of heads in multi-head attention
n_layer = 4 # number of transformer blocks
dropout = 0.2
# --------------

if torch.cuda.is_available():
  device = 'cuda'
  print("Using GPU: " + torch.cuda.get_device_name(0))
else:
  device = 'cpu'
  print("Using CPU with", torch.get_num_threads(), "threads.")

# get our data
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get sorted list of all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from characters to integers
str_to_int = { char:itgr for itgr, char in enumerate(chars) }
int_to_str = { itgr:char for itgr, char in enumerate(chars) }

# encoder: take a string, output a list of integers
def encode(str):
  encoding = []
  for char in str:
    encoding.append(str_to_int[char])
  return encoding

# decoder: take a list of integers, output a string
def decode(ints):
  decoding = ""
  for i in ints:
    decoding += int_to_str[i]
  return decoding

# encode the full text and wrap it in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train/test split on dataset
train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# generate a small batch of inputs x and targets y
def get_batch(data_split):
  if data_split == "train":
    data = train_data
  elif data_split == "test":
    data = test_data

  ix = torch.randint(len(data) - block_size, (batch_size,)) # generate batch_size random starting points in the data to grab our batches from. In this case, for 4 random starting points
  x = torch.stack([data[i:i + block_size] for i in ix]) # get batches for each starting point and stack them on top of each other. Each row of x is a batch
  y = torch.stack([data[i + 1:i + block_size +1] for i in ix]) # same as above, offset by one to get targets
  x, y = x.to(device), y.to(device) # send data to the gpu
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "test"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

model = CVGPT(vocab_size, n_embed, n_head, n_layer, block_size, dropout, device)
m = model.to(device) # send model to gpu
# print number of parameters in the model
print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 'M parameters')

# create PyTorch Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # backprop algorithm

# training loop
for step in range(max_iters):
  # evaluate the loss on the training and testing steps on a given interval
  if step % eval_interval == 0:
    losses = estimate_loss()
    print(f"Step {step} out of {max_iters}: Train loss {losses['train']:.4f}, Test loss {losses['test']:.4f}")

  # get batch
  x_batch, y_batch = get_batch('train')

  # forward pass
  logits, loss = model(x_batch, y_batch)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))