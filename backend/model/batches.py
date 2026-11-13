import torch
from tokenizer import char_tokenizer

#training text
text = "Hello, how are you? I am fine. Thank you for asking."

tokenizer = char_tokenizer(text)

encoded= tokenizer.encode(text)

#convert id to pythorch tensor

data =torch.tensor(encoded,dtype=torch.long)

#hyper params

block_size = 5
batch_size = 2

def get_batch():
    #generate random starting positions
    ix=torch.randint(
        len(data)-block_size-1, (batch_size,)
    )
    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])
    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x, y

# Generate a batch
x, y = get_batch()

print("Input batch:")
print(x)

print("\nTarget batch:")
print(y)

print("\nInput shape:", x.shape)
print("Target shape:", y.shape)
