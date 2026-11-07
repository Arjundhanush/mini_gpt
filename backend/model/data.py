import torch
from tokenizer import char_tokenizer

# Sample training text
text = "hello world! hello gpt! "

# Create tokenizer
tokenizer = char_tokenizer(text)

# Convert text into token IDs
encoded = tokenizer.encode(text)

# Convert IDs into a PyTorch tensor
data = torch.tensor(encoded, dtype=torch.long)

# Input and target sequences
x = data[:-1]
y = data[1:]

print("Original text:", text)
print("Token IDs:", encoded)

print("\nInput (x):", x)
print("Target (y):", y)

print("\nFirst training example:")
print("Input character:", tokenizer.decode([x[0].item()]))
print("Target character:", tokenizer.decode([y[0].item()]))
