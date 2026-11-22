
import torch
import torch.nn as nn
from tokenizer import char_tokenizer

# Training text
text = "hello world! hello gpt! "

# Create tokenizer
tokenizer = char_tokenizer(text)

# Convert text into token IDs
encoded = tokenizer.encode(text)

# Convert IDs into tensor
data = torch.tensor(encoded, dtype=torch.long)

# Hyperparameters
block_size = 8
batch_size = 4
embedding_dim = 8

# Generate a batch
ix = torch.randint(
    len(data) - block_size,
    (batch_size,)
)

x = torch.stack([
    data[i:i + block_size] for i in ix
])

# Create embedding layer
embedding_layer = nn.Embedding(
    tokenizer.vocab_size,
    embedding_dim
)

# Convert token IDs into embedding vectors
embedded = embedding_layer(x)

print("Vocabulary size:", tokenizer.vocab_size)
print("Input shape:", x.shape)
print("Embedded shape:", embedded.shape)
print("\nEmbedded vectors:")
print(embedded)


# Create positional embedding layer
position_embedding = nn.Embedding(
    block_size,
    embedding_dim
)

# Generate position IDs: 0, 1, 2, ..., 7
positions = torch.arange(block_size)

# Convert position IDs into positional vectors
pos_embedded = position_embedding(positions)

# Combine token and positional embeddings
final_embedding = embedded + pos_embedded

print("\nPosition IDs:")
print(positions)

print("\nPositional embedding shape:")
print(pos_embedded.shape)

print("\nFinal embedding shape:")
print(final_embedding.shape)
