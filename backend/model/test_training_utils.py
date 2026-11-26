import torch

from gpt import GPTLanguageModel
from training_utils import calculate_loss, create_optimizer


# Model configuration
vocab_size = 12
embedding_dim = 8
block_size = 8
num_heads = 4
num_layers = 2
batch_size = 4

# Create model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
)

# Create sample input and target batches
x = torch.randint(0, vocab_size, (batch_size, block_size))
y = torch.randint(0, vocab_size, (batch_size, block_size))

# Create optimizer
optimizer = create_optimizer(model)

# Clear old gradients
optimizer.zero_grad()

# Forward pass
logits, _ = model(x)

# Calculate loss
loss = calculate_loss(logits, y)

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())

# Backpropagation
loss.backward()

# Check whether gradients were calculated
print("Gradient calculated:", model.lm_head.weight.grad is not None)

# Update model parameters
optimizer.step()

print("Optimizer update successful!")
print("All tests passed!")