
import torch

from data import tokenizer
from gpt import GPTLanguageModel


# Configuration (must match training configuration)
block_size = 8
embedding_dim = 32
num_heads = 4
num_layers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


# Create the model
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)


# Load trained parameters
model.load_state_dict(
    torch.load("mini_gpt.pth", map_location=device)
)

model.eval()


# Starting prompt
prompt = "hello"

# Convert prompt into token IDs
context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)


# Generate text
generated = model.generate(
    context,
    max_new_tokens=100,
    temperature=0.8
)


# Decode token IDs into text
output = tokenizer.decode(
    generated[0].tolist()
)

print("\nGenerated text:")
print(output)
