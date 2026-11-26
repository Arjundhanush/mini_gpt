
import sys
from pathlib import Path

import torch

# Make backend/model modules accessible
MODEL_DIR = Path(__file__).parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from data import tokenizer, get_batch
from gpt import GPTLanguageModel
from training_utils import calculate_loss, create_optimizer


# -------------------------------
# 1. Configuration
# -------------------------------

batch_size = 32
block_size = 8
embedding_dim = 32
num_heads = 4
num_layers = 2

learning_rate = 0.001
max_steps = 500
eval_interval = 100
eval_iters = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Training on:", device)
print("Vocabulary size:", tokenizer.vocab_size)


# -------------------------------
# 2. Create GPT model
# -------------------------------

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)


# -------------------------------
# 3. Create optimizer
# -------------------------------

optimizer = create_optimizer(
    model,
    learning_rate=learning_rate
)


# -------------------------------
# 4. Estimate average loss
# -------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = get_batch(
                split,
                batch_size,
                block_size
            )

            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)

            loss = calculate_loss(logits, y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# -------------------------------
# 5. Training loop
# -------------------------------

print("\nStarting training...\n")

model.train()

for step in range(max_steps):

    # Get training batch
    x, y = get_batch(
        "train",
        batch_size,
        block_size
    )

    x = x.to(device)
    y = y.to(device)

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    logits, _ = model(x)

    # Calculate loss
    loss = calculate_loss(logits, y)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print evaluation periodically
    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train Loss: {losses['train']:.4f}, "
            f"Validation Loss: {losses['val']:.4f}"
        )


print("\nTraining completed!")

# Save trained model
torch.save(model.state_dict(), "mini_gpt.pth")

print("Model saved as mini_gpt.pth")
