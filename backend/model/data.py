
import torch
from tokenizer import char_tokenizer

# Training corpus (small educational dataset)
text = (
    "hello world! hello gpt! "
    "we are learning transformers. "
    "gpt predicts the next character. "
    "deep learning is interesting. "
) * 20

# Create ONE shared tokenizer
tokenizer = char_tokenizer(text)

# Encode the complete dataset
data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

# Split into training and validation data
split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]


def get_batch(split, batch_size, block_size):
    """
    Generate a batch of input and target sequences.

    x: characters given to the model
    y: next characters the model must predict
    """

    source = train_data if split == "train" else val_data

    if len(source) <= block_size:
        raise ValueError("Dataset is too small for the selected block_size.")

    # Random starting positions
    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        source[i:i + block_size] for i in ix
    ])

    # Next-character targets
    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in ix
    ])

    return x, y
