
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        # First Layer Normalization
        self.ln1 = nn.LayerNorm(embedding_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        # Second Layer Normalization
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Feed-Forward Network
        self.feedforward = FeedForward(embedding_dim)

    def forward(self, x):

        # Step 1: Normalize input
        normalized = self.ln1(x)

        # Step 2: Multi-Head Attention
        attention_output, attention_weights = self.attention(
            normalized
        )

        # Step 3: First residual connection
        x = x + attention_output

        # Step 4: Normalize again
        normalized = self.ln2(x)

        # Step 5: Feed-Forward Network
        ff_output = self.feedforward(normalized)

        # Step 6: Second residual connection
        x = x + ff_output

        return x, attention_weights


# =====================================
# Testing Transformer Block
# =====================================

if __name__ == "__main__":

    batch_size = 4
    block_size = 8
    embedding_dim = 8
    num_heads = 4

    # Random input
    x = torch.randn(
        batch_size,
        block_size,
        embedding_dim
    )

    # Create Transformer Block
    transformer = TransformerBlock(
        embedding_dim,
        num_heads
    )

    # Forward pass
    output, attention_weights = transformer(x)

    print("Input shape:", x.shape)

    print("Attention weights shape:", attention_weights.shape)

    print("Output shape:", output.shape)

    # Verify output shape
    assert output.shape == x.shape

    # Verify causal masking
    future_attention = torch.triu(
        attention_weights,
        diagonal=1
    )

    assert torch.all(future_attention == 0)

    print("Causal masking verified!")

    print("All tests passed!")
