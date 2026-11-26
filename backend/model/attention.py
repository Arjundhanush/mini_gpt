
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.embedding_dim = embedding_dim

        # Query projection
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Key projection
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Value projection
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        # Get sequence length
        T = x.shape[1]

        # Step 1: Generate Query, Key and Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Step 2: Calculate attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # Step 3: Scale attention scores
        scores = scores / math.sqrt(self.embedding_dim)

        # Step 4: Create causal mask
        mask = torch.tril(
            torch.ones(
                T,
                T,
                device=x.device
            )
        )

        # Step 5: Block future token positions
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Step 6: Apply Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Step 7: Multiply attention weights with Values
        output = torch.matmul(
            attention_weights,
            V
        )

        return Q, K, V, scores, attention_weights, output


# -----------------------------------
# Testing Causal Self-Attention
# -----------------------------------

if __name__ == "__main__":

    batch_size = 4
    block_size = 8
    embedding_dim = 8

    # Random input embeddings
    x = torch.randn(
        batch_size,
        block_size,
        embedding_dim
    )

    # Create self-attention object
    attention = SelfAttention(embedding_dim)

    # Forward pass
    Q, K, V, scores, attention_weights, output = attention(x)

    print("Input shape:", x.shape)
    print("Query shape:", Q.shape)
    print("Key shape:", K.shape)
    print("Value shape:", V.shape)

    print("Masked scores shape:", scores.shape)
    print("Attention weights shape:", attention_weights.shape)
    print("Final output shape:", output.shape)

    print("\nCausal mask:")
    print(torch.tril(torch.ones(block_size, block_size)))

    print("\nAttention weights (first batch):")
    print(attention_weights[0])

    print("\nRow sums:")
    print(attention_weights[0].sum(dim=-1))

    # Verify future-token attention is zero
    future_attention = torch.triu(
        attention_weights[0],
        diagonal=1
    )

    print("\nFuture attention values:")
    print(future_attention)

    print(
        "\nAre all future attention values zero?",
        torch.all(future_attention == 0).item()
    )
