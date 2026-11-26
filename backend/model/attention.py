
import torch
import torch.nn as nn
import math


# =====================================
# 1. Single Attention Head
# =====================================

class AttentionHead(nn.Module):

    def __init__(self, embedding_dim, head_dim):

        super().__init__()

        # Query, Key and Value projections
        self.query = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

    def forward(self, x):

        # Input shape: (B, T, embedding_dim)

        T = x.shape[1]

        # Step 1: Generate Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Step 2: Calculate attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # Step 3: Scale scores
        scores = scores / math.sqrt(Q.shape[-1])

        # Step 4: Create causal mask
        mask = torch.tril(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool
            )
        )

        # Step 5: Block future tokens
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Step 6: Apply Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Step 7: Weighted sum of Values
        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


# =====================================
# 2. Multi-Head Attention
# =====================================

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Dimensions assigned to each head
        self.head_dim = embedding_dim // num_heads

        # Create multiple independent attention heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Final output projection
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # Store each head's output and weights
        head_outputs = []
        all_attention_weights = []

        # Run all attention heads
        for head in self.heads:

            output, weights = head(x)

            head_outputs.append(output)
            all_attention_weights.append(weights)

        # Concatenate all head outputs
        concatenated = torch.cat(
            head_outputs,
            dim=-1
        )

        # Final linear projection
        final_output = self.output_projection(concatenated)

        # Combine attention weights for inspection
        attention_weights = torch.stack(
            all_attention_weights,
            dim=1
        )

        return final_output, attention_weights


# =====================================
# 3. Testing Multi-Head Attention
# =====================================

if __name__ == "__main__":

    batch_size = 4
    block_size = 8
    embedding_dim = 8
    num_heads = 4

    # Random input embeddings
    x = torch.randn(
        batch_size,
        block_size,
        embedding_dim
    )

    # Create multi-head attention
    attention = MultiHeadAttention(
        embedding_dim,
        num_heads
    )

    # Forward pass
    output, attention_weights = attention(x)

    # Display shapes
    print("Input shape:", x.shape)

    print("Number of heads:", num_heads)

    print("Dimensions per head:", embedding_dim // num_heads)

    print("Attention weights shape:", attention_weights.shape)

    print("Final output shape:", output.shape)

    # Display first batch, first head
    print("\nFirst head attention weights:")
    print(attention_weights[0, 0])

    # Verify softmax row sums
    print("\nAttention row sums (first head):")
    print(attention_weights[0, 0].sum(dim=-1))

    # Verify future-token attention is zero
    future_attention = torch.triu(
        attention_weights,
        diagonal=1
    )

    print("\nAre all future attention values zero?")
    print(torch.all(future_attention == 0).item())

    # Verify output dimensions
    assert output.shape == x.shape

    # Verify causal masking
    assert torch.all(future_attention == 0)

    print("\nAll tests passed!")
