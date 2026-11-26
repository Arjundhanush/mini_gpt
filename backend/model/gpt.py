
import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        block_size,
        num_heads,
        num_layers
    ):

        super().__init__()

        self.block_size = block_size

        # 1. Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # 2. Positional embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # 3. Stack multiple Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(embedding_dim)

        # 5. Language Model Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx):

        B, T = idx.shape

        assert T <= self.block_size, "Sequence too long!"

        # Token embeddings
        token_emb = self.token_embedding(idx)

        # Positional embeddings
        positions = torch.arange(T, device=idx.device)
        position_emb = self.position_embedding(positions)

        # Combine embeddings
        x = token_emb + position_emb

        # Pass through Transformer blocks
        all_attention_weights = []

        for block in self.blocks:

            x, attention_weights = block(x)

            all_attention_weights.append(attention_weights)

        # Final normalization
        x = self.final_norm(x)

        # Generate vocabulary scores (logits)
        logits = self.lm_head(x)

        all_attention_weights = torch.stack(
            all_attention_weights,
            dim=1
        )

        return logits, all_attention_weights



    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):

        """
        Generate new tokens autoregressively.

        idx: starting token IDs (B, T)
        max_new_tokens: number of new tokens to generate
        temperature: controls randomness
        """

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        for _ in range(max_new_tokens):

            # Keep only the most recent block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get predictions for the final position
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Convert logits into probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append the new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# =====================================
# Testing Complete GPT Model
# =====================================

if __name__ == "__main__":

    batch_size = 4
    block_size = 8
    embedding_dim = 8
    vocab_size = 12
    num_heads = 4
    num_layers = 2

    # Random token IDs
    idx = torch.randint(
        0,
        vocab_size,
        (batch_size, block_size)
    )

    # Create GPT model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        block_size=block_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

    # Forward pass
    logits, attention_weights = model(idx)

    print("Input shape:", idx.shape)

    print("Logits shape:", logits.shape)

    print("Attention weights shape:", attention_weights.shape)

    # Check dimensions
    assert logits.shape == (
        batch_size,
        block_size,
        vocab_size
    )

    # Check causal masking in every block
    assert torch.all(
        torch.triu(attention_weights, diagonal=1) == 0
    )

    print("Causal masking verified!")

    print("All tests passed!")
