
import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.network = nn.Sequential(

            # Expand dimensions
            nn.Linear(embedding_dim, 4 * embedding_dim),

            # Non-linear activation
            nn.GELU(),

            # Project back
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):

        return self.network(x)


# Testing
if __name__ == "__main__":

    batch_size = 4
    block_size = 8
    embedding_dim = 8

    x = torch.randn(
        batch_size,
        block_size,
        embedding_dim
    )

    ff = FeedForward(embedding_dim)

    output = ff(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == x.shape

    print("All tests passed!")
