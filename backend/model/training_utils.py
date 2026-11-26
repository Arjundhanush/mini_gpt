import torch
import torch.nn as nn


def calculate_loss(logits, targets):
    """
    Calculate next-token prediction loss.

    logits:  (batch_size, sequence_length, vocab_size)
    targets: (batch_size, sequence_length)
    """

    batch_size, sequence_length, vocab_size = logits.shape

    # Flatten batch and sequence dimensions
    logits = logits.reshape(batch_size * sequence_length, vocab_size)
    targets = targets.reshape(batch_size * sequence_length)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(logits, targets)

    return loss


def create_optimizer(model, learning_rate=0.001):
    """
    Create an AdamW optimizer for the model.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    return optimizer