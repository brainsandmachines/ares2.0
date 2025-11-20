import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss.cross_entropy import SoftTargetCrossEntropy,  LabelSmoothingCrossEntropy

def test_softtarget_crossentropy():
    # Generate random predictions (logits) and soft labels
    batch_size, num_classes = 4, 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    soft_labels = F.softmax(torch.randn(batch_size, num_classes), dim=-1)  # Soft labels

    # Initialize the custom SoftTargetCrossEntropy
    custom_loss_fn = SoftTargetCrossEntropy()
    custom_loss = custom_loss_fn(logits, soft_labels)

    # Compute the loss manually using nn.CrossEntropyLoss with soft labels
    nn_loss_fn = nn.CrossEntropyLoss()
    log_probs = F.log_softmax(logits, dim=-1)
    manual_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
    
    # Print and compare the losses
    print(f"Custom SoftTargetCrossEntropy Loss: {custom_loss.item()}")
    print(f"Manual CrossEntropy Loss with Soft Labels: {manual_loss.item()}")

    # Assert that the losses are close
    assert torch.isclose(custom_loss, manual_loss, atol=1e-6), "Losses do not match!"

def test_softtarget_vs_nn_crossentropy():
    # Generate random predictions (logits) and hard labels
    batch_size, num_classes = 4, 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    hard_labels = torch.randint(0, num_classes, (batch_size,))

    # Convert hard labels to soft labels for SoftTargetCrossEntropy
    label_smoothing = 0.1
    soft_labels = F.one_hot(hard_labels, num_classes).float()
    soft_labels = soft_labels * (1 - label_smoothing) + label_smoothing / num_classes

    # Initialize the custom SoftTargetCrossEntropy
    custom_loss_fn = SoftTargetCrossEntropy()
    custom_loss = custom_loss_fn(logits, soft_labels)

    # Compute the loss using nn.CrossEntropyLoss with label smoothing
    nn_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    nn_loss = nn_loss_fn(logits, hard_labels)

    # Print and compare the losses
    print(f"Custom SoftTargetCrossEntropy Loss: {custom_loss.item()}")
    print(f"nn.CrossEntropyLoss with Label Smoothing: {nn_loss.item()}")

    # Assert that the losses are close
    assert torch.isclose(custom_loss, nn_loss, atol=1e-6), "Losses do not match!"
    
def test_label_smoothing_crossentropy():
    # Generate random predictions (logits) and hard labels
    batch_size, num_classes = 4, 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    hard_labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize the custom LabelSmoothingCrossEntropy
    label_smoothing = 0.1
    custom_loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    custom_loss = custom_loss_fn(logits, hard_labels)

    # Compute the loss using nn.CrossEntropyLoss with label smoothing
    nn_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    nn_loss = nn_loss_fn(logits, hard_labels)

    # Print and compare the losses
    print(f"Custom LabelSmoothingCrossEntropy Loss: {custom_loss.item()}")
    print(f"nn.CrossEntropyLoss with Label Smoothing: {nn_loss.item()}")

    # Assert that the losses are close
    assert torch.isclose(custom_loss, nn_loss, atol=1e-6), "Losses do not match!"

if __name__ == "__main__":
    test_softtarget_crossentropy()
    test_softtarget_vs_nn_crossentropy()
    test_label_smoothing_crossentropy()