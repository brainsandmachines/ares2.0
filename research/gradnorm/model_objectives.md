# Input-Gradient-Regularized ImageNet CNNs

This folder tracks the open ImageNet-1k CNN checkpoints selected for the GradNorm/gradient-regularization AutoAttack sweep.

## RIG GradNorm ResNet50+GeLU

- Source: <https://github.com/adrianrm99/robustness_input_gradients>
- Released model: `GradNorm - ResNet50+GeLU`
- Checkpoint: Google Drive file `1CvLhHaFVyqmqL6W0P_-H8iw2R_uk8MbM`
- Reported by the repo: `60.34` clean accuracy and `30.00` AutoAttack robust accuracy at Linf `4/255`.

The RIG training code adds an input-gradient penalty to the natural cross-entropy objective. In the GradNorm branch, it computes the gradient of the cross-entropy loss with respect to the input image, applies the selected regularizer (`DBP` in the released GradNorm setting), and adds it with a scheduled weight:

`loss = ce_weight * CE(model(x), y) + gradnorm_weight * alpha * regularizer(d CE / d x, x)`

Their eval path treats this checkpoint as a ResNet50 with GELU activations. ReLU layers must therefore be replaced with GELU before loading/evaluating the checkpoint, even though the activation layers have no state-dict parameters.

## TULIP ResNet50 L2-lambda-0.1 and L2-lambda-1

- Source: <https://github.com/cfinlay/tulip/tree/master/imagenet>
- Checkpoints: `resnet50-L2-lambda-0.1.pth.tar` and `resnet50-L2-lambda-1.pth.tar`

TULIP trains ImageNet-1k ResNet50 models with cross-entropy plus squared L2 input-gradient regularization, described in their ImageNet README as Tikhonov regularization. Their implementation approximates the directional input-gradient norm with a finite difference:

1. Compute per-example cross-entropy `l(x, y)`.
2. Compute `d l / d x` and normalize that gradient per image.
3. Evaluate the model at `x + h * normalized_grad` with `h = 0.01`.
4. Estimate `(l(x + h v, y) - l(x, y)) / h`, square it, average it, divide by `2`, and add `lambda * penalty`.

The two released checkpoints differ only in the regularization strength: `lambda = 0.1` and `lambda = 1`.
