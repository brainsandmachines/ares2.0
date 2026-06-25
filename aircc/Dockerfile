FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git wget && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# PyTorch 2.9.0 with CUDA 12.8
RUN pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Core ML / CV packages (pinned to tomer_advtrain_pro versions)
RUN pip install \
    timm==1.0.22 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.23.0 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    kornia==0.8.3 \
    kornia_rs==0.1.14 \
    numpy==2.3.4 \
    scipy==1.16.3 \
    matplotlib==3.10.8 \
    pandas==2.3.3 \
    pillow==12.0.0 \
    safetensors==0.6.2 \
    huggingface_hub==1.1.2 \
    gitpython==3.1.45 \
    pydantic==2.12.4 \
    pytest==9.0.3 \
    gdown==5.2.0 \
    sentry-sdk==2.45.0

# Git-sourced packages — pinned to the same commits as tomer_advtrain_pro
RUN pip install git+https://github.com/fra31/auto-attack.git@a39220048b3c9f2cca9a4d3a54604793c68eca7e
RUN pip install git+https://github.com/KietzmannLab/DVD.git@8a4e919812cfa52aa66ecaab121d0bdf5840a257
RUN pip install git+https://github.com/KietzmannLab/DVD.git@36e56374bb73f52a97fc6e3a39531cba2443bb2f

# Build-time import test — fails the build if anything is broken
RUN python -c "import torch, torchvision, timm, hydra, wandb, kornia, autoattack, dvd; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('all imports ok')"
