import gc, os
import torch
from timm.models import create_model
from autoattack import AutoAttack
import re
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
import random
import datetime
import logging

def n_batch_loader(loader, n=8):
    idx = list(range(len(loader)))
    random.shuffle(idx)
    chosen = set(idx[:n])

    for i, batch in enumerate(loader):
        if i in chosen:
            yield batch

def save_csv(results_dict, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "clean_acc", "robust_acc"])
        for model, (clean, robust) in results_dict.items():
            writer.writerow([model, clean, robust])

def get_imagenet_raw_loader(batch_size=128, workers=8, path="/mnt/data/datasets/imagenet/val"):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()      # <-- STOP here
    ])

    ds = datasets.ImageFolder(path, transform=tf)

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)

class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def parse_attack_from_path(checkpoint_path):
    name = checkpoint_path.split("/")[-1].lower()

    # ---------- Baseline ----------
    if "baseline" in name:
        return True, None, None

    # ---------- Norm ----------
    if "linf" in name:
        norm = "Linf"
    elif "l2" in name:
        norm = "L2"
    else:
        raise ValueError("Cannot detect norm from checkpoint name.")

    # ---------- Extract eps number ----------
    # matches: linf_1, linf_2, linf_8, l2_0.5, etc.
    m = re.search(rf"{norm.lower()}[_\-]?([0-9]*\.?[0-9]+)", name)
    if not m:
        raise ValueError("Cannot extract eps from filename.")

    eps_raw = float(m.group(1))

    # ---------- Convert to true epsilon ----------
    if norm == "Linf":
        eps = eps_raw / 255
    else:  # L2
        eps = eps_raw

    return False, norm, eps


def autoattack_eval(checkpoint_path, batch_size=128, device="cuda",num_batches=5, logger=None, log_path=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    skip_auto, norm, eps = parse_attack_from_path(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(ckpt["arch"], pretrained=False, num_classes=1000).eval().to(device)

    if "state_dict_ema" in ckpt:
        model.load_state_dict(ckpt["state_dict_ema"])
    else:
        model.load_state_dict(ckpt["state_dict"])

    model = NormalizeWrapper(model).to(device).eval()

    loader = get_imagenet_raw_loader(batch_size=batch_size)

    # ---- Clean accuracy ----
    logger.info("Running clean evaluation...")
    correct, total = 0, 0

    for x,y in tqdm(loader, desc=f"AA {norm} eps={eps}", ncols=100):

        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    clean_acc = correct/total
    logger.info(f"Clean Acc: {clean_acc*100:.2f}%")


    if skip_auto:
        return clean_acc, None

    # ---- AutoAttack ----
    adversary = AutoAttack(model, norm=norm, eps=eps, version="standard", verbose=True, log_path=log_path)

    logger.info(f"Running AutoAttack on {num_batches} random batches ({norm}, eps={eps})...")
    robust_correct, robust_total = 0, 0

    for x,y in n_batch_loader(loader, n=num_batches):
        x,y = x.to(device), y.to(device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))

        with torch.no_grad():
            pred = model(x_adv).argmax(1)

        robust_correct += (pred==y).sum().item()
        robust_total += y.size(0)

    robust_acc = robust_correct / robust_total

    logger.info(f"Robust Acc: {robust_acc*100:.2f}%")

    return clean_acc, robust_acc



if __name__ == "__main__":
    log_path = f"/home/tomer_a/Documents/project/ares/data_analysis/logs/autoattack_eval-{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(f"auto-attack_eval-{datetime.datetime.now().strftime('%Y-%m-%d')}")
    models_path = "/mnt/data/robustness models/madry/"
    batch_size = 64
    num_batches = 20
    results = {}
    for root, dirs, files in os.walk(models_path):
        for filename in files:
            if filename.endswith(".tar"):
                ckpt = os.path.join(root, filename)
                name = filename.replace(".tar","")

                logger.info(f"\n===== Evaluating {name} =====")
                results[name] = autoattack_eval(ckpt, batch_size=batch_size, device="cuda", num_batches=num_batches, logger=logger, log_path=log_path)

                # free attack memory
                torch.cuda.empty_cache()
                gc.collect()
    save_path = f"autoattack_results-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
    save_csv(results, save_path)
    logger.info(f"Results saved to {save_path}")