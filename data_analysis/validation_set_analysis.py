import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

def pairwise_l2(A, B):
    """
    Compute pairwise L2 distances between two batches using GPU.
    A: (m, D)
    B: (n, D)
    Returns: (m, n) matrix of L2 distances.
    """
    normA = (A * A).sum(1).unsqueeze(1)
    normB = (B * B).sum(1).unsqueeze(0)
    dist = torch.sqrt(torch.clamp(normA + normB - 2 * A @ B.T, min=0))
    return dist


def build_and_compute_l2_norms(
    eval_dir,
    batch_size=128,
    num_images=None,
    output_path="results_matrix.pt"
):

    ############################################################################
    # 1) LOAD DATASET + PREPARE LABELS
    ############################################################################
    print("Building dataset...")

    dataset_eval = datasets.ImageFolder(
        root=eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    if num_images is not None:
        dataset_eval.samples = dataset_eval.samples[:num_images]

    N = len(dataset_eval)
    print(f"Dataset has {N} images.")

    # Prepare loaders
    loader_A = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    loader_B = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    ############################################################################
    # 2) PREALLOCATE RESULTS MATRIX  (N × N)
    ############################################################################
    # results[i, j] = distance
    results = torch.zeros((N, N), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ############################################################################
    # 3) MAIN LOOP FOR BATCH A
    ############################################################################
    start_A = 0

    for batch_idx_A, (batch_A_cpu, labels_A) in enumerate(loader_A):
        print(f"[A] Batch {batch_idx_A+1}/{len(loader_A)}")

        batch_A = batch_A_cpu.flatten(1).to(device)
        bszA = batch_A.size(0)
        end_A = start_A + bszA

        ########################################################################
        # 3A) A×A block (upper triangle only) — VECTORIZED WRITES
        ########################################################################
        dist_AA = pairwise_l2(batch_A, batch_A).cpu()  # (bszA, bszA)

        # Create matrix of global indices
        idx_i = torch.arange(start_A, end_A).unsqueeze(1).expand(bszA, bszA)
        idx_j = torch.arange(start_A, end_A).unsqueeze(0).expand(bszA, bszA)

        # Upper triangle mask (exclude diagonal)
        mask = torch.triu(torch.ones((bszA, bszA), dtype=torch.bool), diagonal=1)

        # Labels for broadcasting
        # Vectorized writes
        results[idx_i[mask], idx_j[mask]] = dist_AA[mask]

        ########################################################################
        # 3B) A×B blocks for later batches B — VECTORIZED
        ########################################################################
        start_B = end_A

        for batch_idx_B, (batch_B_cpu, labels_B) in enumerate(tqdm.tqdm(loader_B)):

            # Skip B <= A
            if batch_idx_B <= batch_idx_A:
                continue

            batch_B = batch_B_cpu.flatten(1).to(device)
            bszB = batch_B.size(0)
            end_B = start_B + bszB

            # Compute distance block A×B
            dist_AB = pairwise_l2(batch_A, batch_B).cpu()

            # Global index grids
            idx_i = torch.arange(start_A, end_A).unsqueeze(1).expand(bszA, bszB)
            idx_j = torch.arange(start_B, end_B).unsqueeze(0).expand(bszA, bszB)

            # Write full rectangular block in one vector call
            results[idx_i, idx_j] = dist_AB
            start_B = end_B

        start_A = end_A

    ########################################################################
    # 4) SAVE RESULTS
    ########################################################################
    print("Saving results...")
    torch.save(results, output_path)
    print("Done:", output_path)


if __name__ == '__main__':
    print("Starting script...")
    build_and_compute_l2_norms(
        eval_dir="/storage/test/bml_group/tomerash/datasets/imagenet/val/",
        batch_size=128,
        num_images=None,   # Set to None to process the entire dataset
        output_path="/home/ashtomer/projects/ares/data_analysis/results_matrix_incremental.pt"
    )