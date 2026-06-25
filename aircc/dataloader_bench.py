#!/usr/bin/env python3
"""Benchmark DataLoader throughput across num_workers values (pin_memory=True).
Runs a warmup batch then measures N batches for each worker count.
"""
import argparse
import csv
import os
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def bench(train_dir, batch_size, num_workers, num_batches):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        datasets.ImageFolder(train_dir, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    it = iter(loader)
    next(it)  # warmup — not measured
    start = time.perf_counter()
    for _ in range(num_batches):
        next(it)
    elapsed = time.perf_counter() - start
    return elapsed, batch_size * num_batches / elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--num-batches', type=int, default=2)
    p.add_argument('--out-csv', required=True)
    p.add_argument('--protocol', default='')
    p.add_argument('--arch', default='')
    args = p.parse_args()

    rows = []
    for nw in [4, 6, 8]:
        print(f'[dataloader_bench] num_workers={nw} pin_memory=True ...', flush=True)
        try:
            elapsed, throughput = bench(args.train_dir, args.batch_size, nw, args.num_batches)
            print(f'  {args.num_batches} batches in {elapsed:.3f}s → {throughput:.1f} samples/s')
        except Exception as e:
            print(f'  ERROR: {e}')
            elapsed, throughput = -1.0, -1.0
        rows.append({
            'protocol': args.protocol,
            'arch': args.arch,
            'bsz': args.batch_size,
            'num_workers': nw,
            'num_batches': args.num_batches,
            'elapsed_s': round(elapsed, 3),
            'throughput_samples_per_s': round(throughput, 1),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f'[dataloader_bench] written → {args.out_csv}')


if __name__ == '__main__':
    main()
