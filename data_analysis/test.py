import torch

x = torch.load("/home/ashtomer/projects/ares/data_analysis/results_matrix_incremental.pt")

print(x)
print(x.shape)
print(x.mean())
print(x.std())
print(x.min())
print(x.max())