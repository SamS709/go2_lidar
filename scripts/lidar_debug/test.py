import torch
torch.set_printoptions(precision=0, linewidth=200, sci_mode=False)
device = "cuda"



h = 15
w = 10
sigma = 4.0

heightmap = torch.ones((1, h, w), dtype=torch.float32)

N = 90
offset = 0


y = torch.arange(h, device=device, dtype=torch.float32)
x = torch.arange(w, device=device, dtype=torch.float32)
yy, xx = torch.meshgrid(y, x, indexing='ij')

# Center of the grid
cy = (h - offset - 1) / 2.0
cx = (w - 1) / 2.0

# Compute 2D Gaussian
gaussian_dist =  1/torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

# Normalize to create probability distribution
gaussian_prob = gaussian_dist / gaussian_dist.sum()

flat_prob = gaussian_prob.flatten()
sampled_indices = torch.multinomial(flat_prob, N, replacement=True)

# Convert flat indices to 2D indices
sampled_h = sampled_indices // w
sampled_w = sampled_indices % w

# Apply the same sampling to all environments
heightmap_modified = heightmap.clone()
heightmap_modified[:, sampled_h, sampled_w] = 0.0

print(heightmap_modified)
n = h * w - torch.sum(heightmap_modified)
print("N= ",n)