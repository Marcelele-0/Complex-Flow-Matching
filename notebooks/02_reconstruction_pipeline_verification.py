# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Verification & Reconstruction Pipeline Notebook
#
# This notebook visually and mathematically validates 4x Cartesian undersampling, manifold roundtrips, DC projection, and model forward passes.

# %%
import math
import torch
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2, fftshift, ifftshift

from cfm.manifolds import CylindricalManifold, EuclideanManifold
from cfm.evaluate import dc_project
from cfm.models.cylindrical_unet import CylindricalUNet
from cfm.models.cylindrical_unet_attention import CylindricalUNetAttention
from cfm.models.cylindrical_unet_cross_slice import CylindricalUNetCrossSlice

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Synthetic MRI Slice Generation

# %%
# Generate a synthetic complex slice (pseudo Shepp-Logan-ish)
H, W = 128, 128
y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
r = torch.sqrt(x**2 + y**2)

# Amplitude
amp = torch.zeros(H, W)
amp[r < 0.8] = 0.5
amp[(r < 0.6) & (y > 0)] = 1.0
amp[(x - 0.2)**2 + (y - 0.2)**2 < 0.05] = 2.0

# Phase
phi = torch.zeros(H, W)
phi[r < 0.8] = (x[r < 0.8] + y[r < 0.8]) * math.pi
phi = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi

# Complex Image
z_clean = torch.polar(amp, phi).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(amp.numpy(), cmap='gray')
ax[0].set_title("Ground Truth Magnitude")
ax[1].imshow(phi.numpy(), cmap='twilight', vmin=-math.pi, vmax=math.pi)
ax[1].set_title("Ground Truth Phase")
plt.show()

# %% [markdown]
# ## 3. 4x Cartesian Undersampling & Aliasing

# %%
# Create 1D Cartesian Mask (simulate 4x acceleration)
mask = torch.zeros(1, 1, H, W)
mask[:, :, :, ::4] = 1.0
# Add fully sampled center
center_fraction = 0.08
center_lines = int(W * center_fraction)
start = W // 2 - center_lines // 2
mask[:, :, :, start:start+center_lines] = 1.0

# K-space
kspace_clean = fftshift(fft2(z_clean, norm='ortho'), dim=(-2, -1))
kspace_masked = kspace_clean * mask

# Zero-filled reconstruction
z_aliased = ifft2(ifftshift(kspace_masked, dim=(-2, -1)), norm='ortho')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(mask[0, 0].numpy(), cmap='gray')
ax[0, 0].set_title("1D Cartesian Mask")

ax[0, 1].imshow(torch.log1p(torch.abs(kspace_masked[0, 0])).numpy(), cmap='gray')
ax[0, 1].set_title("Masked K-Space (Log Mag)")

ax[1, 0].imshow(z_aliased[0, 0].abs().numpy(), cmap='gray')
ax[1, 0].set_title("Aliased Magnitude")

ax[1, 1].imshow(z_aliased[0, 0].angle().numpy(), cmap='twilight', vmin=-math.pi, vmax=math.pi)
ax[1, 1].set_title("Aliased Phase")
plt.show()

# %% [markdown]
# ## 4. Manifold Representations & Roundtrip

# %%
cyl_manifold = CylindricalManifold()
euc_manifold = EuclideanManifold()

# Test Cylindrical Roundtrip
state_cyl = cyl_manifold.from_complex(z_clean)
z_rec_cyl = cyl_manifold.to_complex(state_cyl)
err_cyl = torch.abs(z_clean - z_rec_cyl).max().item()

# Test Euclidean Roundtrip
state_euc = euc_manifold.from_complex(z_clean)
z_rec_euc = euc_manifold.to_complex(state_euc)
err_euc = torch.abs(z_clean - z_rec_euc).max().item()

print(f"Cylindrical State Shape: {state_cyl.shape} | Channels: A, cos(θ), sin(θ)")
print(f"Euclidean State Shape: {state_euc.shape} | Channels: Re, Im")
print(f"Cylindrical Roundtrip Max Error: {err_cyl:.2e}")
print(f"Euclidean Roundtrip Max Error: {err_euc:.2e}")

# %% [markdown]
# ## 5. Data Consistency (DC) Projection

# %%
# Let's say our model predicted the aliased image as its clean output
x_pred = cyl_manifold.from_complex(z_aliased)

# Apply DC projection using the measured k-space
x_dc = dc_project(x_pred, cyl_manifold, kspace_masked, mask)
z_dc = cyl_manifold.to_complex(x_dc)
kspace_dc = fftshift(fft2(z_dc, norm='ortho'), dim=(-2, -1))

# Residual on measured lines
residual = torch.abs(kspace_dc * mask - kspace_masked * mask).max().item()
print(f"DC Projection K-Space Residual on sampled lines: {residual:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(z_dc[0, 0].abs().numpy(), cmap='gray')
ax[0].set_title("DC Projected Magnitude")
ax[1].imshow(z_dc[0, 0].angle().numpy(), cmap='twilight', vmin=-math.pi, vmax=math.pi)
ax[1].set_title("DC Projected Phase")
plt.show()

# %% [markdown]
# ## 6. Model Architectures Sanity Checks

# %%
# Dummy batches
x_2d = torch.randn(2, 3, 64, 64, device=device)
x_25d = torch.randn(2, 3, 3, 64, 64, device=device)
t = torch.rand(2, device=device)

# 1. CylindricalUNet
unet = CylindricalUNet(in_channels=3, out_channels=2, base_channels=8).to(device)
out_unet = unet(x_2d, t)
print(f"CylindricalUNet Output Shape: {out_unet.shape} | Mean: {out_unet.mean().item():.4f}, Std: {out_unet.std().item():.4f}")

# 2. CylindricalUNetAttention
unet_attn = CylindricalUNetAttention(in_channels=3, out_channels=2, base_channels=8, channel_mults=[1, 2]).to(device)
out_attn = unet_attn(x_2d, t)
print(f"CylindricalUNetAttention Output Shape: {out_attn.shape} | Mean: {out_attn.mean().item():.4f}, Std: {out_attn.std().item():.4f}")

# 3. CylindricalUNetCrossSlice (2.5D)
unet_25d = CylindricalUNetCrossSlice(in_channels=3, base_channels=8, channel_mults=[1, 2], attn_heads=2).to(device)
out_25d = unet_25d(x_25d, t)
print(f"CylindricalUNetCrossSlice Output Shape: {out_25d.shape} | Mean: {out_25d.mean().item():.4f}, Std: {out_25d.std().item():.4f}")
