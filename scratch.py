import torch

def calc(mask_varnet):
    proj = mask_varnet[0, 0, :, :, 0].any(dim=0).to(torch.int8)
    cent = proj.shape[0] // 2
    
    left_half = proj[:cent].flip(0)
    left_zero = torch.argmin(left_half)
    if left_half[left_zero] != 0:
        left_zero = torch.tensor(cent)
        
    right_half = proj[cent:]
    right_zero = torch.argmin(right_half)
    if right_half[right_zero] != 0:
        right_zero = torch.tensor(proj.shape[0] - cent)
        
    num_low_frequencies = int(max(2 * min(left_zero, right_zero), 1))
    return num_low_frequencies

mask = torch.zeros(1, 1, 32, 32, 1, dtype=torch.bool)
mask[:, :, 16-4:16+4, 16-4:16+4, :] = True
print(calc(mask))

mask = torch.zeros(1, 1, 32, 32, 1, dtype=torch.bool)
mask[:, :, :, 16-2:16+2, :] = True
print(calc(mask))

mask = torch.ones(1, 1, 32, 32, 1, dtype=torch.bool)
print(calc(mask))
