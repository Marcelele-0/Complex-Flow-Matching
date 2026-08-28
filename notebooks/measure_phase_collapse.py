import torch
import numpy as np
from torch.utils.data import DataLoader
from cfm.data.dataset import SKMTEADataset
from cfm.data.transforms import (
    Compose,
    ComplexToCylinderTransform,
    AmplitudeNormalize,
    CenterCropModulo,
)


def main():
    pipeline = Compose(
        [ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=16)]
    )
    dataset = SKMTEADataset(
        data_dir="/home/marcel/Programming/papers/cfm/data/skm-tea-mini/v1-release",
        transform=pipeline,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    ratios = []

    for batch in loader:
        # batch is [B, 3, H, W]: Amp, cos, sin
        amp_1 = batch[:, 0]
        cos_1 = batch[:, 1]
        sin_1 = batch[:, 2]

        # reconstruct complex z1
        z1 = amp_1 * torch.exp(1j * torch.atan2(sin_1, cos_1))

        # sample z0 (noise)
        amp_0 = torch.rand_like(amp_1)
        phi_0 = torch.rand_like(amp_1) * 2 * np.pi
        z0 = amp_0 * torch.exp(1j * phi_0)

        # midpoint euclidean
        z_mid = 0.5 * z0 + 0.5 * z1

        amp_mid = torch.abs(z_mid)
        amp_expected = 0.5 * (torch.abs(z0) + torch.abs(z1))

        # ratio
        ratio = amp_mid / (amp_expected + 1e-8)

        ratios.append(ratio.mean().item())

        if len(ratios) > 10:
            break

    print(f"Mean amplitude ratio |z_mid| / (0.5*(|z0| + |z1|)) = {np.mean(ratios):.4f}")


if __name__ == "__main__":
    main()
