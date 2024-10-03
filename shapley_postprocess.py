import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


if __name__ == "__main__":
    LOGGING_DIR = "logs/2024-09-29_02-25-13_vit_spectrogram"
    SHAPLEY_DIR = LOGGING_DIR + "/shapley_values"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get all of the names of the files in this directory

    shapley_values = []
    filenames = os.listdir(SHAPLEY_DIR)
    for filename in filenames:
        with open(os.path.join(SHAPLEY_DIR, filename), "r") as f:
            shapley_values.append(float(f.read()))

    # get the index of the smallest number
    min_index = np.argmin(shapley_values)
    print(f"min_index: {min_index}")

    shapley_values = np.array(shapley_values)
    # convert to a tensor
    shapley_values = torch.tensor(shapley_values, device=device)

    # reshape to [3, 16]
    shapley_values = shapley_values.view(43, 40)

    # intensity plot
    # plt.imshow(shapley_values.cpu().numpy(), cmap="hot", interpolation="nearest")
    # make it red and blue
    shapley_values_np = shapley_values.cpu().numpy()

    vmin, vmax = np.min(shapley_values_np), np.max(shapley_values_np)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(shapley_values_np, cmap="bwr", norm=norm, interpolation="nearest")
    plt.colorbar()
    plt.title("Shapley Values (sampling of 100)")
    plt.savefig(f"{LOGGING_DIR}/shapley_values.png")

    print()



