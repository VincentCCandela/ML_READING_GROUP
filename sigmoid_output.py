import torch

if __name__ == "__main__":
    LOGGING_DIR = "logs/2024-09-29_02-25-13_vit_spectrogram"

    # load the model
    model = torch.load(f"{LOGGING_DIR}/vit_spectrogram_model.pth")

    # print the names and sizes to file
    