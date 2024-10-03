"""
https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file#usage
"""

import torch
import torch.nn as nn
import torchaudio
# from vit_pytorch import ViT
from vit_pytorch import SimpleViT # trains faster and better according to the original authors
from datetime import datetime
# from scipy.signal import ShortTimeFFT
import os
import json
import shap

from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, Dataset

from torch.utils.data import random_split

""" 
Example params (practical params will not be as strong to provide for weaker hardware):
{
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
}
"""

def import_data(data_filename, device):
    with open(data_filename, 'r') as f:
        data = json.load(f)

    targets = torch.tensor(data['Y']).clone().detach().to(device)
    inputs = torch.tensor(data['respMatrix']).clone().detach().to(device)

    return inputs, targets

def train_one_epoch(model, output_fn, train_loader, optimizer, criterion, device):
    model.train()
    train_loss_avg = 0
    output_total = []
    targets_total = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output_raw = model(data)
        output_finished = output_fn(output_raw)
        loss = criterion(output_raw, target)
        loss.backward()
        optimizer.step()

        output_total.extend(output_finished.detach().cpu().numpy())
        targets_total.extend(target.detach().cpu().numpy())
        train_loss_avg += loss.to('cpu').item()

    train_loss_avg /= len(train_loader)
    return train_loss_avg, output_total, targets_total

def validate_one_epoch(model, output_fn, val_loader, criterion, device):
    model.eval()
    val_loss_avg = 0
    output_total = []
    targets_total = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).float()
            output_raw = model(data)
            output_finished = output_fn(output_raw)
            val_loss = criterion(output_raw, target).to('cpu').item()

            output_total.extend(output_finished.detach().cpu().numpy())
            targets_total.extend(target.detach().cpu().numpy())
            val_loss_avg += val_loss

    val_loss_avg /= len(val_loader)
    return val_loss_avg, output_total, targets_total

def calculate_metrics(output_total, targets_total):
    # output_total_sigmoid = torch.sigmoid(torch.tensor(output_total)).numpy()
    # output_total_rounded = (output_total_sigmoid > 0.5).astype(int)
    # accuracy = accuracy_score(targets_total, output_total_rounded)
    # auroc = roc_auc_score(targets_total, output_total_rounded)
    accuracy = accuracy_score(targets_total, output_total)
    auroc = roc_auc_score(targets_total, output_total)
    return accuracy, auroc

def log_epoch_results(epoch, train_loss, val_loss, train_accuracy, train_auroc, val_accuracy, val_auroc, logging_dir, model_name):
    with open(f"{logging_dir}/epoch_info_{model_name}.csv", 'a') as f:
        f.write(f"{epoch}, {train_loss}, {val_loss}, {train_accuracy}, {train_auroc}, {val_accuracy}, {val_auroc}\n")

def create_spectrograms(inputs, device, n_fft=4, win_length=None, hop_length=None):
    # Initialize the Spectrogram transform
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, 
        win_length=win_length, 
        hop_length=hop_length
    ).to(device)

    # Create a list of spectrograms
    spectrograms = []
    for i in range(inputs.shape[0]):
        spectrogram = spectrogram_transform(inputs[i])
        spectrograms.append(spectrogram)
    
    return torch.stack(spectrograms)

def training_loop(model, output_fn, train_loader, val_loader, optimizer, criterion, device, max_epochs, logging_dir, model_name='vit_spectrogram'):
    for epoch in range(max_epochs):
        train_loss, train_output, train_targets = train_one_epoch(model, output_fn, train_loader, optimizer, criterion, device)
        train_accuracy, train_auroc = calculate_metrics(train_output, train_targets)

        val_loss, val_output, val_targets = validate_one_epoch(model, output_fn, val_loader, criterion, device)
        val_accuracy, val_auroc = calculate_metrics(val_output, val_targets)

        log_epoch_results(epoch, train_loss, val_loss, train_accuracy, train_auroc, val_accuracy, val_auroc, logging_dir, model_name)
        
        if val_accuracy == 1.0 or val_auroc == 1.0 or \
        train_accuracy == 1.0 or train_auroc == 1.0:
            print(f"Early stopping at epoch {epoch}")
            # save the model
            torch.save(model.state_dict(), f"{logging_dir}/{model_name}_model.pth")
            return model
    return model

def create_data_loaders(inputs, targets, train_split=0.8, batch_size=8, seed=None):
    # Ensure reproducibility if seed is provided
    if seed is not None:
        torch.manual_seed(seed)

    dataset = TensorDataset(inputs, targets)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class OutputFn(nn.Module):
    def __init__(self):
        super(OutputFn, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        x = torch.round(x)
        return x

def create_model(vit_params, spectrogram_params):
    model = SimpleViT(
        image_size = vit_params['image_size'],
        patch_size = vit_params['patch_size'],
        num_classes = vit_params['num_classes'],
        dim = vit_params['dim'],
        depth = vit_params['depth'],
        heads = vit_params['heads'],
        mlp_dim = vit_params['mlp_dim'],
        channels = vit_params['channels'],
        dim_head = vit_params['dim_head'],
    )

    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=spectrogram_params['n_fft'],
        win_length=spectrogram_params['win_length'],
        hop_length=spectrogram_params['hop_length'],
    )

    return OverallModel(model, spectrogram_transform)

class OverallModel(torch.nn.Module):
    def __init__(self, model, spectrogram_transform):
        super(OverallModel, self).__init__()
        self.model = model
        self.spectrogram_transform = spectrogram_transform

    def forward(self, x):
        x = self.spectrogram_transform(x)
        x = self.model(x)
        x = torch.squeeze(x, dim=1)
        return x

def shap_analysis(model, inputs, targets):
    def model_predict(x):
        x = torch.tensor(x).to(device)
        spectrograms = create_spectrograms(x, device)
        return model(spectrograms).squeeze(1).detach().cpu().numpy()
    
    explainer = shap.Explainer(model_predict, inputs, max_evals=3441)
    shap_values = explainer(inputs)
    # explainer = shap.Explainer(model, inputs, max_evals=3441)
    # shap_values = explainer(inputs)

    shap.plots.waterfall(shap_values[0])

    shap.summary_plot(shap_values, inputs, feature_names="binary")
    
    shap.dependence_plot("RM", shap_values, inputs)

    return shap_values

def create_test_spectrogram(time_series, spectrogram_params, device):
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=spectrogram_params['n_fft'],
        win_length=spectrogram_params['win_length'],
        hop_length=spectrogram_params['hop_length'],
    ).to(device)

    return spectrogram_transform(time_series)

if __name__ == "__main__":
    CURRENT_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CURRENT_DIR = os.getcwd()
    LOGGING_DIR = CURRENT_DIR + "/logs/" + CURRENT_DATETIME + "_vit_spectrogram"
    # create the logging directory
    if os.path.exists(LOGGING_DIR):
        print(f"Directory {LOGGING_DIR} already exists")
        exit()
    os.makedirs(LOGGING_DIR)
    DATA_FILENAME = 'prelick_data_no_zeros1.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inputs: [n_samples, dim0, dim1, frames], targets: [n_samples,]
    # Current dataset sizes: inputs: [3590, 43, 40, 30], targets: [3590]
    inputs, targets = import_data(DATA_FILENAME, device)

    N_SAMPLES = inputs.shape[0]
    DIM0 = inputs.shape[1] # 43
    DIM1 = inputs.shape[2] # 40
    FRAMES = inputs.shape[3]
    NUM_PIXELS = DIM0 * DIM1

    MAX_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    TRAIN_TEST_SPLIT = 0.8

    spectrogram_params = {
        'n_fft': 4,
        'win_length': None,
        'hop_length': None
    }

    # reshape inputs to [n_samples, total_pixels, frames] by flattening the first two dimensions
    inputs = inputs.view(N_SAMPLES, NUM_PIXELS, FRAMES)

    test_spectrogram = create_test_spectrogram(inputs[0][0].unsqueeze(0), spectrogram_params, device) # [1, 3, 16]

    SPEC_DIM0 = test_spectrogram.shape[1] # 3
    SPEC_DIM1 = test_spectrogram.shape[2] # 16

    model_params = {
        'image_size': (SPEC_DIM0, SPEC_DIM1),
        'patch_size': 1, # 4,

        'num_classes': 1,
        'dim': 1024,
        'depth': 2,
        'heads': 16,
        'mlp_dim': 2048,
        'channels': NUM_PIXELS,
        'dim_head': 64
    }

    """ 
    Important Model Conditions:
    1) eight and width are less than or equal to the image_size, and both divisible by patch_size
    2) image size is a tuple of (height, width)
    3) patch_size is a tuple of (height, width)
    """
    model = create_model(model_params, spectrogram_params).to(device)
    output_fn = OutputFn().to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = create_data_loaders(inputs, 
                                                   targets, 
                                                   train_split=TRAIN_TEST_SPLIT, 
                                                   batch_size=BATCH_SIZE,
    )

    # save the training and validation data loaders
    torch.save(train_loader, f"{LOGGING_DIR}/train_loader.pth")
    torch.save(val_loader, f"{LOGGING_DIR}/val_loader.pth")
    
    training_loop(model, output_fn, train_loader, val_loader, optimizer, criterion, device, MAX_EPOCHS, LOGGING_DIR)

    shap_values = shap_analysis(model, inputs, targets)

    print()
    