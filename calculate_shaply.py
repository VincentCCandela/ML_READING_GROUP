import torch
import torch.nn as nn
from main import create_model, import_data, create_test_spectrogram, validate_one_epoch, OutputFn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
import subprocess
import time
import sys
import os

def shapely(model, output_fn, criterion, data_loader, num_samples=100, device="cpu"):
    inputs, targets = data_loader.dataset[:]

    n_features = inputs.shape[1]
    shap_values = np.zeros(n_features)

    # Function to calculate the prediction with only specific features present
    def model_prediction(subset_indices):
        subset_data = inputs.clone()
        subset_data[:, [i for i in range(n_features) if i not in subset_indices], :] = 0
        # outputs = model(subset_data)

        dataset = TensorDataset(subset_data, targets)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        val_loss_avg, output_total, targets_total = validate_one_epoch(model, 
                                                                       output_fn, 
                                                                       data_loader,
                                                                       criterion, 
                                                                       device,
                                                                       )
        accuracy = accuracy_score(targets_total, output_total)
        return accuracy

    # Compute Shapley values using Monte Carlo sampling
    for i in range(n_features):
        pre_sum_values = np.zeros(num_samples)
        for j in range(num_samples):
            subset_indices = np.random.choice([j for j in range(n_features) if j != i], size=np.random.randint(0, n_features), replace=False)
            
            subset_indices_with_feature = list(subset_indices) + [i]

            # Calculate marginal contribution of adding feature `i`
            without_feature_pred = model_prediction(subset_indices)
            with_feature_pred = model_prediction(subset_indices_with_feature)

            marginal_contribution = with_feature_pred - without_feature_pred
            weight = np.math.factorial(len(subset_indices)) * np.math.factorial(n_features - len(subset_indices) - 1) / np.math.factorial(n_features)
            pre_sum_values[j] = marginal_contribution * weight

        shap_values[i] = np.sum(pre_sum_values)
    
    return shap_values

def shapely_pytorch_parallel(model, output_fn, criterion, data_loader, num_processes, num_samples=100, device="cpu"):
    inputs, targets = data_loader.dataset[:]
    n_features = inputs.shape[1]
    shap_values = np.zeros(n_features)

    def model_predictions(subset_indices):
        subset_data = inputs.clone()
        subset_data[:, [i for i in range(n_features) if i not in subset_indices], :] = 0

        dataset = TensorDataset(subset_data, targets)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        val_loss_avg, output_total, targets_total = validate_one_epoch(model, 
                                                                       output_fn, 
                                                                       data_loader,
                                                                       criterion, 
                                                                       device,
                                                                       )
        accuracy = accuracy_score(targets_total, output_total)
        return accuracy
    
    def calculate_shapley_value(i):
        pre_sum_values = np.zeros(num_samples)
        for j in range(num_samples):
            subset_indices = np.random.choice([j for j in range(n_features) if j != i], size=np.random.randint(0, n_features), replace=False)
            
            subset_indices_with_feature = list(subset_indices) + [i]

            without_feature_pred = model_predictions(subset_indices)
            with_feature_pred = model_predictions(subset_indices_with_feature)

            marginal_contribution = with_feature_pred - without_feature_pred
            weight = np.math.factorial(len(subset_indices)) * np.math.factorial(n_features - len(subset_indices) - 1) / np.math.factorial(n_features)
            pre_sum_values[j] = marginal_contribution * weight

        return np.sum(pre_sum_values)

    # with mp.Pool(processes=num_processes) as pool:
    #     shap_values = pool.map(calculate_shapley_value, range(n_features))

    with Pool(processes=num_processes) as pool:
        shap_values = pool.map(calculate_shapley_value, range(n_features))

    return shap_values

def parallel_compute_shapley_values(num_features, num_processes, logging_dir):
    script_name = "independent_shapely_calculation.py"

    processes = []
    for i in range(num_features):
        process = subprocess.Popen([sys.executable, script_name, str(i), logging_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)

        while len(processes) >= num_processes:
            # Iterate through processes and remove the ones that are finished
            for p in processes[:]:
                if p.poll() is not None:  # If the process is finished
                    processes.remove(p)

            # Sleep for a short time to avoid busy-waiting
            time.sleep(0.1)

    # Wait for all remaining processes to finish
    for process in processes:
        process.wait()

    return

def get_predefined_constants():
    SPEC_DIM0 = 3
    SPEC_DIM1 = 16
    NUM_PIXELS = 1720
    return SPEC_DIM0, SPEC_DIM1, NUM_PIXELS

def get_model_params():
    SPEC_DIM0, SPEC_DIM1, NUM_PIXELS = get_predefined_constants()
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

    return model_params

def get_spectrogram_params():
    spectrogram_params = {
        'n_fft': 4,
        'win_length': None,
        'hop_length': None
    }

    return spectrogram_params

def get_parallel_compute_params():
    LOGGING_DIR = "logs/2024-09-29_02-25-13_vit_spectrogram"
    NUM_PIXELS = 1720
    SHAPELY_MONTE_CARLO_SAMPLES = 100

    model_params = get_model_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectrogram_params = get_spectrogram_params()
    model = create_model(model_params, spectrogram_params).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    output_fn = OutputFn().to(device)

    val_loader = torch.load(f"{LOGGING_DIR}/val_loader.pth")
    val_inputs = val_loader.dataset[:][0].to(device)
    val_targets = val_loader.dataset[:][1].to(device)

    parallel_compute_params = {
        "model": model,
        "output_fn": output_fn,
        "criterion": criterion,
        "device": device,
        "inputs": val_inputs,
        "targets": val_targets,
        "n_features": NUM_PIXELS, # 1720
        "num_samples": SHAPELY_MONTE_CARLO_SAMPLES # 100
    }

    return parallel_compute_params



if __name__ == "__main__":
    SHAPELY_MONTE_CARLO_SAMPLES = 100

    logging_dir = "logs/2024-09-29_02-25-13_vit_spectrogram"
    # create f"{LOGGING_DIR}/shapley_values" directory
    os.makedirs(f"{logging_dir}/shapley_values", exist_ok=True)
    model_name = "vit_spectrogram"
    DATA_FILENAME = "prelick_data_no_zeros1.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, targets = import_data(DATA_FILENAME, device)

    N_SAMPLES = inputs.shape[0]
    DIM0 = inputs.shape[1]
    DIM1 = inputs.shape[2]
    FRAMES = inputs.shape[3]
    NUM_PIXELS = DIM0 * DIM1

    inputs = inputs.view(N_SAMPLES, NUM_PIXELS, FRAMES)

    spectrogram_params = {
        'n_fft': 4,
        'win_length': None,
        'hop_length': None
    }

    test_spectrogram = create_test_spectrogram(inputs[0][0].unsqueeze(0), spectrogram_params, device)

    SPEC_DIM0 = test_spectrogram.shape[1]
    SPEC_DIM1 = test_spectrogram.shape[2]

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

    assert model_params == get_model_params()

    model = create_model(model_params, spectrogram_params).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    output_fn = OutputFn().to(device)


    model.load_state_dict(torch.load(f"{logging_dir}/{model_name}_model.pth"))
    model.eval()

    # save the model layers to file
    with open(f"{logging_dir}/layer_names.txt", 'w') as f:
        for name, layer in model.named_modules():
            if name:  # Skip empty names representing the overall model itself
                f.write(f"{name}\n")

    # get the layer name "model.linear_head"
    target_name = "linear_head"

    for name, layer in model.named_modules():
        if name == target_name:
            print(f"Found layer '{name}': {layer}")
            output_layer = layer
            break
    else:
        print(f"Layer '{target_name}' not found.")


    val_loader = torch.load(f"{logging_dir}/val_loader.pth")
    val_inputs = val_loader.dataset[:][0].to(device)
    val_targets = val_loader.dataset[:][1].to(device)

    NUM_SHAP_WORKERS = 6
    parallel_compute_params = {
        "model": model,
        "output_fn": output_fn,
        "criterion": criterion,
        "device": device,
        "inputs": val_inputs,
        "targets": val_targets,
        "n_features": NUM_PIXELS, # 1720
        "num_samples": SHAPELY_MONTE_CARLO_SAMPLES # 100
    }
    parallel_compute_shapley_values(NUM_PIXELS, NUM_SHAP_WORKERS, logging_dir)

    print()