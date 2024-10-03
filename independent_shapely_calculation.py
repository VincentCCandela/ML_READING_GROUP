import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from main import validate_one_epoch
from calculate_shaply import get_parallel_compute_params
import sys

def model_predictions(params, subset_indices):
    model = params["model"]
    output_fn = params["output_fn"]
    criterion = params["criterion"]
    device = params["device"]
    inputs = params["inputs"]
    targets = params["targets"]
    n_features = params["n_features"]
    # num_samples = params["num_samples"]
    
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


def calculate_shapley_value(i, params):
    # model = params["model"]
    # output_fn = params["output_fn"]
    # criterion = params["criterion"]
    # device = params["device"]
    # inputs = params["inputs"]
    # targets = params["targets"]
    n_features = params["n_features"]
    num_samples = params["num_samples"]
    pre_sum_values = np.zeros(num_samples)
    for j in range(num_samples):
        subset_indices = np.random.choice([j for j in range(n_features) if j != i], size=np.random.randint(0, n_features), replace=False)
        
        subset_indices_with_feature = list(subset_indices) + [i]

        without_feature_pred = model_predictions(params, subset_indices)
        with_feature_pred = model_predictions(params, subset_indices_with_feature)

        marginal_contribution = with_feature_pred - without_feature_pred
        weight = np.math.factorial(len(subset_indices)) * np.math.factorial(n_features - len(subset_indices) - 1) / np.math.factorial(n_features)
        pre_sum_values[j] = marginal_contribution * weight

    return np.sum(pre_sum_values)

if __name__ == "__main__":
    i = int(sys.argv[1])
    LOGGING_DIR = sys.argv[2]
    
    parallel_compute_params = get_parallel_compute_params()

    shapely_value_i = calculate_shapley_value(i, parallel_compute_params)

    # save the shapley value to a file
    with open(f"{LOGGING_DIR}/shapley_values/shapley_values_{i}.txt", "w") as f:
        f.write(str(shapely_value_i))
