import numpy as np
import pandas as pd
import polars as pl
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# LSTM Encoder using GRU
class LSTMEncode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMEncode, self).__init__()
        self.output_size = output_size
        self.lstm = nn.GRU(input_size=input_size + output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_past, y_past):
        x = torch.cat((x_past, y_past), dim=2).to(device)
        # Flatten GRU parameters to resolve memory chunk issue
        self.lstm.flatten_parameters()
        x, ht = self.lstm(x)
        x = self.linear(x)
        return x[:, -1, :], ht[-1, :, :]


# LSTM Decoder using GRU
class LSTMDecode(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMDecode, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.teacher_forcing_prob = 0.0
        self.lstm_cell = nn.GRUCell(input_size=input_size + output_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_fut, y_past, ht, y_fut_teacher=None):
        y_last = y_past[:, -1, :]
        yh_fut = torch.zeros((x_fut.shape[0], x_fut.shape[1], self.output_size)).to(device)
        t_horizon = x_fut.shape[1]

        for t in range(t_horizon):
            X_t = torch.cat((x_fut[:, t, :], y_last), dim=1)
            ht = self.lstm_cell(X_t, ht)
            y_fut = self.linear(ht)
            yh_fut[:, t, :] = y_fut
            y_last = y_fut if (y_fut_teacher is None or np.random.rand() > self.teacher_forcing_prob) else y_fut_teacher[:, t, :]

        return yh_fut

# Sequence to Sequence LSTM model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, lstm_encoder: LSTMEncode, lstm_decoder: LSTMDecode):
        super(Seq2SeqLSTM, self).__init__()
        self.enc = lstm_encoder
        self.dec = lstm_decoder

    def set_teacher_forcing_prob(self, prob):
        self.dec.teacher_forcing_prob = prob

    def forward(self, x_past, y_past, x_fut, y_fut_teacher=None):
        _, ht = self.enc(x_past, y_past)
        yfut_all = self.dec(x_fut, y_past, ht, y_fut_teacher=y_fut_teacher)
        return yfut_all

# Create batches for training
def make_batch(X_vals, y_vals=None, lookback=1, lookfor=1, rand_batch=True, start_idx=0):
    """
    Create a batch of input data with flexible lookback and lookfor sizes.
    """
    data_size = len(X_vals)
    start_index = random.randint(0, data_size - lookback - lookfor) if rand_batch else start_idx

    inp_features_past = X_vals[start_index:start_index + lookback]
    inp_features_future = X_vals[start_index + lookback:start_index + lookback + lookfor]

    if y_vals is not None:
        out_features_past = y_vals[start_index:start_index + lookback]
        out_features_future = y_vals[start_index + lookback:start_index + lookback + lookfor]
        return inp_features_past, inp_features_future, out_features_past, out_features_future
    return inp_features_past, inp_features_future

# Generate multiple batches for training
def gen_batches(x_train_tensor, y_train_tensor, n_batches, lookback, lookfor, input_size, start_idxs=[]):
    """
    Generate batches of training data with dynamic lookback and lookfor windows.
    """
    x_train_batches, y_train_batches = [], []
    x_train_batch_future, y_train_batch_future = [], []

    for b in range(n_batches):
        if len(start_idxs) == n_batches:
            x_bat_past, x_bat_fut, y_bat_past, y_bat_fut = make_batch(
                x_train_tensor, y_vals=y_train_tensor, lookback=lookback, lookfor=lookfor, rand_batch=False, start_idx=start_idxs[b])
        else:
            x_bat_past, x_bat_fut, y_bat_past, y_bat_fut = make_batch(x_train_tensor, y_vals=y_train_tensor, lookback=lookback, lookfor=lookfor)

        x_train_batches.append(x_bat_past)
        y_train_batches.append(y_bat_past)
        x_train_batch_future.append(x_bat_fut)
        y_train_batch_future.append(y_bat_fut)

    return torch.stack(x_train_batches), torch.stack(x_train_batch_future), \
           torch.stack(y_train_batches), torch.stack(y_train_batch_future)

def pre_fit(X, y):
    """
    Scale and split data into training and testing sets, then convert to tensors.
    """
    scaler_x, scaler_y = StandardScaler(), StandardScaler()  # Create the scalers
    scaled_X = scaler_x.fit_transform(X)  # Scale the input features
    scaled_y = scaler_y.fit_transform(y)  # Scale the target values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size=0.375, shuffle=False)

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, len(X_train), len(X_test), len(X), scaler_y


# Custom loss function
def custom_loss(y_pred, y_known):
    y_known = y_known.to(y_pred.device)
    inital_cond_loss = ((y_pred[:, 0, :] - y_known[:, 0, :]) ** 2.0).mean()
    return ((y_pred - y_known) ** 2.0).mean() * 0.5 + inital_cond_loss * 0.5

# LSTM Model Estimator class
class MyLSTMEstimator(BaseEstimator):
    def __init__(self, input_size, epochs=2500, hidden_size= 32, num_layers=1, num_batches=256, lookback= 720, lookfor= 720, lr=1e-4, output_size=1):
        self.lookfor = lookfor
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_batches = num_batches
        self.lookback = lookback
        self.lr = lr
        self.output_size = output_size
        self.input_size = input_size

        self.enc = LSTMEncode(self.input_size, self.hidden_size, self.num_layers, self.output_size).to(device)
        self.dec = LSTMDecode(self.input_size, self.hidden_size, self.num_layers, self.output_size).to(device)
        self.model = Seq2SeqLSTM(self.enc, self.dec).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.loss_fn = custom_loss
        self.best_model = None

    def fit(self, X_train_tensor, y_train_tensor, X_test_tensor=None, y_test_tensor=None):
        train_loss_hist, test_loss_hist = [], []
        best_test_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            x_train_batched, x_train_batched_fut, y_train_batched, y_train_batched_fut = gen_batches(
                X_train_tensor, y_train_tensor, self.num_batches, self.lookback, self.lookfor, self.input_size)
            
            teacher_forcing_prob = np.clip((1. - epoch / self.epochs) ** 4.5, 0.0, 1.0)
            self.model.set_teacher_forcing_prob(teacher_forcing_prob)

            y_pred = self.model(x_train_batched, y_train_batched, x_train_batched_fut, y_train_batched_fut)
            loss = self.loss_fn(y_pred, y_train_batched_fut)
            train_loss_hist.append(float(loss))

            loss.backward()
            self.optimizer.step()

            if X_test_tensor is not None:
                with torch.no_grad():
                    x_test_batched, x_test_batched_fut, y_test_batched, y_test_batched_fut = gen_batches(
                        X_test_tensor, y_test_tensor, self.num_batches, self.lookback, self.lookfor, self.input_size)
                    y_pred_test = self.model(x_test_batched, y_test_batched, x_test_batched_fut)
                    test_loss = self.loss_fn(y_pred_test, y_test_batched_fut)
                    test_loss_hist.append(float(test_loss))

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        self.best_model = deepcopy(self.model)

            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if self.best_model:
            self.model = self.best_model

        self.train_loss_hist = train_loss_hist
        self.test_loss_hist = test_loss_hist
        return self

    # Save the model to file
    def save_model_to_file(self, filename):
        torch.save(self.model.state_dict(), filename)

    # Load the model from a file
    def load_model_from_file(self, filename):
        self.model.load_state_dict(torch.load(filename))

    # Predict method using the trained model
    def predict(self, x_fut, x_past=None, y_past=None):
        self.model.eval()
        assert x_past is not None and y_past is not None, "x_past and y_past are required for prediction"
        y_pred = self.model(x_past, y_past, x_fut)
        return y_pred

    # Plot training and test loss over epochs
    def post_plot(self):
        plt.plot(list(range(self.epochs)), self.train_loss_hist, lw=1, alpha=0.75, label='Train')
        plt.plot(list(range(self.epochs)), self.test_loss_hist, lw=1, alpha=0.75, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.grid(ls='--')
        plt.legend()
        plt.savefig('seq2seq_training_curve.jpg', dpi=300)
        plt.close()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def single_lstm_run(X, y, lookback, lookfor, target_feature_idx, scaler_y, load_model=False):


    input_size = X.shape[-1]

    # Initialize the estimator
    lstm_estimator = MyLSTMEstimator(
        input_size, epochs=2500, hidden_size= 32, num_layers= 3, num_batches=256,  ################################################
        lookback=lookback, lookfor=lookfor, lr=1e-3, output_size=3)

    # Preprocess the data (get X_train, y_train, X_test, y_test)
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, _, _, _, scaler_y = pre_fit(X, y)


    # Train or load the model
    if not load_model:
        lstm_estimator.fit(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        lstm_estimator.save_model_to_file("seq2seq_model.torch")
    else:
        lstm_estimator.load_model_from_file("seq2seq_model.torch")

    # Plot the loss curves
    if not load_model:
        lstm_estimator.post_plot()

    # Predict the entire y_test in chunks of lookfor (60)
    total_test_samples = X_test_tensor.shape[0]  # Number of samples in test data
    full_steps = total_test_samples // lookfor  # Full steps that fit exactly into `lookfor` chunks
    remainder = total_test_samples % lookfor  # Remaining samples that don't fit into full steps

    all_predictions = []  # To store all the y_pred values
    current_x_past = X_test_tensor[:lookback].unsqueeze(0)  # Initialize with the first lookback points from X_test
    current_y_past = y_test_tensor[:lookback].unsqueeze(0)  # Initialize with the first lookback points from y_test

    # Loop through the test set to predict in batches of `lookfor` (60)
    for i in range(full_steps):
        # Predict the next `lookfor` future points
        current_y_pred = lstm_estimator.predict(current_x_past, current_x_past, current_y_past)
        all_predictions.append(current_y_pred.cpu().detach().numpy())  # Collect predictions

        # Update the past inputs for the next prediction
        if (i+1)*lookfor < total_test_samples:
            next_start = (i+1)*lookfor
            next_end = next_start + lookfor
            current_x_past = X_test_tensor[next_start:next_end].unsqueeze(0)  # Update with the next lookfor points
            current_y_past = y_test_tensor[next_start:next_end].unsqueeze(0)  # Update with the next lookfor points

    # Handle the remaining samples that don't fit into a full `lookfor` chunk
    if remainder > 0:
        # Predict for the remaining samples
        last_x_past = X_test_tensor[-(lookback + remainder):-remainder].unsqueeze(0)  # Last `lookback` samples
        last_y_past = y_test_tensor[-(lookback + remainder):-remainder].unsqueeze(0)  # Last `lookback` target samples

        last_y_pred = lstm_estimator.predict(last_x_past, last_x_past, last_y_past)
        last_y_pred = last_y_pred[:, :remainder]  # Ensure we only take the remaining predictions
        all_predictions.append(last_y_pred.cpu().detach().numpy())  # Only take the remaining predictions

    # Concatenate all the predicted future points
    all_predictions = np.concatenate(all_predictions, axis=1)

    # Ensure `y_pred` and `y_test` have the same number of samples
    total_predicted_samples = all_predictions.shape[1]  # Get total predicted samples
    y_test = y_test_tensor.cpu().detach().numpy()

    if total_predicted_samples > y_test.shape[0]:
        # Trim the predicted values if there are too many
        all_predictions = all_predictions[:, :y_test.shape[0], :]
    elif total_predicted_samples < y_test.shape[0]:
        # Trim the actual test values if the predictions are fewer
        y_test = y_test[:total_predicted_samples]

    # Reshape the predicted values for easier comparison
    y_pred = all_predictions.reshape(-1, all_predictions.shape[-1])  # Reshape to (total_test_samples, output_size)

    # **Perform Inverse Scaling before plotting and saving**
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Calculate RMSE and MAE for each feature (using inverse-scaled values)
    for k in range(y_pred_inverse.shape[1]):  # Loop through each output feature
        rmse = np.sqrt(mean_squared_error(y_test_inverse[2000:10000, k], y_pred_inverse[2700:10700, k]))
        mae = mean_absolute_error(y_test_inverse[2000:10000, k], y_pred_inverse[2700:10700, k])
        print(f"Feature {k}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # Save the inverse-scaled predictions and actual values for comparison
    np.savetxt("y_pred_inversedays.csv", y_pred_inverse, delimiter=",")
    #np.savetxt("y_test_inverse.csv", y_test_inverse, delimiter=",")
    print("Inverse-scaled predictions and actual test data saved to 'y_pred_inverse.csv' and 'y_test_inverse.csv'.")


    return y_pred_inverse, y_test_inverse



# Main function to execute the LSTM model or grid search
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_search", help="Do gridsearch if 1", type=int, default=0)
    args = parser.parse_args()

    # Load the data
    try:
        data = pl.read_csv("../../data/CAISO_zone_1_.csv")
    except:
        data = pl.read_csv("/home/unabila/seq2w/2w.csv")

    data_size = 34560 #####################20160
    data = data.limit(n=data_size)

    # Set the target output columns
    target_output_col_name = ["wind_power", "solar_power", "load_power", ]

    # Select the input features (excluding time and output features)
    input_data = data.select(pl.col("*").exclude("time", "solar_power", "wind_power", "load_power"))
    print(input_data.shape)

    # Select the output features
    target_data = data.select(pl.col(target_output_col_name))

    # Get the column index for each target feature
    target_feature_idx = [
        target_data.get_column_index(col) for col in target_output_col_name
    ]

    # Prompt for grid search or single run
    test_var = int(input("Enter 1 for grid search, or anything else for no grid search: "))

    # Preprocess the data (including scaling) and get the scaler
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, _, _, _, scaler_y = pre_fit(input_data, target_data)

    # Set dynamic lookback and lookfor windows
    lookback = 720  # 60
    lookfor = 720   #  change according to case requirement

    # Execute grid search or single run based on input
    if test_var == 1:
        print("Performing grid search...")
        gridsearch_run(input_data, target_data, target_feature_idx)
    else:
        print("Executing single LSTM run...")
        single_lstm_run(input_data, target_data, lookback, lookfor, target_feature_idx, load_model=False, scaler_y=scaler_y)

