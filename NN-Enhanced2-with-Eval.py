import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf

# --- 1. Robust Loss Class ---
class RobustLoss(nn.Module):
    def __init__(self, alpha=0.05, delta=1.0):
        super().__init__()
        self.alpha = alpha  # Tail probability for CVaR
        self.delta = delta  # Huber loss threshold
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, returns=None):
        # Huber loss component
        huber_loss = F.huber_loss(y_pred, y_true, reduction='mean', delta=self.delta)
        
        if returns is not None and len(returns) > 0:  # CVaR component (optional)
            # Ensure returns are 1D for quantile calculation
            returns_1d = returns.squeeze()
            if returns_1d.dim() == 0: # Handle single element tensor
                returns_1d = returns_1d.unsqueeze(0)

            # Calculate VaR (Value at Risk)
            # torch.quantile requires input to be 1D
            var = torch.quantile(returns_1d, self.alpha)
            
            # Identify tail losses (returns below VaR)
            tail_losses = returns_1d[returns_1d <= var]
            
            # Calculate CVaR penalty
            cvar_penalty = -torch.mean(tail_losses) if len(tail_losses) > 0 else 0.0
            
            return huber_loss + 0.5 * cvar_penalty
        return huber_loss

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Data Collection and Preprocessing ---
def get_and_preprocess_data(tickers, start_date, end_date, lookback_period):
    """
    Downloads data, calculates weekly average daily returns, normalizes them,
    and adds rolling skewness, kurtosis, VIX, and rolling volume as features.
    """
    print("Step 1: Downloading data...")
    # Download Close prices for returns, and Volume for rolling volume feature
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    close_prices = data['Close']
    volumes = data['Volume']

    close_prices = close_prices.dropna(axis=0, how='any')
    close_prices = close_prices.dropna(axis=1, how='all')
    volumes = volumes.dropna(axis=0, how='any')
    volumes = volumes.dropna(axis=1, how='all')

    if close_prices.empty:
        raise ValueError("No valid close price data retrieved after dropping NaNs. Check tickers/dates.")
    if volumes.empty:
        print("Warning: No valid volume data retrieved. Proceeding without volume features.")
        volumes = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns) # Placeholder

    print("Step 2: Calculating daily returns...")
    daily_returns = close_prices.pct_change().dropna()

    print("Step 3: Resampling to weekly average daily returns and volumes...")
    weekly_avg_returns = daily_returns.resample('W').mean().dropna()
    weekly_avg_volume = volumes.resample('W').mean().dropna()

    if weekly_avg_returns.empty:
        raise ValueError("No weekly average returns generated. Check data range.")

    print("Step 4: Calculating rolling skewness, kurtosis, and average volume...")
    # Calculate rolling skewness and kurtosis for each asset based on weekly returns
    rolling_skew = weekly_avg_returns.rolling(window=lookback_period).skew().dropna()
    rolling_kurtosis = weekly_avg_returns.rolling(window=lookback_period).kurt().dropna()
    
    # Calculate rolling average volume for each asset
    rolling_avg_volume = weekly_avg_volume.rolling(window=lookback_period).mean().dropna()

    # Download VIX data (market volatility index)
    print("Step 4.1: Downloading VIX data...")
    vix_data = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)['Close']
    vix_weekly = vix_data.resample('W').mean().dropna()
    vix_weekly.name = 'VIX' # Rename for concatenation

    print("Step 5: Aligning all features and concatenating...")
    # Align indices across all generated dataframes
    # Start with the index of the most restrictive rolling calculation (which is usually `lookback_period`)
    # and then intersect with all others.
    
    # Ensure all dataframes have the same columns as original tickers for consistency in rolling calcs
    # and then align their indices.
    
    # First, align the primary dataframes (returns, skew, kurtosis, volume)
    aligned_dfs = [weekly_avg_returns, rolling_skew, rolling_kurtosis, rolling_avg_volume]
    
    # Find the common index across all these asset-specific time series
    common_index_assets = weekly_avg_returns.index
    for df in [rolling_skew, rolling_kurtosis, rolling_avg_volume]:
        common_index_assets = common_index_assets.intersection(df.index)

    weekly_avg_returns_aligned = weekly_avg_returns.loc[common_index_assets]
    rolling_skew_aligned = rolling_skew.loc[common_index_assets]
    rolling_kurtosis_aligned = rolling_kurtosis.loc[common_index_assets]
    rolling_avg_volume_aligned = rolling_avg_volume.loc[common_index_assets]

    # Now align VIX with the common index of asset data
    vix_weekly_aligned = vix_weekly.loc[vix_weekly.index.intersection(common_index_assets)]
    
    # Ensure VIX is a DataFrame for concatenation
    if isinstance(vix_weekly_aligned, pd.Series):
        vix_weekly_aligned = vix_weekly_aligned.to_frame()
    
    # Rename columns for concatenation to avoid conflicts
    rolling_skew_aligned.columns = [col + '_skew' for col in rolling_skew_aligned.columns]
    rolling_kurtosis_aligned.columns = [col + '_kurt' for col in rolling_kurtosis_aligned.columns]
    rolling_avg_volume_aligned.columns = [col + '_vol' for col in rolling_avg_volume_aligned.columns]
    
    # Concatenate all features
    all_features = pd.concat([
        weekly_avg_returns_aligned, 
        rolling_skew_aligned, 
        rolling_kurtosis_aligned, 
        rolling_avg_volume_aligned,
        vix_weekly_aligned # VIX is a single column
    ], axis=1)

    # Drop any remaining NaNs that might result from final concatenation misalignment
    all_features = all_features.dropna()
    weekly_avg_returns_aligned = weekly_avg_returns_aligned.loc[all_features.index] # Align returns for MVO targets

    if all_features.empty:
        raise ValueError("No valid features generated after aligning and dropping NaNs. Adjust lookback period or data range.")

    print("Step 6: Normalizing all features...")
    scaler = StandardScaler()
    normalized_all_features = pd.DataFrame(
        scaler.fit_transform(all_features),
        index=all_features.index,
        columns=all_features.columns
    )
    
    return data, weekly_avg_returns_aligned, normalized_all_features, scaler

# --- MVO Functions ---
def robust_mean_variance_optimization(returns_df_window, target_return=None):
    """MVO with Ledoit-Wolf shrinkage covariance"""
    mu = returns_df_window.mean().values
    lw = LedoitWolf().fit(returns_df_window)
    Sigma = lw.covariance_
    
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(verbose=False, solver='ECOS', max_iters=5000)
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            weights[weights < 0] = 0
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            else:
                return np.full(n, 1/n)
            return weights
        else:
            return None
    except Exception as e:
        return None

def find_max_sharpe_portfolio(returns_df_window, risk_free_rate_weekly=0.0):
    """
    Finds the portfolio with the maximum Sharpe Ratio on the efficient frontier
    for a given window of returns. Includes fallback to GMV or Equal Weights.
    """
    mu = returns_df_window.mean().values
    Sigma = returns_df_window.cov().values
    n = len(mu)

    min_mu = mu.min()
    max_mu = mu.max()
    
    if min_mu == max_mu:
        target_returns_range = np.array([min_mu])
    else:
        upper_search_bound = max(max_mu + (max_mu - min_mu) * 0.15, risk_free_rate_weekly + 1e-6)
        target_returns_range = np.linspace(min_mu, upper_search_bound, 100)
        target_returns_range = target_returns_range[target_returns_range >= risk_free_rate_weekly]
        
        if len(target_returns_range) == 0:
            target_returns_range = np.array([risk_free_rate_weekly + 1e-6])

    max_sharpe = -np.inf
    optimal_weights_sharpe = None

    for r_target in target_returns_range:
        weights = robust_mean_variance_optimization(returns_df_window, target_return=r_target)
        if weights is not None:
            portfolio_return = np.dot(weights, mu)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))

            if portfolio_volatility > 0:
                sharpe = (portfolio_return - risk_free_rate_weekly) / portfolio_volatility
                if sharpe > max_sharpe:
                    max_sharpe = sharpe
                    optimal_weights_sharpe = weights

    # --- Fallback Mechanism ---
    if optimal_weights_sharpe is None:
        gmv_weights = robust_mean_variance_optimization(returns_df_window, target_return=None)
        if gmv_weights is not None:
            return gmv_weights
        else:
            return np.full(n, 1/n)

    return optimal_weights_sharpe

# --- Neural Network Model ---
class RobustPortfolioNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Increased complexity for new features (more input dimensions)
        self.fc1 = nn.Linear(input_size, 512) # Increased from 256
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)        # Increased from 128
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)        # Increased from 64
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)      
        self.fc4 = nn.Linear(128, output_size) # Adjusted input size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.dropout3(self.fc3(x)))
        x = self.softmax(self.fc4(x))
        return x

# --- Data Preparation ---
def prepare_nn_data(normalized_all_features_df, weekly_avg_returns_aligned, lookback_period=52, validation_split=0.1, risk_free_rate_weekly=0.0):
    """
    Prepares training data for NN using expanded features and Max Sharpe targets.
    """
    print(f"\nPreparing NN data with lookback_period={lookback_period} for Max Sharpe targets (with higher moments, VIX, and volume)...")
    
    num_assets = weekly_avg_returns_aligned.shape[1]
    
    sequences = []
    targets = []
    successful_optimizations = 0
    
    total_windows = len(normalized_all_features_df) - lookback_period + 1

    for i in range(lookback_period, len(normalized_all_features_df) + 1):
        features_window = normalized_all_features_df.iloc[i - lookback_period:i]
        returns_window_for_mvo = weekly_avg_returns_aligned.iloc[i - lookback_period:i]
        
        optimal_weights = find_max_sharpe_portfolio(returns_window_for_mvo, risk_free_rate_weekly)
        
        if optimal_weights is not None:
            sequences.append(features_window.values.flatten())
            targets.append(optimal_weights)
            successful_optimizations += 1
        
        if (i - lookback_period + 1) % 50 == 0 or (i == len(normalized_all_features_df)):
            print(f"  Processed {i - lookback_period + 1}/{total_windows} windows. Successful optimizations: {successful_optimizations}")

    if not sequences:
        raise ValueError("No data sequences generated for NN. This should not happen with current fallbacks. Check data range or fundamental issues.")

    X = np.array(sequences)
    y = np.array(targets)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_split, shuffle=False)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=52, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=52, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=52, shuffle=False)

    print(f"  Total NN samples: {len(sequences)}")
    print(f"  Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    train_val_sequences_count = len(X_train) + len(X_val)

    return train_loader, val_loader, test_loader, num_assets, X_test_tensor, y_test_tensor, train_val_sequences_count

# --- Evaluation Functions ---
def evaluate_portfolio_performance(returns_df, weights, risk_free_rate_weekly=0.0):
    """Evaluates portfolio performance metrics"""
    if weights is None or len(weights) == 0 or np.sum(weights) == 0:
        return {'annualized_return': 0, 'annualized_volatility': 0, 'sharpe_ratio': 0, 'cumulative_return': 0}

    weights = np.array(weights)
    weights[weights < 0] = 0
    if np.sum(weights) == 0:
        return {'annualized_return': 0, 'annualized_volatility': 0, 'sharpe_ratio': 0, 'cumulative_return': 0}
    weights = weights / np.sum(weights)

    portfolio_returns = returns_df.dot(weights)
    annualized_return = portfolio_returns.mean() * 52
    annualized_volatility = portfolio_returns.std() * np.sqrt(52)
    sharpe_ratio = (annualized_return - risk_free_rate_weekly * 52) / annualized_volatility if annualized_volatility > 0 else 0
    cumulative_return = (1 + portfolio_returns).prod() - 1

    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return
    }

def evaluate_model(model, test_loader, actual_test_period_returns_df, risk_free_rate_weekly):
    """Evaluates model performance on the test set."""
    model.eval() # Set model to evaluation mode
    nn_predicted_weights_list = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs) # Get predictions
            nn_predicted_weights_list.extend(outputs.cpu().numpy())
    
    avg_nn_weights = np.mean(nn_predicted_weights_list, axis=0) # Average weights over the test set

    # Evaluate the performance of the portfolio formed by these average weights
    metrics = evaluate_portfolio_performance(actual_test_period_returns_df, avg_nn_weights, risk_free_rate_weekly)
    
    return metrics, avg_nn_weights

# --- Plotting Functions ---
def plot_training_loss(train_losses, val_losses):
    """Plots training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_efficient_frontier(returns_df, risk_free_rate_weekly=0.0):
    """Plots efficient frontier"""
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    
    # Ensure a robust range for target returns
    min_mu = mu.min()
    max_mu = mu.max()
    if min_mu == max_mu:
        target_returns = np.array([min_mu])
    else:
        upper_bound = max_mu + (max_mu - min_mu) * 0.2
        target_returns = np.linspace(min_mu, upper_bound, 50)
        target_returns = target_returns[target_returns >= risk_free_rate_weekly]
        if len(target_returns) == 0: # Fallback if all target returns are too low
            target_returns = np.array([risk_free_rate_weekly + 1e-6]) # Just above risk-free

    risks, rets = [], []
    for r in target_returns:
        w = cp.Variable(len(mu))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                          [cp.sum(w) == 1, w >= 0, mu @ w >= r])
        try:
            prob.solve(verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                risks.append(np.sqrt(w.value.T @ Sigma @ w.value))
                rets.append(mu @ w.value)
        except Exception:
            pass # Skip if solver fails for a specific target return
    
    plt.figure(figsize=(12, 7))
    plt.plot(risks, rets, 'b-', label='Efficient Frontier')
    plt.scatter(returns_df.std(), returns_df.mean(), c='red', marker='o', label='Assets')
    
    # Plot Max Sharpe point on the efficient frontier for the full period
    max_sharpe_weights_full_period = find_max_sharpe_portfolio(returns_df, risk_free_rate_weekly)
    if max_sharpe_weights_full_period is not None:
        port_ret_max_sharpe = np.dot(max_sharpe_weights_full_period, mu)
        port_vol_max_sharpe = np.sqrt(np.dot(max_sharpe_weights_full_period.T, np.dot(Sigma, max_sharpe_weights_full_period)))
        plt.scatter(port_vol_max_sharpe, port_ret_max_sharpe, c='green', marker='*', s=200, label='Max Sharpe Portfolio')

    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Configuration
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT']
    start_date = "2015-01-01"
    end_date = "2024-01-01"
    lookback_period = 104 # Changed to 104 weeks (2 years)
    risk_free_rate_weekly = 0.0001 # Example: ~0.5% annual risk-free rate (0.0001 * 52 weeks)
    
    try:
        # Data preparation
        raw_prices, weekly_returns, normalized_all_features, scaler = \
            get_and_preprocess_data(tickers, start_date, end_date, lookback_period)
        
        # Plot data
        raw_prices.plot(figsize=(12, 6), title="Daily Adjusted Close Prices (2015–2024)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        weekly_returns.plot(figsize=(12, 6), title="Weekly Average Daily Returns")
        plt.xlabel("Week")
        plt.ylabel("Average Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        normalized_all_features.plot(figsize=(12, 6), title="Normalized All Features (Returns, Skew, Kurtosis, Volume, VIX)")
        plt.xlabel("Week")
        plt.ylabel("Normalized Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot Efficient Frontier for the full period
        plot_efficient_frontier(weekly_returns, risk_free_rate_weekly)

        # Prepare NN data
        train_loader, val_loader, test_loader, num_assets, X_test_tensor, y_test_tensor, train_val_sequences_count = prepare_nn_data(
            normalized_all_features, weekly_returns, lookback_period, validation_split=0.15, risk_free_rate_weekly=risk_free_rate_weekly
        )
        
        if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print("Not enough data to create training/testing sets. Adjust lookback period or data range.")
            exit()

        # Input size now includes returns, skew, kurtosis, volume for each asset, and VIX
        # num_assets * 3 (returns, skew, kurtosis) + num_assets (volume) + 1 (VIX) = num_assets * 4 + 1
        input_size = lookback_period * (num_assets * 4 + 1) 
        output_size = num_assets
        model = RobustPortfolioNN(input_size, output_size).to(device)
        print("\nNeural Network Model Architecture:")
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3) # Increased weight_decay
        criterion = RobustLoss(alpha=0.05, delta=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100) 
        
        # Training
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience, epochs_no_improve = 250, 0 
        epochs = 2000 
        
        print("\nStarting NN training...")
        for epoch in range(epochs):
            model.train() # Set model to training mode (dropout active)
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs) # Predicted weights (batch_size, num_assets)

                # Reshape inputs to get current week's returns for loss calculation
                # The input 'inputs' tensor is (batch_size, lookback_period * (num_assets * 4 + 1))
                # We need to extract the last 'num_assets' values that correspond to the actual returns.
                # The order of concatenation in get_and_preprocess_data is returns, skew, kurtosis, volume, VIX.
                # So, the returns are the first 'num_assets' columns.
                
                # Reshape inputs to (batch_size, lookback_period, total_features_per_timestep)
                total_features_per_timestep = num_assets * 4 + 1
                inputs_reshaped_full = inputs.view(inputs.size(0), lookback_period, total_features_per_timestep)
                
                # Extract only the returns part from the last week of the lookback period
                current_week_returns_batch = inputs_reshaped_full[:, -1, :num_assets] # (batch_size, num_assets)
                
                # Calculate portfolio returns for the CVaR component of the loss
                portfolio_returns_for_loss = torch.sum(current_week_returns_batch * outputs, dim=1) # (batch_size,)

                loss = criterion(outputs, targets, portfolio_returns_for_loss)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0) 
            
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval() # Set model to evaluation mode (dropout inactive)
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs) # Get predictions

                    inputs_reshaped_full = inputs.view(inputs.size(0), lookback_period, total_features_per_timestep)
                    current_week_returns_batch = inputs_reshaped_full[:, -1, :num_assets]
                    portfolio_returns_for_loss = torch.sum(current_week_returns_batch * outputs, dim=1)

                    running_val_loss += criterion(outputs, targets, portfolio_returns_for_loss).item() * inputs.size(0) 
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            
            scheduler.step(epoch_val_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
            
            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        print("\nNN Training complete.")
        
        # Plot training
        plot_training_loss(train_losses, val_losses)
        
        # Evaluation
        test_start_idx_in_aligned_df = normalized_all_features.index[lookback_period - 1 + train_val_sequences_count]
        actual_test_period_returns_df = weekly_returns.loc[test_start_idx_in_aligned_df:]


        if actual_test_period_returns_df.empty:
            print("Not enough actual returns for the test period to evaluate. Adjust data range or lookback.")
            exit()

        nn_metrics, avg_nn_weights = evaluate_model(model, test_loader, actual_test_period_returns_df, risk_free_rate_weekly)
        print("\nNN Portfolio Performance (Average Weights on Test Period):")
        print(f"  Annualized Return: {nn_metrics['annualized_return']:.4f}")
        print(f"  Annualized Volatility: {nn_metrics['annualized_volatility']:.4f}")
        print(f"  Sharpe Ratio: {nn_metrics['sharpe_ratio']:.4f}")
        print(f"  Cumulative Return: {nn_metrics['cumulative_return']:.4f}")
        print("  Average NN Predicted Weights:", [f"{w:.4f}" for w in avg_nn_weights])

        # For MVO target, use the average MVO weights from y_test_tensor
        mvo_target_weights_test_period = y_test_tensor.mean(dim=0).cpu().numpy()
        mvo_portfolio_metrics = evaluate_portfolio_performance(actual_test_period_returns_df, mvo_target_weights_test_period, risk_free_rate_weekly)
        print("\nMVO Target Portfolio Performance (Average Weights on Test Period):")
        print(f"  Annualized Return: {mvo_portfolio_metrics['annualized_return']:.4f}")
        print(f"  Annualized Volatility: {mvo_portfolio_metrics['annualized_volatility']:.4f}")
        print(f"  Sharpe Ratio: {mvo_portfolio_metrics['sharpe_ratio']:.4f}")
        print(f"  Cumulative Return: {mvo_portfolio_metrics['cumulative_return']:.4f}")
        print("  Average MVO Target Weights:", [f"{w:.4f}" for w in mvo_target_weights_test_period])
        
        # Plot cumulative returns
        nn_returns_series_for_plot = actual_test_period_returns_df.dot(avg_nn_weights)
        mvo_returns_series_for_plot = actual_test_period_returns_df.dot(mvo_target_weights_test_period)
        
        plt.figure(figsize=(12, 6))
        (1 + nn_returns_series_for_plot).cumprod().plot(label='NN Portfolio')
        (1 + mvo_returns_series_for_plot).cumprod().plot(label='MVO Target Portfolio')
        (1 + actual_test_period_returns_df.mean(axis=1)).cumprod().plot(label='Equal Weight Portfolio')
        plt.title("Cumulative Returns (Test Period)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")


# === NEW SECTION: Rolling Evaluation Comparison ===

def rolling_evaluation(model, test_loader, test_returns_df, risk_free_rate_weekly=0.0):
    model.eval()
    nn_weights_series = []
    mvo_weights_series = []
    actual_returns_series = []
    nn_portfolio_returns = []
    mvo_portfolio_returns = []

    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):
            # NN prediction
            nn_weights = model(features).cpu().numpy()[0]
            nn_weights[nn_weights < 0] = 0
            nn_weights /= np.sum(nn_weights) if np.sum(nn_weights) > 0 else 1

            # MVO prediction
            start_idx = i
            end_idx = i + 52
            returns_window = test_returns_df.iloc[start_idx:end_idx]
            actual_next_returns = test_returns_df.iloc[end_idx] if end_idx < len(test_returns_df) else None

            if actual_next_returns is None or returns_window.shape[0] < 10:
                break

            mvo_weights = find_max_sharpe_portfolio(returns_window, risk_free_rate_weekly)
            if mvo_weights is None:
                mvo_weights = np.ones_like(nn_weights) / len(nn_weights)

            mvo_weights[mvo_weights < 0] = 0
            mvo_weights /= np.sum(mvo_weights) if np.sum(mvo_weights) > 0 else 1

            # Record returns
            r_t = actual_next_returns.values
            nn_r = np.dot(nn_weights, r_t)
            mvo_r = np.dot(mvo_weights, r_t)

            nn_portfolio_returns.append(nn_r)
            mvo_portfolio_returns.append(mvo_r)

    nn_cum_returns = np.cumprod([1 + r for r in nn_portfolio_returns]) - 1
    mvo_cum_returns = np.cumprod([1 + r for r in mvo_portfolio_returns]) - 1

    plt.figure(figsize=(12, 6))
    plt.plot(nn_cum_returns, label="Neural Network Portfolio")
    plt.plot(mvo_cum_returns, label="MVO Portfolio")
    plt.title("Rolling Cumulative Returns: NN vs MVO")
    plt.xlabel("Weeks")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    return nn_cum_returns, mvo_cum_returns
