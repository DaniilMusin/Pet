import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import time
from datetime import datetime

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model with Holt-Winters component (Helformer)
    """
    def __init__(
        self, 
        input_dim, 
        d_model=64, 
        nhead=4, 
        num_encoder_layers=2,
        dim_feedforward=256, 
        dropout=0.1, 
        activation='gelu',
        use_holt_winters=True
    ):
        super().__init__()
        
        self.use_holt_winters = use_holt_winters
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # LSTM layer for time-series modeling combined with transformer outputs
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer (predicts next price)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Predicting next close price
        )
        
    def forward(self, x, hw_components=None):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        # No need for attention masks as we're using all past context
        x = self.transformer_encoder(x)
        
        # Apply LSTM layer (combines sequential information with transformer)
        x, _ = self.lstm(x)
        
        # If using Holt-Winters decomposition, add the components
        if self.use_holt_winters and hw_components is not None:
            # Assuming hw_components contains level, trend, and seasonal components
            # Shape should be [batch_size, seq_len, 3] where 3 represents level, trend, season
            hw_projection = nn.Linear(hw_components.shape[-1], self.d_model).to(device)
            hw_features = hw_projection(hw_components)
            x = x + hw_features
            
        # Get the last time step for prediction
        last_time_step = x[:, -1, :]
        
        # Project to output
        output = self.output_projection(last_time_step)
        
        return output
    
class AdaptiveSharpeRatioLoss(nn.Module):
    """
    Custom loss function that optimizes for risk-adjusted returns (Sharpe Ratio)
    """
    def __init__(self, risk_free_rate=0.0, lambda_sharpe=0.5, lambda_mse=0.5):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.lambda_sharpe = lambda_sharpe
        self.lambda_mse = lambda_mse
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_returns, true_returns):
        # MSE component
        mse = self.mse_loss(pred_returns, true_returns)
        
        # Sharpe ratio component (simplified calculation for batches)
        # For actual trading you may want a more sophisticated implementation
        mean_return = torch.mean(pred_returns - self.risk_free_rate)
        std_return = torch.std(pred_returns) + 1e-6  # Add small epsilon to avoid division by zero
        sharpe_ratio = mean_return / std_return
        
        # We want to maximize Sharpe ratio, so negate it for minimization
        sharpe_loss = -sharpe_ratio
        
        # Combined loss
        total_loss = self.lambda_mse * mse + self.lambda_sharpe * sharpe_loss
        
        return total_loss

class CryptoDataset(Dataset):
    """
    Dataset class for cryptocurrency data
    """
    def __init__(self, data, seq_length=60, target_horizon=1, hw_decomposition=True):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with crypto data
            seq_length: Length of input sequences
            target_horizon: How many steps ahead to predict
            hw_decomposition: Whether to use Holt-Winters decomposition
        """
        self.data = data
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.hw_decomposition = hw_decomposition
        
        # Normalize features
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Extract features
        self.features = self._preprocess_features()
        
        # Calculate returns (percentage change in close price)
        self.returns = data['Close'].pct_change().fillna(0).values
        
        # Create Holt-Winters decomposition if enabled
        self.hw_components = None
        if hw_decomposition:
            self.hw_components = self._create_hw_decomposition()
            
    def _preprocess_features(self):
        """Preprocess the features for the model"""
        # Select relevant features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add technical indicators
        df = self.data.copy()
        
        # 1. Add moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # 2. Add MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 3. Add RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Add volatility (rolling std dev)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 5. Add price momentum
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Fill NaN values that emerge from calculations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Scale features
        feature_set = df[feature_columns + ['MA_5', 'MA_20', 'MACD', 'MACD_Signal', 'RSI', 'Volatility', 'Momentum']]
        scaled_features = self.feature_scaler.fit_transform(feature_set)
        
        return scaled_features
        
    def _create_hw_decomposition(self):
        """Create Holt-Winters decomposition for time series"""
        # We'll use a simplified approach for now:
        # For each window, we'll fit HW and extract components
        
        # This is computationally expensive during training, so we precompute:
        print("Generating Holt-Winters decomposition...")
        
        # Use statsmodels ExponentialSmoothing for Holt-Winters
        # We'll just decompose the Close price series for simplicity
        close_series = self.data['Close'].values
        
        # Initialize storage for components
        level = np.zeros_like(close_series)
        trend = np.zeros_like(close_series)
        seasonal = np.zeros_like(close_series)
        
        # For full series, we'll use a single HW model
        # In practice, you might want to do this for each window separately
        try:
            # Fit Holt-Winters model (additive with daily seasonal period of 96)
            # 96 represents 24 hours with 15-minute intervals
            hw_model = ExponentialSmoothing(
                close_series,
                trend='add',
                seasonal='add',
                seasonal_periods=96
            ).fit()
            
            # Extract components
            level = hw_model.level
            trend = hw_model.slope
            seasonal = hw_model.season
            
        except Exception as e:
            print(f"Error in Holt-Winters decomposition: {e}")
            print("Using simple moving average decomposition instead.")
            
            # Fallback: simple decomposition
            # Level: 20-period moving average
            for i in range(len(close_series)):
                if i < 20:
                    level[i] = np.mean(close_series[:i+1])
                else:
                    level[i] = np.mean(close_series[i-19:i+1])
            
            # Trend: difference in levels
            trend[1:] = level[1:] - level[:-1]
            
            # Seasonal: original - level
            seasonal = close_series - level
        
        # Combine components into a single array
        components = np.column_stack((level, trend, seasonal))
        
        # Scale components
        hw_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_components = hw_scaler.fit_transform(components)
        
        print("Holt-Winters decomposition complete.")
        return scaled_components
        
    def __len__(self):
        """Return the number of sequences available"""
        return len(self.data) - self.seq_length - self.target_horizon
        
    def __getitem__(self, idx):
        """Get a single sequence"""
        # Extract sequence of features
        features_seq = self.features[idx:idx+self.seq_length]
        
        # Target is future close price
        target_idx = idx + self.seq_length + self.target_horizon - 1
        target = self.data['Close'].iloc[target_idx]
        
        # Scale target
        target_scaled = self.target_scaler.fit_transform(
            np.array(target).reshape(-1, 1)
        ).flatten()[0]
        
        # Prepare HW components if available
        hw_seq = None
        if self.hw_decomposition and self.hw_components is not None:
            hw_seq = self.hw_components[idx:idx+self.seq_length]
            
        return {
            'features': torch.FloatTensor(features_seq),
            'target': torch.FloatTensor([target_scaled]),
            'hw_components': torch.FloatTensor(hw_seq) if hw_seq is not None else None,
            'raw_target': target,
            'idx': idx + self.seq_length
        }

def load_and_prepare_data(file_path):
    """
    Load and prepare crypto data from CSV
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Rename columns based on the header in the screenshot
    df.columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    # Convert timestamps
    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    
    # Set index to datetime
    df.set_index('Open time', inplace=True)
    
    # Ensure numeric values
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'Quote asset volume', 'Number of trades',
                      'Taker buy base asset volume', 'Taker buy quote asset volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # For simplicity, drop the 'Ignore' column
    df = df.drop(columns=['Ignore'])
    
    return df

def train_val_test_split(df, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets
    """
    n = len(df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    return train_data, val_data, test_data

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, 
                patience=10, model_save_path='best_model.pth'):
    """
    Train the model with early stopping
    """
    # Initialize variables for tracking performance
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Create directory for the model if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    
    # Start training
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        
        # Training loop
        for batch in train_loader:
            # Get data
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            hw_components = batch.get('hw_components')
            
            if hw_components is not None:
                hw_components = hw_components.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, hw_components)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                hw_components = batch.get('hw_components')
                
                if hw_components is not None:
                    hw_components = hw_components.to(device)
                
                outputs = model(features, hw_components)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch stats
        time_per_epoch = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Time: {time_per_epoch:.2f}s")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, dataset):
    """
    Evaluate the model on test data
    """
    model.eval()
    predictions = []
    actuals = []
    indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            hw_components = batch.get('hw_components')
            
            if hw_components is not None:
                hw_components = hw_components.to(device)
            
            # Get predictions
            outputs = model(features, hw_components)
            
            # Convert back to original scale
            pred_np = outputs.cpu().numpy().reshape(-1, 1)
            pred_original = dataset.target_scaler.inverse_transform(pred_np).flatten()
            
            # Store results
            predictions.extend(pred_original)
            actuals.extend(batch['raw_target'].numpy())
            indices.extend(batch['idx'].numpy())
    
    # Calculate metrics
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    # Calculate returns-based metrics
    pred_returns = np.diff(predictions) / predictions[:-1]
    actual_returns = np.diff(actuals) / actuals[:-1]
    
    # Sharpe ratio (assuming daily returns, annualized)
    sharpe = np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(365 * 24 * 4)  # 15-min intervals
    
    # Correlation of returns
    corr = np.corrcoef(pred_returns, actual_returns)[0, 1]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': [dataset.data.index[i] for i in indices],
        'Actual': actuals,
        'Predicted': predictions
    })
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Return Correlation: {corr:.4f}")
    
    return results_df

def plot_predictions(results_df, save_path=None):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual')
    plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted')
    plt.title('BTC Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    # Parameters
    file_path = 'btc_15m_data_2018_to_2025.csv'  # Replace with your actual file path
    seq_length = 96  # 24 hours of 15-minute data
    target_horizon = 1  # Predict next 15-minute interval
    batch_size = 64
    num_epochs = 100
    patience = 15
    learning_rate = 0.001
    
    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data(file_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Split data
    train_data, val_data, test_data = train_val_test_split(df)
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CryptoDataset(train_data, seq_length, target_horizon)
    val_dataset = CryptoDataset(val_data, seq_length, target_horizon)
    test_dataset = CryptoDataset(test_data, seq_length, target_horizon)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Calculate input dimension from a sample batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize model
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation='gelu',
        use_holt_winters=True
    ).to(device)
    
    # Define loss function and optimizer
    criterion = AdaptiveSharpeRatioLoss(lambda_sharpe=0.3, lambda_mse=0.7)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model_checkpoints_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_model.pth")
    
    # Train model
    print("Starting training...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=patience,
        model_save_path=model_save_path
    )
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    
    # Evaluate on test data
    print("Evaluating model...")
    results_df = evaluate_model(trained_model, test_loader, test_dataset)
    
    # Plot results
    results_plot_path = os.path.join(save_dir, "predictions.png")
    plot_predictions(results_df, save_path=results_plot_path)
    
    # Save results
    results_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    
    print(f"All results saved to {save_dir}")

if __name__ == "__main__":
    main()