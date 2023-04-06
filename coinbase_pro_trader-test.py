import ccxt
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from talib import RSI, BBANDS, MACD, ATR
import optuna
import logging

api_key = ""
api_secret = ""
password = ""

# Define device as a global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exchange = ccxt.coinbasepro({
    'apiKey': api_key,
    'secret': api_secret,
    'password': password,  # Add the password field here
    'enableRateLimit': True,
    'urls': {
        'api': {
            'public': 'https://api-public.sandbox.pro.coinbase.com',
            'private': 'https://api-public.sandbox.pro.coinbase.com',
        }
    }
})

symbol = 'BTC/USD'
timeframe = '1h'

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def create_features(df):
    close = df['close']
    high = df['high']
    low = df['low']

    df['rsi'] = RSI(close)

    upper, middle, lower = BBANDS(close)
    df['upper_bb'] = upper
    df['middle_bb'] = middle
    df['lower_bb'] = lower

    macd, signal, hist = MACD(close)
    df['macd'] = macd
    df['signal'] = signal

    df['atr'] = ATR(high, low, close)

    return df

def preprocess_data(df):
    df = create_features(df)
    df = df.dropna()

    X = df.drop(columns=['timestamp']).values
    y = df['close'].shift(-1).dropna().values
    return X, y

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_lstm_model(X, y):
    n_steps = 20
    X, y = split_sequences(X, n_steps)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size=X.shape[2], hidden_size=50, num_layers=1, batch_first=True)
            self.fc = nn.Linear(50, 1)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.fc(x[:, -1, :])
            return x.squeeze()

    model = LSTM().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    model.train()

    for epoch in range(epochs):
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")

    return model, scaler

def trade_logic(context, order_size):
    in_position, stop_loss, take_profit, current_price, decision = context.values()
    if in_position:
        if current_price <= stop_loss:
            print("Stop-loss triggered. Selling")
            # Execute sell order
            order = exchange.create_market_sell_order(symbol, order_size)
            print("Sell order executed:", order)
            in_position = False
            stop_loss = 0
            take_profit = 0
        elif current_price >= take_profit:
            print("Take-profit triggered. Selling")
            # Execute sell order
            order = exchange.create_market_sell_order(symbol, order_size)
            print("Sell order executed:", order)
            in_position = False
            stop_loss = 0
            take_profit = 0
    else:
        if decision > 0:
            print("Buy signal")
            # Execute buy order
            order_size = 0.01  # Example order size, adjust this accordingly
            order = exchange.create_market_buy_order(symbol, order_size)
            print("Buy order executed:", order)
            in_position = True
            entry_price = current_price
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.04

    return in_position, stop_loss, take_profit

def main():
    data = fetch_data()
    X, y = preprocess_data(data)
    model, scaler = train_lstm_model(X, y)

    in_position = False
    stop_loss = 0
    take_profit = 0
    order_size = 0.01  # Example order size, adjust this accordingly

    while True:
        new_data = fetch_data()
        X_test, _ = preprocess_data(new_data)
        X_test = scaler.transform(X_test)
        X_test = X_test[-1]

        prediction = model(torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)).item()
        current_price = new_data.iloc[-1]['close']

        # Print the relevant variables to track the monitoring
        print(f"Current price: {current_price}")
        print(f"In position: {in_position}")
        print(f"Stop loss: {stop_loss}")
        print(f"Take profit: {take_profit}")

        context = {
            "in_position": in_position,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "current_price": current_price,
            "decision": prediction,
        }

        in_position, stop_loss, take_profit = trade_logic(context, order_size)

        time.sleep(1)  # 1-second sleep

if __name__ == '__main__':
    main()