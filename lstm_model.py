import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import MinMaxScaler

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = 1  # Set default batch size to 100
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

        # Permute the dimensions of the input tensor to match the expected input shape of the LSTM layer
        batch_size = input.size(0)
        input = input.view(batch_size, -1, self.input_dim)

        out, (hidden_state, cell_state) = self.lstm(input, (h0.detach(), c0.detach()))

        out = self.linear(out[:, -1, :])

        return out, hidden_state, cell_state

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # The shape of the tensor is (num_layers, batch_size, hidden_dim)
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        # If using GPU, move the tensors to the GPU
        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        # Return a tuple of the hidden state and cell state tensors
        return (hidden_state, cell_state)

def lstm_predict_next_year_prices():
    # Connect to the database
    db_uri = "sqlite:///oil_prices.db"
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load historical oil prices from the database
    query = text("SELECT price FROM oil_prices ORDER BY date ASC")
    result = session.execute(query)
    prices = [r[0] for r in result]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    # Split the data into training and testing sets
    split_ratio = 0.8
    prices_train, prices_test = prices[0:int(len(prices)*split_ratio), :], prices[int(len(prices)*split_ratio):, :]

    # Create the input and output data for the LSTM model
    look_back = 100

    # Create the input and output data for the LSTM model
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(prices_train), torch.from_numpy(prices_train))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False, drop_last=True)

    test_data = torch.utils.data.TensorDataset(torch.from_numpy(prices_test), torch.from_numpy(prices_test))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, drop_last=True)

    # Train the LSTM model
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    lstm_model = LSTMPredictor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    num_epochs = 100
    train_losses = []

    # Train the model
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            # Convert the training data to PyTorch tensors
            inputs = inputs.type(torch.float)
            targets = targets.type(torch.float)

            # Reset the hidden state at the start of each epoch
            lstm_model.hidden = lstm_model.init_hidden(batch_size=inputs.shape[0])

            # Forward pass
            outputs, _, _ = lstm_model(inputs)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, np.mean(train_losses)))

    # Use the trained model to predict the next year's oil prices
    next_year_prices = []
    last_year_prices = prices[-look_back:]

    for i in range(1000):
        # Convert the last year's prices to a PyTorch tensor
        inputs = torch.from_numpy(last_year_prices[-look_back:]).type(torch.float)
        lstm_model.hidden = lstm_model.init_hidden(batch_size=1)

        # Make a prediction and add it to the list of predictions
        output, _, _ = lstm_model(inputs)
        last_year_prices = np.append(last_year_prices, output.detach().numpy()[0][0])
        next_year_prices.append(output.detach().numpy()[0][0])

    # Denormalize the predicted prices
    next_year_prices = scaler.inverse_transform(np.array(next_year_prices).reshape(-1, 1))

    session.close()

    print(next_year_prices)

    return next_year_prices