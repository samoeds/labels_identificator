import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Add pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Placeholder for LSTM. Input size will be dynamically set later.
        self.lstm = None

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        batch_size, channels, height, width = x.size()

        # Calculate LSTM input size dynamically
        if self.lstm is None:
            input_size = channels * height
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=False)

        x = x.permute(3, 0, 1, 2)  # Change to (width, batch_size, channels, height)
        x = x.view(width, batch_size, -1)  # Flatten channels and height into one dimension

        x, _ = self.lstm(x)  # Pass through LSTM
        return x
