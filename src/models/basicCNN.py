from types import SimpleNamespace
from torch import nn

# contains available activation functions, TODO: move this to config file
act_fn_by_name = {"tanh": nn.Tanh,
                  "relu": nn.ReLU,
                  "leakyrelu": nn.LeakyReLU,
                  "gelu": nn.GELU}


class CNNBlock(nn.Module):
    """
    Implemetation of single basic CNN block
    Contains:
        convolutianal layer with 3x3 kernel
        batch normalization
        activation function - i.e. relu
    """

    def __init__(self, input_channels, output_channels, act_fn):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              kernel_size=(3, 3),
                              out_channels=output_channels)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.act_fn = act_fn

    def forward(self, batch):
        output = self.conv(batch)
        output = self.bn(output)
        output = self.act_fn(output)
        return output


class basicCNN(nn.Module):
    """
    Implementation of simple CNN
    Contains:
        CNNBlock layers
        MaxPool layers
        Fully connected layers
    """

    def __init__(self, n_classes, act_fn_name="relu"):
        """
        Inputs:
            n_classes - number of target classes
            act_fn_name - name of activation function
        """
        super(basicCNN, self).__init__()

        self.hparams = SimpleNamespace(
            n_classes=n_classes,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name]
        )

        self.network = nn.Sequential(
            CNNBlock(input_channels=3, output_channels=32, act_fn=self.hparams.act_fn),
            CNNBlock(input_channels=32, output_channels=32, act_fn=self.hparams.act_fn),
            CNNBlock(input_channels=32, output_channels=32, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            CNNBlock(input_channels=3, output_channels=32, act_fn=self.hparams.act_fn),
            CNNBlock(input_channels=32, output_channels=32, act_fn=self.hparams.act_fn),
            CNNBlock(input_channels=32, output_channels=32, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        self.fc = nn.Linear(in_features=256, out_features=self.hparams.n_classes)

    def forward(self, batch):
        output = self.network(batch)
        output = self.fc(batch)
        return output
