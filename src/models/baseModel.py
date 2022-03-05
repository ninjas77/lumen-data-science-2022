import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from basicCNN import basicCNN

# contains available models, TODO: move this to a config file
model_dict = {"basicCNN": basicCNN}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'


# noinspection PyAbstractClass,PyUnusedLocal
class BaseModel(LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - name of the model architecture - currently implemented: basicCNN
            model_hparams - dictionary of model hyperparameters, i.e. activation function
            optimizer_name - name of optimizer - currently implemented: Adam, SGD
            optimizer_hparams - dictionary of optimizer hyperparameters - i.e. learning rate
        """
        super(BaseModel, self).__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        loss = self.loss(predictions, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        return optimizer
