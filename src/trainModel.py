import pytorch_lightning as pl
import torch
from models.baseModel import BaseModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_model(model_name, **kwargs):
    """
    Inputs:
        model_name - name to be passed to baseModel constructor
        save_name - name for model save file
    """
    trainer = pl.Trainer(default_root_dir='../models/',
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=100)

    model = BaseModel(model_name=model_name, **kwargs)

    # TODO: implement training and validation dataloaders
    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result


def main():
    train_model(model_name="basicCNN",
                model_hparams={"num_classes": 4,
                               "act_fn_name": "relu"},
                optimizer_name="Adam",
                optimizer_hparams={"lr": 1e-3})


if __name__ == "__main__":
    main()
