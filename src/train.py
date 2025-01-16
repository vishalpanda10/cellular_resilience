import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np

def train(
    model_class,
    X_train,
    y_train,
    X_test,
    y_test,
    loss_fn=torch.nn.MSELoss(),
    learning_rate=0.05,
    max_epochs=100,
    batch_size=None,
    early_stopping=True,  
    patience=10,
    **model_params,
):
    if batch_size is None:
        batch_size = X_train.shape[0]

    X_train_t = torch.Tensor(X_train).to("cuda")
    y_train_t = torch.Tensor(y_train).to("cuda")
    X_test_t = torch.Tensor(X_test).to("cuda")
    y_test_t = torch.Tensor(y_test).to("cuda")

    dataset_train = TensorDataset(X_train_t, y_train_t)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(X_test_t, y_test_t)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = model_class(
        loss_fn=loss_fn,
        n_inputs=X_train.shape[1],
        n_outputs=y_train.shape[1],
        learning_rate=learning_rate,
        **model_params,
    )
    model = model.to("cuda")

    #callbacks = []
    #if early_stopping:
    #    early_stopping = EarlyStopping(
    #        monitor="val_loss",
    #        patience=patience,
    #        mode="min",
    #        verbose=True
    #    )
    #    callbacks.append(early_stopping)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices= "auto" #auto, [1, 2]
        #callbacks=callbacks
    )
    trainer.fit(model, dataloader_train, dataloader_test)

    if hasattr(model, "output_layer"):
        weights = np.append(
            model.output_layer.bias.detach().cpu().numpy(),
            model.output_layer.weight.detach().cpu().numpy(),
        )
    elif hasattr(model, "output_weights"):
        weights = model.output_weights.detach().cpu().numpy()
    else:
        weights = None

    return {
        "model": model,
        "weights": weights,
        "train_log": model.train_log,
    }
