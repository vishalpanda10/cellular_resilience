import torch
import torch.nn as nn
import lightning as L


class MLP(L.LightningModule):
    def __init__(
        self,
        loss_fn,
        n_inputs: int = 1,
        n_outputs: int = 1,
        hidden_layers: list = [128, 64],
        learning_rate=0.001,
        dropout_rate=0.1,
        weight_decay=1e-5,
        batch_norm=True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.train_log = []

        layers = []
        input_dim = n_inputs

        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            #if batch_norm:
            #    layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, n_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss", loss)
        self.train_log.append(loss.detach().cpu().numpy())
        return loss
