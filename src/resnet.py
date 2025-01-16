import torch
import torch.nn as nn
import lightning as L


class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features)
        )
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        return x + residual


class ResNetRegression(L.LightningModule):
    def __init__(
        self,
        loss_fn,
        n_inputs: int = 1,
        n_outputs: int = 1,
        hidden_layers: list = [128, 128],
        dropout_rate=0.1,
        learning_rate=0.001,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.train_log = []
        layers = []
        input_dim = n_inputs

        for h_dim in hidden_layers:
            layers.append(ResNetBlock(input_dim, h_dim, dropout_rate))
            input_dim = h_dim

        self.output_layer = nn.Linear(input_dim, n_outputs)
        layers.append(self.output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss", loss)
        self.train_log.append(loss.detach().cpu().numpy())
        return loss
