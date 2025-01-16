import torch
import torch.nn as nn
import lightning as L


class ElasticLinear(L.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 1, n_outputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, n_outputs)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()
        
        self.log("loss", loss)
        self.train_log.append(loss.detach().cpu().numpy())
        return loss
    
