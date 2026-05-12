"""
Modelo del Experimento 4: Cross-encoder con LoRA.

Es la arquitectura ganadora del proyecto (Pearson 0.8671 en test).

Diferencias frente al Exp 1 (full fine-tuning):
  - Todos los pesos del backbone congelados.
  - Se inyectan matrices LoRA en las capas query/value de la atención.
  - <0.3% de parámetros entrenables que en el Exp 1.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from peft import get_peft_model, LoraConfig
from torchmetrics import PearsonCorrCoef


class STSRegressorLoRA(pl.LightningModule):
    """
    Cross-encoder RoBERTa + LoRA para regresión de similitud semántica.

    Recibe el par de frases concatenado y predice un score continuo en [0, 1]
    (luego se desnormaliza a [0, 5] al servir la predicción).
    """

    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        optimizer_params: dict,
        pooling: str = "mean",
    ):
        super().__init__()
        # save_hyperparameters NO incluye lora_config porque no es serializable
        # de forma directa por Lightning. Lo guardamos manualmente al final.
        self.save_hyperparameters(ignore=["lora_config"])

        # Backbone RoBERTa
        base_model = AutoModel.from_pretrained(model_name)
        # Inyectamos LoRA: congela el backbone y añade matrices de bajo rango
        self.model = get_peft_model(base_model, lora_config)

        # Cabeza de regresión
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, 1),
        )

        assert pooling in ["cls", "mean"], f"Pooling no soportado: {pooling}"
        self.pooling = pooling
        self.optimizer_params = optimizer_params
        self.loss_fn = nn.MSELoss()
        self.pearson = PearsonCorrCoef()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            # Mean pooling ignorando padding
            mask = attention_mask.float()
            mean_coeffs = mask / mask.sum(dim=1, keepdim=True)
            pooled = torch.einsum("bld,bl->bd", last_hidden, mean_coeffs)

        score = self.regressor(pooled).squeeze(-1)  # (B,)
        return torch.sigmoid(score)  # [0, 1]

    def _step(self, batch):
        encoded, labels = batch
        preds = self(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded.get("token_type_ids"),
        )
        loss = self.loss_fn(preds, labels)
        pearson = self.pearson(preds, labels)
        return loss, pearson

    def training_step(self, batch, batch_idx):
        loss, pearson = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_pearson", pearson, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pearson = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pearson", pearson, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pearson = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_pearson", pearson)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_params)
