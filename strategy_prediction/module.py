import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class TransformerClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        lr: float = 1e-5,
        warmup_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.train_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_labels, average="macro")

    def forward(self, texts):
        features = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.encoder(**features)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb

    def _step(self, batch, stage: str):
        texts, labels = batch["context"], batch["label"]
        emb = self(texts)
        logits = self.classifier(emb)

        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        getattr(self, f"{stage}_f1")(preds, labels)
        batch_size = labels.size(0)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_f1",
            getattr(self, f"{stage}_f1"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, min(int(total_steps * self.hparams.warmup_ratio), total_steps - 1))

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }
