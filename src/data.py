"""
Carga de datos y preprocesado para STS Benchmark.

Contiene:
  - normalize_score / denormalize_score: conversión [0, 5] <-> [0, 1]
  - STSCollateFn: tokeniza pares de frases (cross-encoder)
  - STSDataModule: LightningDataModule con train/val/test
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from src.config import SCORE_MIN, SCORE_MAX, DATASET_NAME


def normalize_score(score: float) -> float:
    """Convierte score de [0, 5] a [0, 1] para entrenar con MSELoss."""
    return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def denormalize_score(score: float) -> float:
    """Convierte score de [0, 1] a [0, 5] para devolverlo en su escala original."""
    return score * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


class STSCollateFn:
    """
    Tokeniza pares de frases para el cross-encoder y devuelve los labels
    normalizados a [0, 1].
    """

    def __init__(self, tokenizer_name: str, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch):
        s1 = [ex["sentence1"] for ex in batch]
        s2 = [ex["sentence2"] for ex in batch]
        scores = [normalize_score(ex["score"]) for ex in batch]

        encoded = self.tokenizer(
            list(zip(s1, s2)),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(scores, dtype=torch.float32)
        return encoded, labels


class STSDataModule(pl.LightningDataModule):
    """LightningDataModule del STS Benchmark para el cross-encoder."""

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = load_dataset(DATASET_NAME)
        self.train = ds["train"]
        self.val = ds["validation"]
        self.test = ds["test"]

    def _dataloader(self, split, shuffle=False):
        return DataLoader(
            split,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=STSCollateFn(self.tokenizer_name, self.max_length),
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val)

    def test_dataloader(self):
        return self._dataloader(self.test)
