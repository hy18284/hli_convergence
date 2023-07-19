import csv
from typing import (
    List,
    Dict,
    Any,
)

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
import torch
from transformers import AutoTokenizer



class PELD(Dataset):
    emotion_2_label = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'joy': 3,
        'neutral': 4,
        'sadness': 5,
        'surprise': 6,
    }    
    sentiment_2_label = {
        'positive': 0,
        'neutral': 1,
        'negative': 2,
    }

    def __init__(
        self, 
        mode: str,
        path: str = 'data/Dyadic_PELD.tsv',
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.path = path 
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.mode = mode
        
        with open(self.path) as f:
            self.data = []
            rows = csv.reader(f, delimiter='\t')
            for row in rows:
                self.data.append(row)
            self.data = self.data[1:]

        train, val, test = random_split(
            self.data,
            [1.0 - self.val_ratio - self.test_ratio, self.val_ratio, self.test_ratio],
            generator=torch.Generator().manual_seed(seed),
        )
        if self.mode == 'train':
            self.data = train
        elif self.mode == 'val':
            self.data = val
        elif self.mode == 'test':
            self.data = test
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        personality = eval(sample[2])
        utterances = sample[3:6]
        emotions = [self.emotion_2_label[emotion] for emotion in sample[6:9]]
        sentiments = [self.sentiment_2_label[sentiment] for sentiment in sample[9:12]]

        return {
            'personality': personality,
            'utterances': utterances,
            'emotions': emotions,
            'sentiments': sentiments,
        }
    
    def __len__(self):
        return len(self.data)


class PELDCollator:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def __call__(self, samples: List[Dict[str, Any]]):
        input_ids = [sample['utterances'] for sample in samples]
        input_ids = [
            ' speaker {}: '.join(sample).format(*range(len(sample) - 1))[1:]
            for sample in samples
        ]
        batch = self.tokenizer(input_ids, return_tensors='pt')
        
        labels = dict()
        labels['emotions'] = torch.tensor([sample['emotions'] for sample in samples])
        labels['sentiments'] = torch.tensor([sample['sentiments'] for sample in samples])
        labels['personality'] = torch.tensor([sample['personality'] for sample in samples])

        return batch, labels


class FeldDatamodule(LightningDataModule):
    def __init__(
        self, 
        tokenizer_path: str,
        path: str = 'data/Dyadic_PELD.tsv', 
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        batch_size: int = 12,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.path = path 
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.batch_size = batch_size
    
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train = PELD(
                mode='train',
                path=self.path,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
            self.val = PELD(
                mode='val',
                path=self.path,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
        else:
            self.test = PELD(
                mode='test',
                path=self.path,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=PELDCollator(self.tokenizer_path),
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=PELDCollator(self.tokenizer_path)
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=PELDCollator(self.tokenizer_path)
        )