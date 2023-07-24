import csv
from typing import (
    List,
    Dict,
    Any,
)
import re
from collections import defaultdict

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
import torch
from transformers import AutoTokenizer


class FriendsPersona(Dataset):
    trait_2_label = {
        'cAGR': 0,
        'cCON': 1,
        'cEXT': 2,
        'cOPN': 3,
        'cNEU': 4,
    }    

    def __init__(
        self, 
        mode: str,
        path: str = 'data/friends-personality.csv',
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
            rows = csv.reader(f, delimiter=',')
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
        print(sample)
        personality = [float(item) for item in sample[-5:]]
        utterances = sample[2].split('<br><br>')[1:]

        speakers = [re.search(r'(?<=<b>).*(?=</b>)', utterance) for utterance in utterances]
        speakers = [speaker.group(0) if speaker is not None else None for speaker in speakers]

        count = 0
        speaker_order = {}
        for speaker in speakers:
            if speaker is None:
                continue
            if speaker not in speaker_order:
                speaker_order[speaker] = count
                count += 1

        target_idx = speaker_order[sample[3]]
        utterances = [
            re.sub(r'<b>.*</b>', f'speaker{speaker_order[speaker]}', utterance)
            if speaker is not None else utterance
            for utterance, speaker in zip(utterances, speakers)
        ]

        utterances = f'Target speaker is speaker{target_idx}. ' + ' '.join(utterances)

        return {
            'personality': personality,
            'utterances': utterances,
        }
    
    def __len__(self):
        return len(self.data)


class FriendsPersonaCollator:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, samples: List[Dict[str, Any]]):
        input_ids = [sample['utterances'] for sample in samples]
        batch = self.tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True)

        labels = dict()
        labels['personality'] = torch.tensor([sample['personality'] for sample in samples])

        return batch, labels


class FriendsPersonaDatamodule(LightningDataModule):
    def __init__(
        self, 
        tokenizer_path: str,
        path: str = 'data/friends-personality.csv',
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
            self.train = FriendsPersona(
                mode='train',
                path=self.path,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
            self.val = FriendsPersona(
                mode='val',
                path=self.path,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
        else:
            self.test = FriendsPersona(
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
            collate_fn=FriendsPersonaCollator(self.tokenizer_path),
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=FriendsPersonaCollator(self.tokenizer_path)
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=FriendsPersonaCollator(self.tokenizer_path)
        )