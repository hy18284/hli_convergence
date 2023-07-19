from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
import wandb


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_labels: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_state):
        logits = self.dropout(hidden_state)
        logits = self.linear(hidden_state)
        return logits


class FELDClassifier(LightningModule):
    def __init__(
        self,
        model_path: str,
        personality: bool,
        emotion: bool,
        sentiment: bool,
        lr: float,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.personality = personality
        self.emotion = emotion
        self.sentiment = sentiment
        self.lr = lr

        if self.personality:
            self.personality_head = ClassificationHead(
                hidden_dim=self.model.config.hidden_size,
                num_labels=5,
                dropout_rate=self.model.config.hidden_dropout_prob,
            )
        
        if self.emotion:
            self.emotion_head = ClassificationHead(
                hidden_dim=self.model.config.hidden_size,
                num_labels=7,
                dropout_rate=self.model.config.hidden_dropout_prob,
            )
            
        if self.sentiment:
            self.sentiment_head = ClassificationHead(
                hidden_dim=self.model.config.hidden_size,
                num_labels=3,
                dropout_rate=self.model.config.hidden_dropout_prob,
            )
            
    def _shared_step(self, pair) -> STEP_OUTPUT:
        batch, labels = pair
        output = self.model(**batch)

        loss = 0.0
        output_dict = dict()

        if self.personality:
            personality_logits = self.personality_head(
                output.pooler_output,
            )
            personality_loss = F.binary_cross_entropy_with_logits(
                personality_logits.view(-1),
                target=labels['personality'].view(-1),
                reduction='mean',
            )
            loss += personality_loss
            output_dict['personality_loss'] = personality_loss
            output_dict['personality_logits'] = personality_logits

        if self.emotion:
            emotion_logits = self.emotion_head(
                output.pooler_output,
            )
            emotion_loss = F.cross_entropy(
                emotion_logits,
                target=labels['emotions'][:, -1].view(-1),
                reduction='mean',
            )
            loss += emotion_loss
            output_dict['emotion_loss'] = emotion_loss
            output_dict['emotion_logits'] = emotion_logits

        if self.sentiment:
            sentiment_logits = self.sentiment_head(
                output.pooler_output,
            )
            sentiment_loss = F.cross_entropy(
                sentiment_logits,
                target=labels['sentiments'][:, -1].view(-1),
                reduction='mean',
            )
            loss += sentiment_loss
            output_dict['sentiment_loss'] = sentiment_loss
            output_dict['sentiment_logits'] = sentiment_logits
        
        output_dict['total_loss'] = loss
        return output_dict
    
    def _make_logging_values(self, output, labels):
        logging_values = dict()
        if self.personality:
            personality_pred = output['personality_logits'].view(-1) > 0.5
            personality_pred = personality_pred.to(torch.long)
            personality_labels = labels['personality'].view(-1) > 0.5
            personality_labels = personality_labels.to(torch.long)
            acc = accuracy_score(personality_labels.tolist(), personality_pred.tolist())
            f1 = f1_score(personality_labels.tolist(), personality_pred.tolist())
            recall = recall_score(personality_labels.tolist(), personality_pred.tolist())
            precision = precision_score(personality_labels.tolist(), personality_pred.tolist())
            logging_values['personality/acc'] = acc
            logging_values['personality/f1'] = f1
            logging_values['personality/recall'] = recall
            logging_values['personality/precision'] = precision
        
        if self.emotion:
            emotion_pred = output['emotion_logits'].view(-1, 7)
            emotion_pred = torch.argmax(emotion_pred, dim=1).view(-1)
            emotion_labels = labels['emotions'][:, -1].reshape(-1)

            acc = accuracy_score(emotion_labels.tolist(), emotion_pred.tolist())
            f1 = f1_score(emotion_labels.tolist(), emotion_pred.tolist(), average='macro')
            recall = recall_score(emotion_labels.tolist(), emotion_pred.tolist(), average='macro')
            precision = precision_score(emotion_labels.tolist(), emotion_pred.tolist(), average='macro')
            logging_values['emotion/acc'] = acc
            logging_values['emotion/f1'] = f1
            logging_values['emotion/recall'] = recall
            logging_values['emotion/precision'] = precision

        if self.sentiment:
            sentiment_pred = output['sentiment_logits'].view(-1, 3)
            sentiment_pred = torch.argmax(sentiment_pred, dim=1).view(-1)
            sentiment_labels = labels['sentiments'][:, -1].reshape(-1)

            acc = accuracy_score(sentiment_labels.tolist(), sentiment_pred.tolist())
            f1 = f1_score(sentiment_labels.tolist(), sentiment_pred.tolist(), average='macro')
            recall = recall_score(sentiment_labels.tolist(), sentiment_pred.tolist(), average='macro')
            precision = precision_score(sentiment_labels.tolist(), sentiment_pred.tolist(), average='macro')
            logging_values['sentiment/acc'] = acc
            logging_values['sentiment/f1'] = f1
            logging_values['sentiment/recall'] = recall
            logging_values['sentiment/precision'] = precision
        
        logging_values['loss'] = torch.mean(output['total_loss'])
        logging_values['personality/loss'] = torch.mean(output['personality_loss'])
        logging_values['emotion/loss'] = torch.mean(output['emotion_loss'])
        logging_values['sentiment/loss'] = torch.mean(output['sentiment_loss'])

        return logging_values
    
    def training_step(self, pair, idx) -> STEP_OUTPUT:
        output = self._shared_step(pair)
        logging_values = self._make_logging_values(output, pair[1])
        
        for key in list(logging_values.keys()):
            logging_values[f'train/{key}'] = logging_values.pop(key)
        logging_values['trainer/global_step'] = self.global_step
        wandb.log(logging_values)
        
        return output['total_loss']

    def validation_step(self, pair, idx) -> STEP_OUTPUT:
        batch, labels = pair
        outputs = self._shared_step(pair)
        self.labels.append(labels)
        self.outputs.append(outputs)
    
    def on_validation_epoch_start(self) -> None:
        self.labels = []
        self.outputs = []

    def on_validation_epoch_end(self) -> None:
        combined_labels = dict()
        for key in self.labels[0].keys():
            items = [label[key] for label in self.labels]
            if items[0].dim() != 0:
                items = torch.cat(items, dim=0)
            else:
                items = torch.stack(items, dim=0)
            combined_labels[key] = items

        combined_outputs = dict()
        for key in self.outputs[0].keys():
            items = [output[key] for output in self.outputs]
            if items[0].dim() != 0:
                items = torch.cat(items, dim=0)
            else:
                items = torch.stack(items, dim=0)
            combined_outputs[key] = items

        logging_values = self._make_logging_values(combined_outputs, combined_labels)
        for key in list(logging_values.keys()):
            logging_values[f'val/{key}'] = logging_values.pop(key)

        logging_values['trainer/global_step'] = self.global_step
        wandb.log(logging_values)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), self.lr)