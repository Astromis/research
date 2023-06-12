import torch
from typing import List, Tuple

from torch.utils.data import Dataset


class TextsLabelsDataset(Dataset):
    """Regular dataset for BERT training and inferencing."""
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], int]:
        item = {k: v[idx] for k, v in self.texts.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    
class TextsLabelsDataCollator:
    """Collator based on tokenizer"""
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, int]]):
        texts, labels = [sample[0] for sample in batch], [sample[1] for sample in batch]
        batch = self.tokenizer.batch_encode_plus(
            *texts,
            padding='longest',
            max_length=512,
            #pad_to_multiple_of=100,
            truncation=True,
            return_tensors='pt'
        )
        if isinstance(labels[0],str) is not True:
            batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch

    
class CoherenceDataset(Dataset):
    """This class represents a coherence dataset."""
    def __init__(self, samples: List[Tuple[List[str], List[str], int]]) -> None:
        """Init.

        Args:
            samples (List[Tuple[List[str], List[str], int]]): The list of tuples that contain
            two texts and binary label indicating whether two texts are coherent.
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str], int]:
        s = self.samples[idx]
        return s[0], s[1], s[2], 

class CoherenceCollatorRNN:
    """The collator class that convert data into format appropriate for LSTM."""

    def __init__(self, tokenizer, max_len:int, is_siames: bool) -> None:
        """Init.

        Args:
            tokenizer (_type_): The tokenizer instance that resamble transformers.AutoTokenizer `encode` method.
            max_len (int): The maximum sequence length.
            is_siames (bool): Whether the used network is seames or not.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_siames = is_siames
        
        
    def __call__(self, batch: List[Tuple[str, str, int]]):
        batch_ = {}
        text_1, text_2, labels = [sample[0] for sample in batch], [sample[1] for sample in batch], [sample[2] for sample in batch]
        
        text_1 = [self.tokenizer.encode(text, add_special_tokens=False, padding="max_length", max_length=self.max_len, truncation="longest_first") for text in text_1]
        text_2 = [self.tokenizer.encode(text, add_special_tokens=False, padding="max_length", max_length=self.max_len, truncation="longest_first") for text in text_2]
        
        # normaly, torch LSTM takes (seq_len, batch_size, input_size) so transpose
        if self.is_siames:
            batch_["premise"] = torch.tensor(text_1, dtype=torch.long).transpose(0,1)
            batch_["hypothesis"] = torch.tensor(text_2, dtype=torch.long).transpose(0,1)
        else:
            prem = torch.tensor(text_1, dtype=torch.long)
            hypo = torch.tensor(text_2, dtype=torch.long)
            batch_["text"] = torch.cat((prem, hypo), dim=1).transpose(0,1)
        if 'None' not in labels:
            labels = list(map(lambda x: int(float(x)), labels))
            batch_['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch_

class PairedTextDataset(Dataset):
    """Yet another variant of coherence dataset for SentenceBERT."""

    def __init__(self, premise: List[str], hypothesis: List[str], labels: List[int]) -> None:
        self.premise = premise
        self.hypothesis = hypothesis
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], int]:
        item = {}
        item["text_pair"] = (self.premise[idx], self.hypothesis[idx])
        item["label"] = torch.tensor([self.labels[idx]])
        return item
    
    