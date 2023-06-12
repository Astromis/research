import torch
import numpy as np
import os
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import CoherenceCollatorRNN, CoherenceDataset, PairedTextDataset
from models import Encoder, Attention, RNN, SiamesRNN

def prepare_dataloader(dataset, tokenizer, max_len, batch_size, is_siames):
    """Prepare dataloader with coherence dataset for L

    Args:
        dataset (List[Tuple[List[str], List[str], int]]): The coherence dataset for dataloader.
        tokenizer (_type_): A tokenizer.
        max_len (_type_): The maximum sequence length.
        batch_size (_type_): batch size
        is_siames (bool): Whether the working network is Siames.

    Returns:
        _type_: A dataloader
    """
    dataset = CoherenceDataset(dataset)
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=CoherenceCollatorRNN(tokenizer, max_len, is_siames),
            pin_memory=True
        )
    return dataloader
    
def get_rnn_model(vocab_size, num_classes, embed_dim, hidden_dim, 
                n_layers, is_bidirectional, rnn_type, padding_idx):
    """The RNN model constructor.

    Args:
        vocab_size (int): The vocab size
        num_classes (int): Number of classes to be predicted.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension.
        n_layers (int): Number of LSTM layers.
        is_bidirectional (bool): Whether the RNN will be bidirectional.
        rnn_type (str): Type of RNN (LTSM or GRU)
        padding_idx (int): Padding symbol id in vocabulary.

    Returns:
        _type_: The completed model.
    """
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, max_norm=1)

    encoder = Encoder(embed_dim, hidden_dim, nlayers=n_layers, 
                        dropout=0.1, bidirectional=is_bidirectional, rnn_type=rnn_type)

    attention_dim = hidden_dim if not is_bidirectional else 2*hidden_dim
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = RNN(embedding, encoder, attention, attention_dim, num_classes)
    return model

def get_siames_model(vocab_size, num_classes, embed_dim, hidden_dim, intermidiate_dim,
                n_layers, is_bidirectional, rnn_type, padding_idx):
    """The RNN model constructor.

    Args:
        vocab_size (int): The vocab size
        num_classes (int): Number of classes to be predicted.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension.
        intermidiate_dim (int): The dimention of intermidiate dimension that gather info from LTSM.
        n_layers (int): Number of LSTM layers.
        is_bidirectional (bool): Whether the RNN will be bidirectional.
        rnn_type (str): Type of RNN (LTSM or GRU)
        padding_idx (int): Padding symbol id in vocabulary.

    Returns:
        _type_: The completed model.
    """
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, max_norm=1, scale_grad_by_freq=True)

    encoder = Encoder(embed_dim, hidden_dim, nlayers=n_layers, 
                        dropout=0.1, bidirectional=is_bidirectional, rnn_type=rnn_type)

    attention_dim = hidden_dim if not is_bidirectional else 2*hidden_dim
    
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = SiamesRNN(embedding, encoder, attention, attention_dim, intermidiate_dim, num_classes)
    return model

def train_model(data, model, tokenizer, device, alighment_len,
                 epochs, learning_rate, batch_size, is_siames,
               save_path, divider=None):
    """Train model that was constructed.

    Args:
        data (List[Tuple[List[str], List[str], int]]): Training bare data
        model (_type_): The model to be trained.
        tokenizer (_type_): The tokenizer for preparing the dataloader.
        device (str): The device to train on.
        alighment_len (int): The length of the sequence alighment.
        epochs (int): Number of train epochs.
        learning_rate (float): The learning rate.
        batch_size (int): The batch size.
        is_siames (bool): Whether the network is Siames.
        save_path (str): The path to save a model checkpoints.
        divider (int, optional): Use to determine the model save gap. Defaults to None.
    """
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, )
    loss_avg = []
    train_dataloader = prepare_dataloader(data, tokenizer, alighment_len, 
                                              batch_size, is_siames)
    save_counter = 0
    already_saved = 0
    if save_path is not None:
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        if not save_path.exists():
            os.mkdir(save_path)
        save_after = len(train_dataloader) // divider
    model = model.to(device)
    model.train()
    for e in range(epochs):
        
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {e}")
                for key in batch:
                    batch[key] = batch[key].to(device)
                model.zero_grad()
                score = model(batch)
                loss = loss_function(score, batch["labels"])
                loss.backward()
                optimizer.step()
                loss_avg.append(loss.item())
                tepoch.set_postfix(loss=np.mean(loss_avg), )
                save_counter += 1
                if save_path is not None:
                    if save_counter > save_after:
                        torch.save(model.state_dict(), save_path / f"model_{save_counter + save_after*already_saved}.cpkt")
                        save_counter = 0
                        already_saved += 1
    if save_path is not None:
        torch.save(model.state_dict(), save_path / f"model_final.cpkt")
            
def inference_model(data, model, tokenizer, device, 
           alighment_len, batch_size, is_siames):
    """Test model.

    Args:
        data (List[Tuple[List[str], List[str], int]]): Testing bare data.
        model (_type_): Model to be tested.
        tokenizer (_type_): The tokenizer for preparing the dataloader.
        device (_type_): The device to train on.
        alighment_len (_type_): The length of the sequence alighment.
        batch_size (_type_):  The batch size.
        is_siames (bool): Whether the network is Siames.

    Returns:
        _type_: The tuple with predictions and its correspondind labels.
    """
    test_dataloader = prepare_dataloader(data, tokenizer, alighment_len, batch_size, is_siames)
    model.eval()
    predictions = []
    correct = []
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                score = model(batch)
            predictions.append(score.argmax(dim=1, keepdim=True).squeeze().detach().cpu().numpy())
            correct.append(batch["labels"].detach().cpu().numpy())
    pred = np.hstack(predictions)
    corr = np.hstack(correct)
    return pred, corr
    
def inference_sbert(df_test_, model):
    """Inference SentenceBERT model.

    Args:
        df_test_ (_type_): DataFrame with test data
        model (_type_): SentenceBERT model

    Returns:
        _type_: The array with predictions
    """
    test_dataset = PairedTextDataset(df_test_.premise.to_list(), 
                                  df_test_.hypothesis.to_list(), 
                                  df_test_.label.to_list())

    dataloader = DataLoader(test_dataset, batch_size=32)
    predictions = []
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                score = model(batch["text_pair"])
                predictions.append(score.argmax(dim=1, keepdim=True).squeeze())
    predictions = [x.detach().cpu().numpy() for x in predictions]
    predictions = np.hstack(predictions)
    return predictions