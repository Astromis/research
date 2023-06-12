import logging

import pandas as pd
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, BertForNextSentencePrediction,
                          Trainer, TrainingArguments)

from src.dataset import TextsLabelsDataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {'F1': f1}

def main():
    logging.info("Loading data")
    df_train = pd.read_csv("train_data.csv", sep="|")
    df_eval = pd.read_csv("eval_data.csv", sep="|")
    logging.info("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("ru_conversational_cased_L-12_H-768_A-12_pt_v1")
    model = BertForNextSentencePrediction.from_pretrained('ru_conversational_cased_L-12_H-768_A-12_pt_v1')
    
    logging.info("Tokenizing data")
    tokenized_texts_tr = tokenizer(df_train.premise.to_list(), df_train.hypothesis.to_list(), return_tensors='pt',
                                truncation=True, max_length=512, padding = 'max_length',)
    train_dataset = TextsLabelsDataset(tokenized_texts_tr, df_train.labels.to_list())
    
    tokenized_texts_ev = tokenizer(df_eval.premise.to_list(), df_eval.hypothesis.to_list(), return_tensors='pt',
                                truncation=True, max_length=512, padding = 'max_length',)
    eval_dataset = TextsLabelsDataset(tokenized_texts_ev, df_eval.labels.to_list())

    training_args = TrainingArguments(
    output_dir = './results', #Выходной каталог
    num_train_epochs = 3, #Кол-во эпох для обучения
    per_device_train_batch_size = 8, #Размер пакета для каждого устройства во время обучения
    per_device_eval_batch_size = 8, #Размер пакета для каждого устройства во время валидации
    weight_decay =0.01, #Понижение весов
    logging_dir = './logs', #Каталог для хранения журналов
    load_best_model_at_end = True, #Загружать ли лучшую модель после обучения
    learning_rate = 1e-5, #Скорость обучения
    evaluation_strategy ='epoch', #Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
    logging_strategy = 'epoch', #Логирование после каждой эпохи
    save_strategy = 'epoch', #Сохранение после каждой эпохи
    save_total_limit = 1,
    seed=21)
    
    trainer = Trainer(model=model,
                  args = training_args,
                  train_dataset = train_dataset,
                  eval_dataset = eval_dataset,
                  compute_metrics = compute_metrics,
                    )
    trainer.train()
    
main()