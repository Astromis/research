# Materials for paper "Who is answering to whom? Modeling reply-to relationships in Russian asynchronous chats"

This dir contains the materials to reproduce the experiments from paper ["Who is answering to whom? Modeling reply-to relationships in Russian asynchronous chats"](https://www.dialog-21.ru/media/5871/buyanoviplusetal046.pdf)

## Where I can download the data

Data can be downloaded [here]() and [here](). Create a directory called "data" and place the dataset there. Each dataset has its own `README.md`.

The source Conversational RuBERT can ber downloaded [here](http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12.tar.gz).

The trained reply recovery BERT model, which is actually the best model, is availabele in Hugging Face [repo](https://huggingface.co/astromis/rubert_reply_recovery).

## How to reproduce

Make sure, that you have all necessary dependencies and run

1. `model_testing_pipeline.ipynb` to reproduce the reply recovery results.
2. `thread_reconstruction.ipynb` to reproduce the thread reconstruction results.
3. `train_model_model.py` to train the bert model.

# Cite

```bibtex
@article{Buyanov2023WhoIA,
  title={Who is answering to whom? Modeling reply-to relationships in Russian asynchronous chats},
  author={Igor Buyanov and Darya Yaskova and Ilya Sochenkov},
  journal={Computational Linguistics and Intellectual Technologies},
  year={2023}
}
```