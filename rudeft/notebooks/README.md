# Анализ данных и эксперименты
> \* HF - Hugging Face, LS - Label Studio

### Конвертация данных:
- `translator_hf2ls_ls2hf.ipynb` - Конвертер датасета HF в LS и LS в HF
- `translator_translation_ls2hf.ipynb` - Конвертер переведенных вручную данных из LS в HF

### Анализ данных:
- `analyze_ls_data.ipynb` - Анализ датасета LS (переведенного из HF) на разные случаи с данными, перевод, разметка, связи и т.д.

### Работа с переводами:
- `collect_sent_for_check_translation.ipynb` - Сбор предложений для проверки перевода по 200 примеров на 3 человек
- `labse_bleu_for_translation_via_llm.ipynb` - Расчет метрик качества перевода (BLEU, LaBSE) полученного от LLM

### Работа с GOLD:
- `make_gold_dataset_for_translation_llm.ipynb` - Создание эталонного (GOLD) датасета для оценки перевода
- `make_dataset_without_gold_for_translation_llm.ipynb` - Подготовка датасета без эталонных примеров для перевода
- `make_dataset_with_adding_gold_elems_to_llm_results.ipynb` - Подготовка итогового датасета (обновляем те примеры, которые из GOLD датасета)

### Обучение моделей:
- `train_ner_model.ipynb` - Обучение NER-модели для распознавания именованных сущностей


[🏠 Назад к главному README](../README.md)
