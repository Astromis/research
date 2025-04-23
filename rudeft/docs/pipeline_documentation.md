# Пайплайн по переводу и переносу разметки

Для работы с проектом необходимо установить все зависимости из requirements.txt в свое виртуальное окружение: `pip install -r requirements.txt`

Для того чтобы преобразовать исходные данные формата conll (.deft) в HF датасет для дальнейшего корректного использования - понадобится использовать скрипт [translator_connl_to_hf.py](./connl_to_hf_docs.md)  
Пример запуска: `python src/translator_connl_to_hf.py --deft_corpus_folder "data/my_deft_data" --output_file "processed_data.json"`  
На выходе получаем датасет в формате HF, который затем сохраняется в json файл. При необходимости его можно загрузить с помощью библиотеки Datasets командой `Dataset.from_json(<filepath>)`

Для того чтобы выполнить перевод датасета с помощью LLM понадобится использовать скрипт [translation_via_llm.py](./translation_via_llm_docs.md), а также полученный на прошлом этапе датасет в формате HF и настроенный конфиг для модели.  
Пример запуска: `python src/translation_via_llm.py --config config.json --input_dataset datasets/test.json --examples ref/examples.json --save_intermediate datasets/test_result_raw.json --output_result datasets/test_result.json --batch_size 100`  
Отслеживать прогресс можно в созданном .log файле.  
На выходе получаем итоговый датасет с переведенными предложениями. 

Для того чтобы перенести разметку с помощью LLM понадобится использовать скрипт [markup_transfer.py](./markup_transfer_docs.md), а также полученный на прошлом этапе переведенный датасет, а также настроенный конфиг для модели, который был настроен также на прошлом этапе, либо видоизменен для текущего этапа.  
Пример запуска: `python src/markup_transfer.py --config config.json --input_dataset datasets/test.json --examples ref/examples.json --save_intermediate datasets/test_result_raw.json --output_result datasets/test_result.json --fuzzy --batch_size 100`  
Отслеживать прогресс можно в созданном .log файле.  
На выходе получаем различные файлы, описанные [здесь](./markup_transfer_docs.md), но главное итоговый датасет с перенесенной разметкой. Важно подметить: возможно после выполнения скрипта у LLM не получится перенести разметку для некоторых примеров из датасета. В таком случае ID этих примеров будут записаны в файл `failed_indexes.txt` и следует создать новый json только с текущими примерами из `failed_indexes.txt` и запустить скрипт еще раз, указав путь к входному датасету на этот json файл. После этого нужно будет объединить полученные файлы с результатами - это и будет итоговым датасетом.

[🏠 Назад к главному README](../README.md)
