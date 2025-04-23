import os
import json
import logging
import time
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from datasets import Dataset
from batches_utils import split_dataset_into_batches, save_batch_to_json
from datetime import datetime
from pydantic import BaseModel, ValidationError
from typing import List, Tuple, Optional, Any
from multiprocessing import Pool

from markup_transfer import (
    make_logger,
    BATCH_RESULT_DIR,
    BATCH_DIR,
    FAILED_INDEXES_PATH,
    create_directory_if_not_exists,

)

# Константы, отличные от скрипта `markup_transfer`
# Удаление из входящего датасета. В итоге оставляем только `id`, `text`
COLUMNS_TO_REMOVE = ["definition_label","source_file","tokens","iob_tags","spans","relations","text_rus","spans_rus","relations_rus","is_gold"]


class ResultSchema(BaseModel):
    id: str
    text: str
    text_rus: str


def load_dataset(file_path: str) -> list[Dataset, Dataset]:
    """Функция для загрузки датасета и удаления результатирующих частей, если они существуют"""
    try:
        dataset = Dataset.from_json(file_path)
        logging.info(f"Dataset loaded successfully from '{file_path}'.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        return None

    output_dataset_correct_results = dataset

    # Проверка на существование указанных колонок и их удаление
    columns_present = [col for col in COLUMNS_TO_REMOVE if col in dataset.column_names]
    input_dataset = dataset.remove_columns(columns_present)

    return [output_dataset_correct_results, input_dataset]


def postprocessing_add_other_fields(orig_dataset_data, list_of_results):
    """Функция постобработки для добавления остальных полей в возвращаемый моделью json.
    
    По сути будем делать немного наоборот:
    - берем оригинальный полный `orig_dataset_data` и 
    - у него обновляем `text_rus` из `list_of_results`
    
    """
    logging.info(f"Postprocessing model data (add other fields) of LLM...")
    # Создаем индекс для быстрого доступа к text_rus от Модели по ID
    id2llm_text_rus = {}
    logging.info(f"Creating index `id2llm_text_rus`...")
    for elem in list_of_results:
        id2llm_text_rus[elem["id"]] = elem["text_rus"]
    logging.info(f"Created index `id2llm_text_rus` succsessfully!")

    # Обновляем text_rus
    orig_dataset_data = orig_dataset_data.to_list()
    errors_cnt = 0
    for res in orig_dataset_data:
        try:
            if 'text_rus' in res and res['text_rus'] != "":
                before_text_rus = res['text_rus']
                after_text_rus = id2llm_text_rus[res['id']]
                res['text_rus'] = after_text_rus
                logging.info(f"ID: '{res['id']}'. Orig dataset has 'text_rus'. Add new 'text_rus' from LLM. Replaced from '{before_text_rus}' to '{after_text_rus}'.")
            else:
                res['text_rus'] = id2llm_text_rus[res['id']]
        except Exception as e:
            # by default
            res['text_rus'] = ''
            errors_cnt += 1
            logging.error(f"Error in postprocessing: {e}. Have not model result for element. This element from dataset has empty 'text_rus'. Please update it (add 'text_rus' field) manually to result if you need. Dataset's element: '{res}'")

    if errors_cnt == 0:
        logging.info("Postprocessing model data (add other fields) completed successfully.")
    else:
        logging.info(f"Postprocessing model data (add other fields) completed successfully, but have {errors_cnt} errors (empty 'text_rus' field, check logs).")

    return orig_dataset_data


def init_chain(config, system_template):
    """Инициализация цепи внутри дочернего процесса"""
    model = ChatOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"]
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    parser = StrOutputParser()
    return prompt_template | model | parser


def process_batch_wrapper(args):
    """Обработка одного батча
    
    Инициализация модели и подключения к АПИ для каждого процесса
    Обработка батча
    """
    # Распаковываем аргументы
    batch_idx, batch, cnt_batches, config, system_template = args
    logging.info(f"Processing batch {batch_idx + 1}/{cnt_batches}...")

    # Инициализируем цепь внутри процесса
    chain = init_chain(config, system_template)

    # Список результатов для текущего батча
    batch_results = []

    for i, input_example in enumerate(batch):
        user_input = 'Input: {0}\nResult:'.format(json.dumps(input_example, ensure_ascii=False))
        example_id = input_example.get("id", f"example_{i}") # получаем id примера или создаем временный

        success = False
        attempt = 0
        max_retries = 3
        retry_delay = 2
        
        while not success and attempt < max_retries:
            try:
                res = chain.invoke({"text": user_input})
                result_json = json.loads(res)

                # ID от модели в нижний регистр
                if "id" in result_json:
                    result_json["id"] = result_json["id"].lower()
                # Проверка совпадения ID
                if result_json.get("id") != example_id:
                    raise ValueError(f"ID mismatch: expected {example_id}, got {result_json.get('id')}")

                # Проверка формата возвращаемых данных
                try:
                    ResultSchema.parse_obj(result_json)
                except ValidationError as e:
                    raise ValueError(f"Bad output format: {e}")

                batch_results.append(result_json)
                logging.info(f"Successfully processed input #{i} (id: {example_id}) in batch {batch_idx + 1} with result: {result_json}.")
                success = True
            except json.JSONDecodeError as e:
                attempt += 1
                logging.warning(f"JSONDecodeError on input #{i} (id: {example_id}) in batch {batch_idx + 1}: {str(e)}. "
                                f"Attempt {attempt} of {max_retries}. Retrying...")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    with open(FAILED_INDEXES_PATH, "a") as f:
                        f.write(f"{example_id}\n")
                    logging.error(f"Failed to process input #{i} (id: {example_id}) in batch {batch_idx + 1} after {max_retries} attempts due to JSON decoding issue."
                                    f"Response: {res}")
            except Exception as e:
                attempt += 1
                logging.warning(f"Error during model execution on input #{i} (id: {example_id}) in batch {batch_idx + 1}: {str(e)}. "
                                f"Attempt {attempt} of {max_retries}. Retrying...")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    with open(FAILED_INDEXES_PATH, "a") as f:
                        f.write(f"{example_id}\n")
                    logging.error(f"Failed to process input #{i} (id: {example_id}) in batch {batch_idx + 1} after {max_retries} attempts due to invoke error."
                                    f"Error: {str(e)}.")

    # Сохраняем результаты батча только если используется разбиение на батчи, т.е. батчей больше 1
    if cnt_batches > 1:
        batch_result_dir = create_directory_if_not_exists(BATCH_RESULT_DIR)
        save_batch_to_json(batch=Dataset.from_list(batch_results), base_filename= "model_result", batch_num = batch_idx, save_dir=batch_result_dir)  # Сохраняем промежуточные результаты
    
    return batch_results


def run_model_and_save_results(
        config: dict,
        filepath_gold_dataset: str, 
        filepath_examples: str, 
        system_message: str, 
        batch_size: int, 
        intermediate_path: str =None,
        proc_cnt: int = 1,
        process_failed_indexes: str = None,
        ):
    """Функция для запуска модели и сохранения результатов с логированием и возможностью разбиения на батчи"""

    # Загрузка датасета
    datasets = load_dataset(filepath_gold_dataset)
    if not datasets:
        logging.error("Failed to load dataset.")
        return

    orig_dataset_data, input_dataset = datasets

    if process_failed_indexes:
        with open(process_failed_indexes, "r") as f:
            # Читаем файл, удаляем пробельные символы и пустые строки
            failed_indexes = [line.strip() for line in f if line.strip()]
        # Преобразуем в множество для быстрого поиска
        failed_ids = set(failed_indexes)
        # Фильтруем orig_dataset_data
        orig_dataset_data = orig_dataset_data.filter(lambda x: x['id'] in failed_ids)
        # Фильтруем input_dataset
        input_dataset = input_dataset.filter(lambda x: x['id'] in failed_ids)

    with open(filepath_examples, encoding='utf-8') as f:
        examples_data = json.load(f)

    examples = examples_data["examples"]
    system_template = system_message
    system_template += "\n".join(
        f"Example Input: {json.dumps(example['input'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}\n"
        f"Example Result: {json.dumps(example['result'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}"
        for example in examples
    )

    # Если batch_size указан и больше 0, делаем разбиение на батчи
    if batch_size > 0:
        logging.info(f"Splitting dataset into batches of size {batch_size}.")
        batched_input_dataset = split_dataset_into_batches(input_dataset, batch_size)
        # batched_input_dataset = split_dataset_into_batches_serializable(input_dataset, batch_size)
        batch_dir = create_directory_if_not_exists(BATCH_DIR)
        for i, batch in enumerate(batched_input_dataset):
            save_batch_to_json(batch=batch, base_filename= "separated_", batch_num = i, save_dir=batch_dir)
    else:
        logging.info("Processing the entire dataset without batching.")
        batched_input_dataset = [input_dataset]  # Оборачиваем в список для единого формата

    list_of_results = []

    # Параллельная обработка
    with Pool(proc_cnt) as pool:
        results = pool.imap_unordered(
            process_batch_wrapper,
            # передаем номер батча, сам батч, количество батчей, chain
            ((i, batch, len(batched_input_dataset), config, system_template) for i, batch in enumerate(batched_input_dataset))
            )
        
        for batch_result in results:
            list_of_results.extend(batch_result)

    if intermediate_path:
        Dataset.from_list(list_of_results).to_json(intermediate_path, force_ascii=False)

    return orig_dataset_data, list_of_results


def _parser():
    parser = argparse.ArgumentParser(description="Run the model for translation with specified arguments")
    parser.add_argument("--config", "-c", type=str, default="references/model_config.json",
                        help="Path to the model config file")
    parser.add_argument("--system_message", "-sm", type=str, default="references/system_message_translation_via_llm_2.txt",
                        help="Path to the system message file.")
    parser.add_argument("--input_dataset", "-i", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--examples", "-e", type=str, default="references/model_examples_translation_via_llm.json",
                        help="Path to the prompt I/O examples")
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="Batch size for dataset processing")
    parser.add_argument("--proc_cnt", "-p", type=int, default=1, help="Amount of processes for multiprocessing of batch processing")
    parser.add_argument("--save_intermediate", "-s", type=str, default="temp_model_result.json",
                        help="Path to save the intermediate model results")
    parser.add_argument("--output_result", "-o", type=str, default="translation_result.json",
                        help="Path to save the final model results")

    parser.add_argument("--process_failed_indexes", "-pf", type=str, default=None,
                        help="Processing failed indexes. In Arg please paste path to your failed_indexes.txt")  
    parser.add_argument("--postprocessing_only", "-pp", action='store_true',
                        help="PostProcessing only. Take raw results (--save_intermediate) and orig dataset (--input_dataset) and create final result (--output_result)")  
    return parser

def main():
    # Настройка логгера
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"model_run_translation_{current_time}.log"
    make_logger(log_filename)

    parser = _parser()
    args = parser.parse_args()

    if args.postprocessing_only:
        # Только постпроцессинг
        orig_dataset_data, _ = load_dataset(args.input_dataset)

        list_of_results = Dataset.from_json(args.save_intermediate)
        list_of_results = list_of_results.to_list()

        # Постобработка, т.е. добавление обратно всех полей в док.
        converted_result = postprocessing_add_other_fields(orig_dataset_data, list_of_results)
        hf_dataset_converted = Dataset.from_list(converted_result)
        hf_dataset_converted.to_json(args.output_result, force_ascii=False)
        return

    # Загрузка модели из конфигурации
    with open(args.config, "r") as f:
        config = json.load(f)

    # Загрузка system_message
    with open(args.system_message, "r") as f:
        system_message = f.read()

    # Запуск модели
    orig_dataset_data, list_of_results = run_model_and_save_results(
        config=config,
        filepath_gold_dataset=args.input_dataset,
        filepath_examples=args.examples,
        system_message=system_message,
        batch_size=args.batch_size,
        proc_cnt=args.proc_cnt,
        process_failed_indexes=args.process_failed_indexes
    )

    # Сохранение промежуточного результата
    if args.save_intermediate:
        hf_dataset = Dataset.from_list(list_of_results)
        hf_dataset.to_json(args.save_intermediate, force_ascii=False)

    # Постобработка, т.е. добавление обратно всех полей в док.
    converted_result = postprocessing_add_other_fields(orig_dataset_data, list_of_results)
    hf_dataset_converted = Dataset.from_list(converted_result)
    hf_dataset_converted.to_json(args.output_result, force_ascii=False)


if __name__ == "__main__":
    main()
