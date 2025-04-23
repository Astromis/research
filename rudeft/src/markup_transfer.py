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
import difflib
from fuzzysearch import find_near_matches
from datetime import datetime
from pydantic import BaseModel, ValidationError
from typing import List, Tuple, Optional, Any
from multiprocessing import Pool


def make_logger(log_filename):
    # Настройка логгера
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - [%(process)d] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8'
    )

# Константы
COLUMNS_TO_REMOVE = ["spans_rus", "relations_rus", "is_gold"]
BATCH_RESULT_DIR = 'batches_res/'
BATCH_DIR = 'batches/'
FAILED_INDEXES_PATH = "failed_indexes.txt"


class ResultSchema(BaseModel):
    id: str
    text: str
    definition_label: int
    source_file: str
    tokens: List[str]
    iob_tags: List[str]
    spans: Optional[List[Tuple[str, str, str, str, str]]] = []
    relations: Optional[List[Tuple[str, str, str]]] = []
    text_rus: str
    spans_rus: Optional[List[Tuple[str, str, str]]] = []
    relations_rus: Optional[Any] = None

def create_directory_if_not_exists(directory_path):
    """Создает папку, если она не существует."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory '{directory_path}' is ready.")
        return directory_path
    except Exception as e:
        logging.error(f"Failed to create directory '{directory_path}': {e}")
        return './'

def add_span_texts(instance):
    """Функция для добавления текста в spans"""
    text = instance["text"]
    updated_spans = []
    for span in instance["spans"]:
        start_idx = int(span[0])
        end_idx = int(span[1])
        span_text = text[start_idx:end_idx]
        span_with_text = span + [span_text]
        updated_spans.append(span_with_text)
    instance["spans"] = updated_spans
    return instance

def load_dataset(file_path: str) -> list[Dataset, Dataset]:
    """Функция для загрузки датасета и удаления результатирующих частей, если они существуют"""
    try:
        dataset = Dataset.from_json(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        return None

    dataset = dataset.map(add_span_texts)
    output_dataset_correct_results = dataset

    # Проверка на существование указанных колонок и их удаление
    columns_present = [col for col in COLUMNS_TO_REMOVE if col in dataset.column_names]
    input_dataset = dataset.remove_columns(columns_present)

    return [output_dataset_correct_results, input_dataset]

def remove_temp_text_from_spans(data):
    """Функция удаления временных переменных из датасета"""
    for entry in data:
        if "spans" in entry:
            entry["spans"] = [span[:4] for span in entry["spans"]]
        if "spans_rus" in entry:
            entry["spans_rus"] = [span[:4] for span in entry["spans_rus"]]
    return data

def convert_spans_rus_format(orig_dataset_data, data, fuzzy):
    """Функция постобработки для преобразования спанов в корректный формат; определение индексов"""

    logging.info(f"Postprocessing model data of LLM...")
    # Создаем индекс для быстрого доступа к объекту intermediate (raw) датасета от Модели по ID
    id2llm_data = {}
    logging.info(f"Creating index `id2llm_data`...")
    for elem in data:
        id2llm_data[elem["id"]] = elem
    logging.info(f"Created index `id2llm_data` succsessfully!")

    # Обновляем text_rus
    orig_dataset_data = orig_dataset_data.to_list()
    errors_cnt = 0
    for res in orig_dataset_data:
        try:
            entry = id2llm_data[res['id']]

            text_rus = entry["text_rus"]
            spans_rus = entry["spans_rus"]
            relations_eng = entry["relations"]

            converted_spans_rus = []
            converted_relations_rus = []

            for span in spans_rus:
                label, id_rus, rus_text = span
                start_idx = text_rus.find(rus_text)

                # Логика зависит от аргумента --fuzzy
                if fuzzy and start_idx == -1:
                    logging.warning(f"Exact match not found for '{rus_text}' for entry {entry['id']} in {text_rus}")
                    matches = find_near_matches(rus_text, text_rus, max_l_dist=3)  # max_l_dist - допустимое число расхождений
                    if matches:
                        start_idx = matches[0].start
                        end_idx = matches[0].end
                        logging.info(f"Fuzzy match found: {text_rus[start_idx:end_idx]}")
                    else:
                        # Используем difflib для нахождения наиболее похожей подстроки
                        closest_match = difflib.get_close_matches(rus_text, [text_rus], n=1, cutoff=0.75)
                        if closest_match:
                            start_idx = text_rus.find(closest_match[0])
                            end_idx = start_idx + len(closest_match[0])
                            logging.info(f"Closest match found: {closest_match[0]}")
                        else:
                            logging.error(f"Text '{rus_text}' not found in '{text_rus}' for entry {entry['id']}.")
                            start_idx = -1
                            end_idx = -1
                elif start_idx == -1:
                    start_idx = -1
                    end_idx = -1
                else:
                    end_idx = start_idx + len(rus_text)

                id_rus = f"{id_rus}_rus"
                converted_spans_rus.append([str(start_idx), str(end_idx), label, id_rus, rus_text])

            for relation in relations_eng:
                entry_link_id, out_link_id, relation_type = relation
                entry_link_id_rus = f"{entry_link_id}_rus"
                out_link_id_rus = f"{out_link_id}_rus"
                converted_relations_rus.append([entry_link_id_rus, out_link_id_rus, relation_type])

            res["spans_rus"] = converted_spans_rus
            res["relations_rus"] = converted_relations_rus

        except Exception as e:
            # by default
            res["spans_rus"] = []
            res["relations_rus"] = []
            errors_cnt += 1
            logging.error(f"Error in postprocessing: {e}. Have not model result for element. This element from dataset has empty 'spans_rus' and 'relations_rus'. Please update it manually to result if you need. Dataset's element: '{res}'")


    if errors_cnt == 0:
        logging.info("Postprocessing - Conversion of spans_rus format completed successfully.")
    else:
        logging.info(f"Postprocessing - Conversion of spans_rus format completed successfully, but have {errors_cnt} errors (empty 'spans_rus' and 'relations_rus' fields, check logs).")

    orig_dataset_data = remove_temp_text_from_spans(orig_dataset_data)

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
    batch_idx, batch, cnt_batches, config, system_template, ignore_non_definition_example = args
    logging.info(f"Processing batch {batch_idx + 1}/{cnt_batches}...")

    # Инициализируем цепь внутри процесса
    chain = init_chain(config, system_template)

    # Список результатов для текущего батча
    batch_results = []

    for i, input_example in enumerate(batch):
        user_input = 'Input: {0}\nResult:'.format(json.dumps(input_example, ensure_ascii=False))
        example_id = input_example.get("id", f"example_{i}") # получаем id примера или создаем временный

        # non diff 
        if ignore_non_definition_example and input_example.get('definition_label') == 0:
            input_example["spans_rus"] = []
            input_example["relations_rus"] = []
            try:
                ResultSchema.parse_obj(input_example)
                batch_results.append(input_example)
                logging.info(f"Skipped example {example_id} (definition_label=1).")
            except ValidationError as e:
                logging.error(f"Validation error for skipped example {example_id}: {e}")
            continue
        
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
        save_batch_to_json(batch=Dataset.from_list(batch_results), base_filename="model_result", batch_num=batch_idx, save_dir=batch_result_dir)  # Сохраняем промежуточные результаты
    
    return batch_results

def run_model_and_save_results(
        config: dict,
        filepath_gold_dataset: str, 
        filepath_examples: str, 
        system_message: str, 
        batch_size: int, 
        intermediate_path: str = None,
        proc_cnt: int = 1,
        process_failed_indexes: str = None,
        ignore_non_definition_example: bool = False
    ):
    """Функция для запуска модели и сохранения результатов с логированием и возможностью разбиения на батчи"""

    # Загрузка датасета
    datasets = load_dataset(filepath_gold_dataset)
    if not datasets:
        logging.error("Failed to load dataset.")
        return

    output_dataset_correct_results, input_dataset = datasets

    if process_failed_indexes:
        with open(process_failed_indexes, "r") as f:
            # Читаем файл, удаляем пробельные символы и пустые строки
            failed_indexes = [line.strip() for line in f if line.strip()]
        # Преобразуем в множество для быстрого поиска
        failed_ids = set(failed_indexes)
        # Фильтруем datasets
        output_dataset_correct_results = output_dataset_correct_results.filter(lambda x: x['id'] in failed_ids)
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
        batch_dir = create_directory_if_not_exists(BATCH_DIR)
        for i, batch in enumerate(batched_input_dataset):
            save_batch_to_json(batch=batch, base_filename="separated_", batch_num=i, save_dir=batch_dir)
    else:
        logging.info("Processing the entire dataset without batching.")
        batched_input_dataset = [input_dataset]  # Оборачиваем в список для единого формата

    list_of_results = []

    # Параллельная обработка
    with Pool(proc_cnt) as pool:
        results = pool.imap_unordered(
            process_batch_wrapper,
            # передаем номер батча, сам батч, количество батчей, config, system_template
            ((i, batch, len(batched_input_dataset), config, system_template, ignore_non_definition_example) for i, batch in enumerate(batched_input_dataset))
        )
        
        for batch_result in results:
            list_of_results.extend(batch_result)

    if intermediate_path:
        Dataset.from_list(list_of_results).to_json(intermediate_path, force_ascii=False)

    return output_dataset_correct_results, list_of_results


def main():
    # Настройка логгера
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"model_run_{current_time}.log"
    make_logger(log_filename)

    parser = argparse.ArgumentParser(description="Run the model with specified arguments")
    parser.add_argument("--config", "-c", type=str, default="references/model_config.json",
                        help="Path to the model config file")
    parser.add_argument("--system_message", "-sm", type=str, default="references/system_message_markup_transfer.txt",
                        help="Path to the system message file.")
    parser.add_argument("--input_dataset", "-i", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--examples", "-e", type=str, default="references/model_examples_markup_transfer.json",
                        help="Path to the prompt I/O examples")
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="Batch size for dataset processing")
    parser.add_argument("--proc_cnt", "-p", type=int, default=1, help="Amount of processes for multiprocessing of batch processing")
    parser.add_argument("--save_intermediate", "-s", type=str, default="temp_model_result.json",
                        help="Path to save the intermediate model results")
    parser.add_argument("--output_result", "-o", type=str, default="result.json",
                        help="Path to save the final model results")
    parser.add_argument("--fuzzy", "-f", action="store_true", help="Enable fuzzy matching for spans_rus processing")

    parser.add_argument("--process_failed_indexes", "-pf", type=str, default=None,
                        help="Processing failed indexes. In Arg please paste path to your failed_indexes.txt")
    parser.add_argument("--postprocessing_only", "-pp", action='store_true',
                        help="PostProcessing only. Take raw results (--save_intermediate) and orig dataset (--input_dataset) and create final result (--output_result)")
    parser.add_argument("--ignore_non_definition_example", "-ign", action="store_true", 
                        help="Ignore examples with definition_label == 0, returning them with empty spans_rus and relations_rus")  
    args = parser.parse_args()

    if args.postprocessing_only:
        # Только постпроцессинг
        orig_dataset_data, _ = load_dataset(args.input_dataset)

        list_of_results = Dataset.from_json(args.save_intermediate)
        list_of_results = list_of_results.to_list()

        # Постобработка
        converted_result = convert_spans_rus_format(orig_dataset_data, list_of_results, fuzzy=args.fuzzy)
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
        process_failed_indexes=args.process_failed_indexes,
        ignore_non_definition_example=args.ignore_non_definition_example
    )

    # Сохранение промежуточного результата
    if args.save_intermediate:
        hf_dataset = Dataset.from_list(list_of_results)
        hf_dataset.to_json(args.save_intermediate, force_ascii=False)

    # Постобработка
    converted_result = convert_spans_rus_format(orig_dataset_data, list_of_results, fuzzy=args.fuzzy)
    hf_dataset_converted = Dataset.from_list(converted_result)
    hf_dataset_converted.to_json(args.output_result, force_ascii=False)

if __name__ == "__main__":
    main()
    