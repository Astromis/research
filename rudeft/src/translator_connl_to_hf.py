from typing import List, Dict, Tuple
from datasets import Dataset
import os
import argparse
import uuid

def parse_conll_line(line: str) -> Tuple[str, str, int, int, str, str, str, str]:
    """Парсинг строки в формате conll + убирает лишние пробелы"""
    fields = line.strip().split('\t')
    if len(fields) < 8:
        raise ValueError(f"Invalid conll line: {line}")
    token, source_file, start_char, end_char, tag, tag_id, root_id, relation = [field.strip() for field in fields]
    return token, source_file, int(start_char), int(end_char), tag, tag_id, root_id, relation

def get_span_indices(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """Получение индексов начала и конца спана в тексте на основе токенов и их позиций в тексте"""
    indices = []
    current_position = 0

    for token in tokens:
        token_start = text.find(token, current_position)
        token_end = token_start + len(token)
        indices.append((token_start, token_end))
        current_position = token_end

    return indices

def extract_spans(tags: List[str], indices: List[Tuple[int, int]], tag_ids: List[str]) -> List[Tuple[str, str, str, str]]:
    """Извлечение спанов на основе IOB тегов и индексов в тексте, с использованием tag_id для идентификации"""
    spans = []
    current_class = None
    span_start = None

    for i, (tag, (start, end), tag_id) in enumerate(zip(tags, indices, tag_ids)):
        tag = tag.strip()

        if tag.startswith('B-'):
            # Закрытие предыдущего спана
            if current_class is not None:
                spans.append((str(span_start), str(span_end), current_class, current_tag_id))

            # Начало нового спана
            current_class = tag[2:]
            span_start = start
            span_end = end
            current_tag_id = tag_id

        elif tag.startswith('I-') and current_class:
            span_end = end  # Продление текущего спана

        else:
            # Закрытие текущего спана
            if current_class is not None:
                spans.append((str(span_start), str(span_end), current_class, current_tag_id))
            current_class = None  # Сброс текущего спана

    # Закрытие последнего спана
    if current_class is not None:
        spans.append((str(span_start), str(span_end), current_class, current_tag_id))

    return spans

def extract_relations(relations: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Создание уникальных отношений между спанами с использованием tag_id"""
    unique_relations = set()

    for tag_id, root_id, relation in relations:
        unique_relations.add((tag_id, root_id, relation))

    return list(unique_relations)

def process_entity(tokens: List[str], iob_tags: List[str], relations: List[Tuple[str, str, str]], tag_ids: List[str], current_source_file: str, is_block_start: bool) -> Dict:
    """Обработка одного предложения conll файла и преобразование его для структуры HF"""
    text = ' '.join(tokens)
    indices = get_span_indices(text, tokens)
    spans = extract_spans(iob_tags, indices, tag_ids)
    unique_relations = extract_relations(relations)

    definition_label = 1 if any(tag != 'O' for tag in iob_tags) else 0  # Если хотя бы один IOB отличен от O, то 1

    # Создание структуры для HF datasets
    entity = {
        "id": uuid.uuid4().hex,
        "text": text,
        "definition_label": definition_label,
        "source_file": current_source_file,
        "tokens": tokens,
        "iob_tags": iob_tags,
        "spans": spans,
        "relations": unique_relations,
        "is_block_start": is_block_start  # Признак начала блока (временная метка, для матчинга русского текста, в последствии удаляется)
    }
    return entity

def handle_block_transition(line: str, prev_line_empty: bool) -> Tuple[bool, bool]:
    """Определяет начало нового блока и переход между предложениями"""
    if not line.strip():
        if prev_line_empty:  # Если две пустые строки подряд
            return True, True  # Начало нового блока
        return False, True  # Просто пустая строка - конец предложения
    return False, False  # Продолжается предложение

def parse_conll_file(filepath: str) -> List[Dict]:
    """Конвертация conll файла в список словарей, соответствующих формату HF"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    converted_entities = []
    tokens, iob_tags, tag_ids, relations = [], [], [], []
    current_source_file = None
    prev_line_empty = False
    is_block_start = True

    for line in lines:
        # Обрабатываем переходы между блоками
        block_start, line_empty = handle_block_transition(line, prev_line_empty)
        prev_line_empty = line_empty

        if block_start:  # Если начинается новый блок
            is_block_start = True

        if line.strip():  # Если строка не пустая
            token, source_file, start_char, end_char, tag, tag_id, root_id, relation = parse_conll_line(line)
            tokens.append(token)
            iob_tags.append(tag.strip())
            tag_ids.append(tag_id)
            current_source_file = source_file

            if relation != "0" and tag_id != "-1":
                relations.append((tag_id, root_id, relation))

        # Если это конец предложения, то сохраняем пример
        if line_empty and tokens:
            converted_entities.append(process_entity(tokens, iob_tags, relations, tag_ids, current_source_file, is_block_start))
            tokens, iob_tags, tag_ids, relations = [], [], [], []  # Сброс временных данных
            is_block_start = False  # После сохранения сбрасываем флаг начала блока

    # Обработка последнего примера (если файл не заканчивается пустой строкой)
    if tokens:
        converted_entities.append(process_entity(tokens, iob_tags, relations, tag_ids, current_source_file, is_block_start))

    return converted_entities

def remove_temp_fields(targer_list_of_dicts: List[Dict]) -> List[Dict]:
    for elem in targer_list_of_dicts:
        if "is_block_start" in elem:
            del elem["is_block_start"]
    return targer_list_of_dicts

def add_ru_eng(parsed_dict: List[Dict], ru_dir: str) -> List[Dict]:
    """Функция для соответствия и добавления русского языка к исходным примерам"""
    from eng_ru_matching_pipeline import correspondence # Импорт функции, т.к она используется только здесь
    
    for elem in parsed_dict:
        elem["text_rus"] = ""
    eng_ru_dict = correspondence(parsed_dict, ru_dir)
    return eng_ru_dict

def find_deft_files(directory: str) -> List[str]:
    """Поиск всех .deft файлов в указанной директории и поддиректориях"""
    deft_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".deft"):
                deft_files.append(os.path.join(root, file))
    return deft_files

def process_directory_and_parse(directory: str) -> List[Dict]:
    """Обработка всех .deft файлов из указанной директории и преобразование их в формат HF"""
    deft_files = find_deft_files(directory)
    all_data = []

    for file_path in deft_files:
        file_data = parse_conll_file(file_path)  # Парсим каждый файл
        all_data.extend(file_data)  # Добавляем данные к общему списку

    return all_data

def main():

    parser = argparse.ArgumentParser(description="Обработка DEFT файлов и преобразование в HF Dataset")
    
    parser.add_argument(
        "--input_folder",
        type=str,
        default="deft_corpus/data/deft_files",
        help="Путь к папке с файлами формата .deft (default: 'deft_corpus/data/deft_files')"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="deft_corpus.json",
        help="Имя выходного файла (default: 'deft_corpus.json')"
    )
    parser.add_argument(
        "--ru_folder",
        type=str,
        default=None,
        help="Путь к папке с русскими текстами. Если не указано, русский текст добавляться не будет"
    )

    args = parser.parse_args()

    # Обработка директории и получение всех данных
    all_data = process_directory_and_parse(args.input_folder)
    
    # Добавляем русский текст, если указана папка
    if args.ru_folder:
        all_data = add_ru_eng(all_data, args.ru_folder)
    
    # Удаляем временные поля
    all_data = remove_temp_fields(all_data)

    # Конвертация данных в формат Dataset
    hf_dataset = Dataset.from_list(all_data)
    hf_dataset.to_json(args.output_file)

    print(f"Обработано записей: {len(all_data)}")
    print(f"Данные сохранены в файл: {args.output_file}")

if __name__ == '__main__':
    main()
