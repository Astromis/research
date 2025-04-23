import os
import re
from typing import List, Dict, Tuple
import chardet

def detect_encoding(file_path: str) -> str:
    """Функция для определения кодировки файла"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def load_and_combine_text_files_auto_encoding(directory: str) -> List[str]:
    """Функция для загрузки и объединения текстовых файлов с автоопределением кодировки"""
    combined_text = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                encoding = detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding) as f:
                    combined_text.extend(f.read().splitlines())
    return combined_text

def extract_russian_sentences(text: List[str]) -> Dict[str, List[str]]:
    """Функция для извлечения русских предложений из текста"""
    numbered_sentences = {}
    pattern = re.compile(r'^[«"]?(\d+)(?:\sгод)?\.')  # Паттерн для номеров предложений
    split_pattern = re.compile(r'\s?"[01]"\s?')  # Паттерн для меток "0" или "1"

    current_number = None
    current_sentences = []

    for line in text:
        line = line.strip()
        if not line:
            continue

        # Разделяем строку по меткам "0" и "1"
        parts = split_pattern.split(line)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            match = pattern.match(part)
            if match:
                if current_number is not None and current_sentences:
                    numbered_sentences[current_number] = current_sentences

                # Новый блок: извлечение номера и обнуление предложений
                current_number = match.group(1)
                current_sentences = [f"{current_number} ."]
                remaining_text = part[match.end():].strip().rstrip('».') + '.'
                if remaining_text:
                    current_sentences.append(remaining_text)
            else:
                # Очистка кавычек и добавление предложения в блок
                if current_number:
                    clean_part = part.strip('«»').rstrip('».') + '.'
                    current_sentences.append(clean_part)

    # Сохранение последнего блока
    if current_number is not None and current_sentences:
        numbered_sentences[current_number] = current_sentences

    return numbered_sentences

def extract_english_sentences(parsed_dict: List[Dict[str, str]]) -> Tuple[Dict[str, List[Tuple[int, str]]], List[Dict[str, str]]]:
    """Функция для извлечения английских предложений из словарей с инкрементными ID"""
    numbered_sentences = {}
    current_number = None
    current_sentences = []
    sentence_id = 0  # Инкрементный ID для каждого предложения

    number_pattern_with_dot = re.compile(r'^(\d+)\s\.$')

    for elem in parsed_dict:
        elem["temp_id"] = sentence_id
        text = elem["text"].strip()
        is_block_start = elem.get("is_block_start", False)

        if is_block_start:
            match = number_pattern_with_dot.match(text)
            if match:
                if current_number is not None:
                    numbered_sentences[current_number] = current_sentences

                current_number = match.group(1)
                current_sentences = [(sentence_id, text)]
                sentence_id += 1
            else:
                current_number = None
                current_sentences = []
        elif current_number is not None:
            if text:
                current_sentences.append((sentence_id, text))
                sentence_id += 1

    if current_number is not None and current_sentences:
        numbered_sentences[current_number] = current_sentences

    return numbered_sentences, parsed_dict

def match_sentences(eng_dict: Dict[str, List[Tuple[int, str]]], rus_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[int, str, str]]]:
    """Функция для сопоставления английских и русских предложений"""
    for num, eng_sents in eng_dict.items():
        if num in rus_dict and len(eng_sents) == len(rus_dict[num]):
            for i, (temp_id, eng_sent) in enumerate(eng_sents):
                eng_sents[i] = (temp_id, eng_sent, rus_dict[num][i])
        else:
            for i, (temp_id, eng_sent) in enumerate(eng_sents):
                eng_sents[i] = (temp_id, eng_sent, '')
    return eng_dict

def update_russian_text(parsed_dict: List[Dict[str, str]], matched_pairs: Dict[str, List[Tuple[int, str, str]]]) -> List[Dict[str, str]]:
    """Функция для обновления словаря английских предложений с добавлением русского текста"""
    for eng_sents in matched_pairs.values():
        for temp_id, _, rus_sent in eng_sents:
            for elem in parsed_dict:
                if elem.get("temp_id") == temp_id:
                    elem["text_rus"] = rus_sent
                    break
    return parsed_dict

def remove_temp_id(target_list_of_dicts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Функция для удаления временных ID из словаря"""
    for elem in target_list_of_dicts:
        if "temp_id" in elem:
            del elem["temp_id"]
    return target_list_of_dicts

def correspondence(eng_dict: Dict[str, List[Dict[str, str]]], rus_dir: str) -> List[Dict[str, str]]:
    """Основная функция для сопоставления английских и русских предложений"""
    combined_rus_text = load_and_combine_text_files_auto_encoding(rus_dir)
    rus_sentences = extract_russian_sentences(combined_rus_text)

    eng_sentences, parsed_dict = extract_english_sentences(eng_dict)

    matched_pairs = match_sentences(eng_sentences, rus_sentences)
    matched_dict = update_russian_text(parsed_dict, matched_pairs)

    return remove_temp_id(matched_dict)
