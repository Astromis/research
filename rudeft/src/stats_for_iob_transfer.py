import argparse
import json
from datasets import Dataset

def compare_datasets(gold_dataset, model_dataset):
    """
    Функция для сравнения двух датасетов с учётом точного, частичного и полного несовпадения спанов
    
    :param gold_dataset: Золотой датасет
    :param model_dataset: Датасет, полученный от модели
    
    :return: Статистика сравнения (словарь)
    """
    statistics = {
        "total_entries": len(gold_dataset),
        "exact_match_count": 0,
        "partial_match_wider_count": 0,
        "partial_match_narrower_count": 0,
        "mismatched_spans_count": 0,
        "total_spans_checked": 0,
        "details": []
    }

    # Преобразуем "золотой" датасет в словарь для быстрого поиска по id
    gold_dict = {entry["id"]: entry for entry in gold_dataset}

    for model_entry in model_dataset:
        model_id = model_entry["id"]
        
        # Проверяем наличие соответствующего entry в "золотом" датасете
        if model_id not in gold_dict:
            statistics["details"].append({
                "id": model_id,
                "error": "Missing in gold dataset"
            })
            continue
        
        gold_entry = gold_dict[model_id]
        model_spans = model_entry["spans_rus"]
        gold_spans = gold_entry["spans_rus"]

        model_spans_by_label = {}
        gold_spans_by_label = {}

        # Группировка спанов по лейблу в датасете модели
        for span in model_spans:
            label = span[2]
            if label not in model_spans_by_label:
                model_spans_by_label[label] = []
            model_spans_by_label[label].append(span)
        
        # Группировка спанов по лейблу в золотом датасете
        for span in gold_spans:
            label = span[2]
            if label not in gold_spans_by_label:
                gold_spans_by_label[label] = []
            gold_spans_by_label[label].append(span)

        # Проверка совпадения спанов по лейблам
        for label in model_spans_by_label:
            if label not in gold_spans_by_label:
                statistics["mismatched_spans_count"] += 1
                statistics["details"].append({
                    "id": model_id,
                    "label": label,
                    "error": f"Label '{label}' missing in gold dataset"
                })
                continue
            
            model_spans_list = model_spans_by_label[label]
            gold_spans_list = gold_spans_by_label[label]

            # Проверка количества спанов с одинаковыми лейблами
            if len(model_spans_list) != len(gold_spans_list):
                statistics["mismatched_spans_count"] += 1
                statistics["details"].append({
                    "id": model_id,
                    "label": label,
                    "error": f"Mismatched number of spans for label '{label}' (model: {len(model_spans_list)}, gold: {len(gold_spans_list)})"
                })
                continue
            
            # Сравнение каждого спана
            for i, (model_span, gold_span) in enumerate(zip(model_spans_list, gold_spans_list)):
                statistics["total_spans_checked"] += 1

                model_start, model_end = int(model_span[0]), int(model_span[1])
                gold_start, gold_end = int(gold_span[0]), int(gold_span[1])
                
                if model_start == gold_start and model_end == gold_end:
                    statistics["exact_match_count"] += 1
                elif model_start <= gold_start and model_end >= gold_end:
                    # Спан модели шире золотого
                    statistics["partial_match_wider_count"] += 1
                    statistics["details"].append({
                        "id": model_id,
                        "label": label,
                        "model_span": model_span,
                        "gold_span": gold_span,
                        "error": "Model span is wider"
                    })
                elif model_start >= gold_start and model_end <= gold_end:
                    # Спан модели уже золотого
                    statistics["partial_match_narrower_count"] += 1
                    statistics["details"].append({
                        "id": model_id,
                        "label": label,
                        "model_span": model_span,
                        "gold_span": gold_span,
                        "error": "Model span is narrower"
                    })
                else:
                    statistics["mismatched_spans_count"] += 1
                    statistics["details"].append({
                        "id": model_id,
                        "label": label,
                        "model_span": model_span,
                        "gold_span": gold_span,
                        "error": "No match"
                    })

    return statistics


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Compare datasets and output statistics")
    parser.add_argument("--gold_dataset", type=str, default="datasets/train_with_rus.json",
                        help="Path to the gold dataset")
    parser.add_argument("--model_dataset", type=str, default="datasets/test_llm_model_converted_train_new.json",
                        help="Path to the model dataset")
    parser.add_argument("--output", type=str, default="model_iob_transfer_stats.log",
                        help="Path to save the statistics")

    args = parser.parse_args()

    gold_dataset = Dataset.from_json(args.gold_dataset)
    model_dataset = Dataset.from_json(args.model_dataset)

    # Получение статистики
    stats = compare_datasets(gold_dataset, model_dataset)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to {args.output}")


if __name__ == "__main__":
    main()
