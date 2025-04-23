# RuDEFT: Transferring Natural Language Datasets between Languages using LLMs 🚀

<!-- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) -->
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)

**Официальный репозиторий статьи**:  
"Transferring Natural Language Datasets between Languages using Large Language Models for Modern Decision Support & Sci-Tech Analytical Systems"

## 📖 О проекте

**Цель**: Создание пайплайна для переноса наборов данных и аннотаций к ним между языками с использованием LLM.

**Полная документация пайплайна**:  
📄 [Pipeline Documentation](./docs/pipeline_documentation.md) - подробное описание скриптов, параметров запуска и примеров использования.

**Ключевые особенности**:
- Корпус DEFT - три уровня аннотации, посвящённые извлечению пар «термин-определение»
- Ручное аннотирование в Label Studio
- Метрики оценки качества перевода (BLEU, LaBSE) и качества переноса разметки
- Пайплайн для переноса аннотаций с помощью LLM 
- Обучение BERT-моделей на переведенном наборе данных для установления базового уровня качества

## 🛠 Технологии
### Основной стек
- **Python 3.11**
- **NLP-библиотеки**:
  - `langchain==0.3.1` - работа с LLM
  - `sentence-transformers==3.1.1` - эмбеддинги LaBSE
  - `datasets==2.21.0` - обработка датасетов (Hugging Face)
  
### Инфраструктура
- `pydantic==2.9.2` - валидация данных
- `fuzzysearch==0.7.3` - поиск неточных совпадений
- `scipy==1.14.0` - научные вычисления
- `evaluate==0.4.3` - оценка моделей

### Аннотация
- `label-studio==1.12.1` - ручное аннотирование

## 📂 Структура репозитория

#### `RuDEFT/`
- `docs/` - Техническая документация → [📖 README](docs/README.md)
- `notebooks/` - Анализ данных и эксперименты → [📖 README](notebooks/README.md)
- `references/` - Конфигурационные и системные файлы → [📖 README](references/README.md)
- `src/` - Скрипты → [📖 README](src/README.md)

> **Подробное описание** каждой директории доступно в соответствующих файлах README.


