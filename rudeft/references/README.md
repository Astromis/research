# Конфигурационные и системные файлы

### Label Studio (LS) проект для ручного аннотирования:
- `deft_ls_interface.xml` - Конфиг интерфейса LS для переноса разметки оригинального DEFT с английского языка на русский
- `translation_deft_ls_interface.xml` - Интерфейс LS для проверки и корректировки перевода в русском датасете

### Взаимодействие с Большой языковой моделью (БЯМ, LLM)
- `model_config.json` - Конфигурационные параметры для работы с LLM (API-ключ, URL и название модели)
- `model_examples_markup_transfer.json` - Примеры запроса и ответа по переносу аннотаций между языками для использования в промпте к LLM
- `model_examples_translation_via_llm.json` - Примеры запроса и ответа по переводу предложений датасета для использования в промпте к LLM
- `system_message_markup_transfer.txt` - Системный промпт для LLM с инструкциями по переносу аннотаций
- `system_message_translation_via_llm_1.txt` и `system_message_translation_via_llm_2.txt` - Системные промпты для контроля качества перевода через LLM (два варианта)

[🏠 Назад к главному README](../README.md)
