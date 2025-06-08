# root31337-ollama_tg_bot  -  Ollama Telegram Bot 🤖


Telegram-бот для взаимодействия с языковыми моделями через Ollama API с поддержкой контекста диалога и управлением моделями.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot_API-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLM-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🔥 Возможности

- Поддержка популярных моделей (Llama3, Mistral, CodeLlama)
- Контекстный диалог с настраиваемой глубиной
- Два режима генерации: потоковый и классический
- Полноценная админ-панель
- Автоматическое резервное копирование
- Подробная статистика использования

## 🛠 Технологический стек

| Компонент       | Описание                          |
|-----------------|-----------------------------------|
| Python 3.8+     | Основной язык разработки          |
| Ollama API      | Доступ к языковым моделям         |
| PTB v20+        | Python Telegram Bot библиотека    |
| Asyncio         | Асинхронная обработка запросов    |

## ⚙️ Конфигурация

Создайте `config.py` или отредактируйте настройки в коде:

```python
CONFIG = {
    'TOKEN': 'YOUR_BOT_TOKEN',          # Токен бота от @BotFather
    'ADMIN_IDS': [123456789],           # Ваш Telegram ID
    'DEFAULT_MODEL': 'llama3',          # Модель по умолчанию
    'MAX_CONTEXT_LENGTH': 3000,         # Макс. длина контекста (символов)
    'RATE_LIMIT': 5,                    # Лимит сообщений в минуту
    # Другие параметры...
}

