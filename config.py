# config.py
import os

# Основные настройки
CONFIG = {
    'MAX_CONTEXT_MESSAGES': 8,
    'MAX_CONTEXT_LENGTH': 3000,
    'MODEL_TIMEOUT': 180,
    'DEFAULT_MODEL': 'llama3',
    'BACKUP_INTERVAL': 300,
    'MAX_MESSAGE_LENGTH': 3900,
    'LOG_LEVEL': 'INFO',
    'RATE_LIMIT': 5,
    'TYPING_DELAY': 0.5,
    'MODEL_OPTIONS': {
        'temperature': 0.7,
        'num_ctx': 2048,
        'top_k': 40,
        'top_p': 0.9
    },
    'MODEL_PRIORITY': {
        'llama3': 1,
        'mistral': 2,
        'codellama': 3
    },
    'MODEL_HEALTH_CHECK': True,
    
    # Настройки бенчмарка
    'BENCHMARK_TASKS': [
        {
            'id': 'math_problem',
            'name': 'Математическая задача',
            'prompt': 'Реши задачу: У Маши было 5 яблок, она купила еще 3. Затем она отдала 2 яблока другу. Сколько яблок осталось у Маши?',
            'expected_answer': '6',
            'timeout': 30,
            'max_tokens': 100,
            'weight': 1.0
        },
        {
            'id': 'logic_puzzle',
            'name': 'Логическая головоломка',
            'prompt': 'Что тяжелее: 1 кг железа или 1 кг ваты?',
            'expected_answer': 'одинаково',
            'timeout': 30,
            'max_tokens': 100,
            'weight': 1.0
        },
        {
            'id': 'translation',
            'name': 'Перевод',
            'prompt': 'Переведи на английский: "Красная машина быстро едет по дороге"',
            'expected_answer': 'red car',
            'timeout': 30,
            'max_tokens': 50,
            'weight': 1.0
        },
        {
            'id': 'code_generation',
            'name': 'Генерация кода',
            'prompt': 'Напиши функцию на Python для вычисления факториала числа',
            'expected_answer': 'def factorial',
            'timeout': 60,
            'max_tokens': 200,
            'weight': 1.2
        },
        {
            'id': 'summarization',
            'name': 'Суммаризация',
            'prompt': 'Кратко опиши основные преимущества искусственного интеллекта в медицине',
            'expected_answer': 'диагностика',
            'timeout': 120,
            'max_tokens': 300,
            'weight': 1.3
        }
    ],
    'BENCHMARK_DEFAULT_TIMEOUT': 180,
    'BENCHMARK_TEMPERATURE': 0.1,
    'BENCHMARK_MAX_TOKENS': 500,
    'BENCHMARK_MAX_PARALLEL_TASKS': 2,
    'BENCHMARK_PROGRESS_UPDATE_INTERVAL': 5
}

# Секретные данные из переменных окружения
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    try:
        from secrets import TOKEN as SECRET_TOKEN
        TOKEN = SECRET_TOKEN
    except ImportError:
        raise ValueError("Не указан токен бота. Установите переменную окружения TELEGRAM_BOT_TOKEN или создайте файл secrets.py")

# ID администраторов
ADMIN_IDS_STR = os.getenv('ADMIN_IDS', '')
if ADMIN_IDS_STR:
    ADMIN_IDS = [int(x.strip()) for x in ADMIN_IDS_STR.split(',') if x.strip()]
else:
    try:
        from secrets import ADMIN_IDS as SECRET_ADMIN_IDS
        ADMIN_IDS = SECRET_ADMIN_IDS
    except ImportError:
        ADMIN_IDS = []

# Добавляем секреты в конфиг
CONFIG['TOKEN'] = TOKEN
CONFIG['ADMIN_IDS'] = ADMIN_IDS
