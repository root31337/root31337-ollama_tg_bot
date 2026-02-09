import logging
import subprocess
import asyncio
import signal
import os
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from datetime import datetime
from functools import wraps
import re
import statistics
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ContextTypes
)
from telegram.error import BadRequest
import ollama
from ollama import ResponseError

# Импортируем конфигурацию
from config import CONFIG

# Настройка логирования
def setup_logging():
    logging.basicConfig(
        level=CONFIG['LOG_LEVEL'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Глобальное состояние
user_data: Dict[int, Dict] = {}
context_memory: Dict[int, Dict] = {}
rate_limit_counters: Dict[int, Tuple[int, float]] = {}
model_info_cache: Dict[str, Dict] = {}
active_requests: Dict[int, asyncio.Task] = {}
shutdown_event = asyncio.Event()
benchmark_results: Dict[str, List[Dict]] = {}
benchmark_semaphore = asyncio.Semaphore(CONFIG['BENCHMARK_MAX_PARALLEL_TASKS'])

# Класс для адаптивных таймаутов
class AdaptiveBenchmark:
    def __init__(self):
        self.task_times = defaultdict(list)
        self.task_timeouts = {}
    
    def get_timeout_for_task(self, task_id: str, default_timeout: int = None) -> int:
        """Возвращает адаптивный таймаут на основе истории"""
        default_timeout = default_timeout or CONFIG['BENCHMARK_DEFAULT_TIMEOUT']
        
        if task_id in self.task_times and self.task_times[task_id]:
            times = self.task_times[task_id][-5:]  # Последние 5 выполнений
            if times:
                avg_time = statistics.mean(times)
                adaptive_timeout = int(avg_time * 2.0) + 10  # Запас 100% + 10 секунд
                result = max(default_timeout, min(adaptive_timeout, 300))  # Ограничиваем 300 секундами
                self.task_timeouts[task_id] = result
                return result
        
        # Используем таймаут из конфига или дефолтный
        for task_config in CONFIG['BENCHMARK_TASKS']:
            if task_config['id'] == task_id:
                return task_config.get('timeout', default_timeout)
        
        return default_timeout
    
    def record_task_time(self, task_id: str, time_taken: float):
        """Записывает время выполнения задачи"""
        self.task_times[task_id].append(time_taken)
        if len(self.task_times[task_id]) > 10:
            self.task_times[task_id] = self.task_times[task_id][-10:]
    
    def get_task_stats(self) -> Dict:
        """Возвращает статистику по таймаутам"""
        stats = {}
        for task_id, times in self.task_times.items():
            if times:
                stats[task_id] = {
                    'avg_time': statistics.mean(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'count': len(times),
                    'current_timeout': self.task_timeouts.get(task_id, 'N/A')
                }
        return stats

adaptive_benchmark = AdaptiveBenchmark()

# Классы для бенчмарка
class BenchmarkTask:
    def __init__(self, id: str, name: str, prompt: str, expected_answer: str, 
                 timeout: int = None, max_tokens: int = None, weight: float = 1.0):
        self.id = id
        self.name = name
        self.prompt = prompt
        self.expected_answer = expected_answer.lower()
        self.timeout = timeout or CONFIG['BENCHMARK_DEFAULT_TIMEOUT']
        self.max_tokens = max_tokens or CONFIG['BENCHMARK_MAX_TOKENS']
        self.weight = weight
    
    def evaluate(self, response: str) -> Tuple[float, str]:
        """Оценивает ответ модели от 0 до 1"""
        response_lower = response.lower()
        
        # Простая проверка на наличие ожидаемого ответа
        if self.expected_answer in response_lower:
            return 1.0, "✅ Полный правильный ответ"
        
        # Для математических задач
        if self.id == 'math_problem':
            numbers = re.findall(r'\d+', response)
            if numbers:
                try:
                    last_number = int(numbers[-1])
                    if last_number == 6:
                        return 1.0, "✅ Правильный числовой ответ"
                    elif abs(last_number - 6) <= 2:
                        return 0.7, f"⚠️ Близкий ответ: {last_number}"
                    else:
                        return 0.3, f"⚠️ Число найдено, но неверное: {last_number}"
                except:
                    pass
        
        # Для логических задач
        if self.id == 'logic_puzzle':
            logic_keywords = ['одинаков', 'равн', 'same', 'equal', 'весят одинаково', 'одинаковый вес']
            if any(keyword in response_lower for keyword in logic_keywords):
                return 1.0, "✅ Правильный логический вывод"
            elif 'желез' in response_lower or 'железо' in response_lower:
                return 0.5, "⚠️ Частично правильный ответ"
        
        # Для перевода
        if self.id == 'translation':
            translation_keywords = ['red car', 'the red car', 'red automobile', 'красная машина']
            if any(keyword in response_lower for keyword in translation_keywords):
                return 1.0, "✅ Правильный перевод"
            elif 'red' in response_lower and 'car' in response_lower:
                return 0.8, "✅ Основные слова переведены правильно"
            elif 'красн' in response_lower and 'машин' in response_lower:
                return 0.6, "⚠️ Перевод не полный"
        
        # Для генерации кода
        if self.id == 'code_generation':
            code_patterns = ['def factorial', 'function factorial', 'factorial(n)', 'def fact']
            if any(pattern in response_lower for pattern in code_patterns):
                if 'return' in response_lower or 'рекурс' in response_lower:
                    return 1.0, "✅ Полная функция факториала"
                else:
                    return 0.8, "✅ Определение функции найдено"
            elif 'factorial' in response_lower:
                return 0.5, "⚠️ Упоминание факториала есть"
        
        # Для суммаризации
        if self.id == 'summarization':
            medical_keywords = ['диагностик', 'лечени', 'анализ', 'прогноз', 'робот', 'хирург', 
                               'diagnos', 'treat', 'analy', 'surger', 'robot']
            found_keywords = sum(1 for kw in medical_keywords if kw in response_lower)
            if found_keywords >= 3:
                return 1.0, "✅ Отличная суммаризация"
            elif found_keywords >= 2:
                return 0.8, "✅ Хорошая суммаризация"
            elif found_keywords >= 1:
                return 0.5, "⚠️ Основные идеи упомянуты"
        
        return 0.1, "❌ Ответ не соответствует ожидаемому"

@dataclass
class BenchmarkResult:
    model: str
    task_id: str
    task_name: str
    score: float
    response_time: float
    tokens_generated: int
    tokens_per_second: float
    evaluation: str
    raw_response: str
    timestamp: float
    timeout_used: int
    task_weight: float = 1.0
    
    def to_dict(self):
        return {
            'model': self.model,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'score': self.score,
            'response_time': self.response_time,
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': self.tokens_per_second,
            'evaluation': self.evaluation,
            'timestamp': self.timestamp,
            'timeout_used': self.timeout_used,
            'task_weight': self.task_weight
        }
    
    @property
    def weighted_score(self):
        return self.score * self.task_weight

class BenchmarkStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Декораторы
def rate_limit_check(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if check_rate_limit(user_id):
            await update.message.reply_text("⚠️ Вы превысили лимит сообщений. Пожалуйста, подождите минуту.")
            return
        return await func(update, context)
    return wrapper

def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BadRequest as e:
            # Обрабатываем ошибку "Message is not modified"
            if "Message is not modified" in str(e):
                # Игнорируем эту ошибку, просто отвечаем на callback_query если он есть
                query = None
                for arg in args:
                    if isinstance(arg, Update) and hasattr(arg, 'callback_query'):
                        query = arg.callback_query
                        break
                
                if query:
                    try:
                        await query.answer()
                    except:
                        pass
                return
            # Для других BadRequest логируем
            logger.error(f"BadRequest in {func.__name__}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            
            update = None
            for arg in args:
                if isinstance(arg, Update):
                    update = arg
                    break
            
            if update:
                try:
                    if hasattr(update, 'callback_query') and update.callback_query:
                        await update.callback_query.message.reply_text("⚠️ Ошибка, попробуйте снова")
                    elif hasattr(update, 'message') and update.message:
                        await update.message.reply_text("⚠️ Ошибка при обработке запроса")
                except Exception as send_error:
                    logger.error(f"Ошибка отправки сообщения: {send_error}")
    return wrapper

def cancel_previous_request(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id in active_requests:
            active_requests[user_id].cancel()
            try:
                await active_requests[user_id]
            except asyncio.CancelledError:
                pass
        task = asyncio.create_task(func(update, context))
        active_requests[user_id] = task
        try:
            return await task
        finally:
            active_requests.pop(user_id, None)
    return wrapper

# Утилиты
def setup_directories():
    """Создает необходимые директории"""
    os.makedirs('backups', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    os.makedirs('benchmarks', exist_ok=True)

async def backup_data():
    """Асинхронное создание резервных копий"""
    try:
        timestamp = int(time.time())
        async with asyncio.Lock():
            with open(f'backups/user_data_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            with open(f'backups/context_memory_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(context_memory, f, ensure_ascii=False, indent=2)
            
            backup_files = sorted([f for f in os.listdir('backups') if f.startswith('user_data_')])
            for old_file in backup_files[:-5]:
                os.remove(f'backups/{old_file}')
            
            logger.info("Резервное копирование данных завершено")
    except Exception as e:
        logger.error(f"Ошибка резервного копирования: {e}")

async def backup_task(context: ContextTypes.DEFAULT_TYPE):
    """Периодическое резервное копирование"""
    if shutdown_event.is_set():
        return
    
    await backup_data()

async def load_backup():
    """Загрузка последней резервной копии"""
    try:
        backup_files = sorted([f for f in os.listdir('backups') if f.startswith('user_data_')])
        if backup_files:
            with open(f'backups/{backup_files[-1]}', 'r', encoding='utf-8') as f:
                global user_data
                user_data = json.load(f)
        
        backup_files = sorted([f for f in os.listdir('backups') if f.startswith('context_memory_')])
        if backup_files:
            with open(f'backups/{backup_files[-1]}', 'r', encoding='utf-8') as f:
                global context_memory
                context_memory = json.load(f)
        
        logger.info("Резервные данные загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки резервной копии: {e}")

def check_rate_limit(user_id: int) -> bool:
    """Проверка ограничения скорости запросов"""
    current_time = time.time()
    count, last_time = rate_limit_counters.get(user_id, (0, current_time))
    
    if current_time - last_time > 60:
        rate_limit_counters[user_id] = (1, current_time)
        return False
    
    if count >= CONFIG['RATE_LIMIT']:
        return True
    
    rate_limit_counters[user_id] = (count + 1, last_time)
    return False

async def get_available_models(refresh: bool = False) -> List[str]:
    """Получение списка доступных моделей с кэшированием"""
    if not refresh and model_info_cache:
        return sorted(
            list(model_info_cache.keys()),
            key=lambda x: CONFIG['MODEL_PRIORITY'].get(x.split(':')[0], 10)
        )
    
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
        
        for model in models:
            if model not in model_info_cache:
                model_info_cache[model] = {
                    'name': model, 
                    'last_used': None,
                    'usage_count': 0,
                    'benchmark_score': 0.0,
                    'benchmark_tested': False,
                    'benchmark_runs': 0,
                    'last_benchmark': None
                }
        
        return sorted(
            models,
            key=lambda x: CONFIG['MODEL_PRIORITY'].get(x.split(':')[0], 10)
        )
    except Exception as e:
        logger.error(f"Ошибка получения списка моделей: {e}")
        return []

async def get_model_info(model: str) -> bool:
    """Проверка доступности модели"""
    try:
        await asyncio.to_thread(ollama.show, model=model)
        return True
    except Exception:
        return False

def split_long_message(text: str, max_length: int = None) -> List[str]:
    """Разбивка длинного сообщения на части"""
    max_length = max_length or CONFIG['MAX_MESSAGE_LENGTH']
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break
        
        for delimiter in ['\n\n', '\n', '. ', '! ', '? ', ' ', '']:
            split_pos = text.rfind(delimiter, 0, max_length)
            if split_pos > 0:
                parts.append(text[:split_pos + len(delimiter)])
                text = text[split_pos + len(delimiter):]
                break
        else:
            parts.append(text[:max_length])
            text = text[max_length:]
    
    return parts

def trim_context(context: List[Dict], max_length: int = None) -> List[Dict]:
    """Обрезка контекста до максимальной длины"""
    max_length = max_length or CONFIG['MAX_CONTEXT_LENGTH']
    total_length = sum(len(msg['content']) for msg in context)
    
    while total_length > max_length and len(context) > 1:
        removed = context.pop(0)
        total_length -= len(removed['content'])
    
    return context

def initialize_user(user_id: int):
    """Инициализация данных пользователя"""
    if user_id not in user_data:
        user_data[user_id] = {
            'current_model': CONFIG['DEFAULT_MODEL'],
            'preferences': {
                'save_context': True,
                'markdown_formatting': True,
                'notifications': True,
                'typing_indicator': True,
                'stream_responses': False
            },
            'stats': {
                'messages_sent': 0,
                'models_used': {},
                'last_active': time.time(),
                'total_tokens': 0,
                'total_requests': 0,
                'benchmarks_run': 0
            },
            'is_admin': user_id in CONFIG['ADMIN_IDS'],
            'created_at': time.time(),
            'benchmark_status': BenchmarkStatus.NOT_STARTED.value,
            'current_benchmark': None,
            'benchmark_history': []
        }
    
    if user_id not in context_memory:
        context_memory[user_id] = {}
    
    current_model = user_data[user_id]['current_model']
    if current_model not in context_memory[user_id]:
        context_memory[user_id][current_model] = []

async def send_long_message(update: Update, text: str, parse_mode: str = None):
    """Отправка длинных сообщений с разбивкой"""
    parts = split_long_message(text)
    for i, part in enumerate(parts):
        if i == 0:
            await update.message.reply_text(part, parse_mode=parse_mode)
        else:
            await update.effective_chat.send_message(part, parse_mode=parse_mode)

# Функции бенчмарка
class BenchmarkProgressTracker:
    def __init__(self, query, total_tasks: int):
        self.query = query
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.current_model = ""
        self.current_task = ""
    
    def increment(self, model: str = "", task: str = ""):
        self.completed_tasks += 1
        self.current_model = model
        self.current_task = task
    
    async def update_display(self):
        """Обновляет отображение прогресса"""
        current_time = time.time()
        # Обновляем не чаще чем раз в N секунд
        if current_time - self.last_update_time < CONFIG['BENCHMARK_PROGRESS_UPDATE_INTERVAL']:
            return
        
        self.last_update_time = current_time
        progress_percent = (self.completed_tasks / self.total_tasks) * 100
        progress_bar = "█" * int(progress_percent / 10) + "░" * (10 - int(progress_percent / 10))
        
        elapsed_time = current_time - self.start_time
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed_time / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks
            estimated_remaining = avg_time_per_task * remaining_tasks
            time_info = f"\n⏱ Время: {elapsed_time:.0f}с | Осталось: ~{estimated_remaining:.0f}с"
        else:
            time_info = ""
        
        try:
            await self.query.edit_message_text(
                f"🏃‍♂️ *Бенчмарк выполняется...*\n\n"
                f"Прогресс: {progress_bar} {progress_percent:.1f}%\n"
                f"Завершено: {self.completed_tasks}/{self.total_tasks} заданий{time_info}\n"
                f"Текущая модель: {self.current_model}\n"
                f"Текущая задача: {self.current_task}",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.debug(f"Не удалось обновить прогресс: {e}")
    
    async def final_update(self):
        """Финальное обновление после завершения"""
        elapsed_time = time.time() - self.start_time
        try:
            await self.query.edit_message_text(
                f"✅ *Бенчмарк завершен!*\n\n"
                f"Всего заданий: {self.total_tasks}\n"
                f"Общее время: {elapsed_time:.1f} секунд\n"
                f"Среднее время на задачу: {elapsed_time/self.total_tasks:.1f}с\n\n"
                f"*Обработка результатов...*",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.debug(f"Не удалось сделать финальное обновление: {e}")

async def run_benchmark_task(model: str, task: BenchmarkTask) -> Optional[BenchmarkResult]:
    """Запуск одного тестового задания для модели"""
    async with benchmark_semaphore:
        try:
            start_time = time.time()
            
            # Используем адаптивный таймаут
            timeout = adaptive_benchmark.get_timeout_for_task(task.id, task.timeout)
            
            logger.info(f"Запуск задачи {task.name} для модели {model} с таймаутом {timeout}с")
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=model,
                    messages=[{'role': 'user', 'content': task.prompt}],
                    options={
                        **CONFIG['MODEL_OPTIONS'],
                        'temperature': CONFIG['BENCHMARK_TEMPERATURE'],
                        'num_predict': task.max_tokens
                    }
                ),
                timeout=timeout
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            adaptive_benchmark.record_task_time(task.id, response_time)
            
            answer = response['message']['content']
            
            eval_count = response.get('eval_count', 0)
            tokens_per_second = eval_count / response_time if response_time > 0 else 0
            
            score, evaluation = task.evaluate(answer)
            
            result = BenchmarkResult(
                model=model,
                task_id=task.id,
                task_name=task.name,
                score=score,
                response_time=response_time,
                tokens_generated=eval_count,
                tokens_per_second=tokens_per_second,
                evaluation=evaluation,
                raw_response=answer[:200] + "..." if len(answer) > 200 else answer,
                timestamp=time.time(),
                timeout_used=timeout,
                task_weight=task.weight
            )
            
            logger.info(f"Бенчмарк: {model} - {task.name} - Оценка: {score:.2f} - Время: {response_time:.2f}с")
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут бенчмарка для модели {model} на задаче {task.name} (таймаут: {timeout}с)")
            return None
        except Exception as e:
            logger.error(f"Ошибка бенчмарка для модели {model}: {e}", exc_info=True)
            return None

async def run_full_benchmark(models: List[str], user_id: int, query):
    """Запуск полного бенчмарка для списка моделей"""
    user_data[user_id]['benchmark_status'] = BenchmarkStatus.RUNNING.value
    
    # Создаем задачи
    tasks = []
    for task_config in CONFIG['BENCHMARK_TASKS']:
        task = BenchmarkTask(
            id=task_config['id'],
            name=task_config['name'],
            prompt=task_config['prompt'],
            expected_answer=task_config['expected_answer'],
            timeout=task_config.get('timeout', CONFIG['BENCHMARK_DEFAULT_TIMEOUT']),
            max_tokens=task_config.get('max_tokens', CONFIG['BENCHMARK_MAX_TOKENS']),
            weight=task_config.get('weight', 1.0)
        )
        tasks.append(task)
    
    results = {}
    total_tasks = len(models) * len(tasks)
    
    # Инициализируем трекер прогресса
    progress_tracker = BenchmarkProgressTracker(query, total_tasks)
    await progress_tracker.update_display()
    
    for model in models:
        model_results = []
        
        # Проверяем доступность модели
        if not await get_model_info(model):
            logger.warning(f"Модель {model} недоступна для бенчмарка")
            # Пропускаем все задачи для этой модели
            for _ in tasks:
                progress_tracker.increment(model, "Пропущена (недоступна)")
                await progress_tracker.update_display()
            continue
        
        for task in tasks:
            if shutdown_event.is_set():
                break
            
            progress_tracker.increment(model, task.name)
            
            result = await run_benchmark_task(model, task)
            if result:
                model_results.append(result)
                
                if model not in benchmark_results:
                    benchmark_results[model] = []
                benchmark_results[model].append(result.to_dict())
            
            await progress_tracker.update_display()
        
        if model_results:
            # Взвешенная средняя оценка
            total_weight = sum(r.task_weight for r in model_results)
            weighted_sum = sum(r.score * r.task_weight for r in model_results)
            avg_score = weighted_sum / total_weight if total_weight > 0 else 0
            
            avg_time = statistics.mean([r.response_time for r in model_results])
            avg_tps = statistics.mean([r.tokens_per_second for r in model_results])
            
            results[model] = {
                'avg_score': avg_score,
                'avg_time': avg_time,
                'avg_tps': avg_tps,
                'weighted_score': avg_score,
                'results': model_results
            }
            
            # Обновляем информацию о модели
            if model in model_info_cache:
                model_info_cache[model]['benchmark_score'] = avg_score
                model_info_cache[model]['benchmark_tested'] = True
                model_info_cache[model]['benchmark_runs'] = model_info_cache[model].get('benchmark_runs', 0) + 1
                model_info_cache[model]['last_benchmark'] = time.time()
    
    # Финальное обновление прогресса
    await progress_tracker.final_update()
    
    # Сохраняем результаты
    if results:
        await save_benchmark_results(results, user_id)
    
    user_data[user_id]['benchmark_status'] = BenchmarkStatus.COMPLETED.value
    user_data[user_id]['stats']['benchmarks_run'] += 1
    
    # Сохраняем историю бенчмарка
    benchmark_history_entry = {
        'timestamp': time.time(),
        'models_tested': list(results.keys()),
        'total_tasks': total_tasks,
        'results_summary': {model: data['avg_score'] for model, data in results.items()}
    }
    user_data[user_id]['benchmark_history'].append(benchmark_history_entry)
    
    return results

async def save_benchmark_results(results: Dict, user_id: int):
    """Сохранение результатов бенчмарка в файл"""
    try:
        timestamp = int(time.time())
        filename = f'benchmarks/benchmark_{user_id}_{timestamp}.json'
        
        data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'results': {},
            'summary': {},
            'adaptive_timeouts': adaptive_benchmark.task_times,
            'task_stats': adaptive_benchmark.get_task_stats()
        }
        
        for model, model_data in results.items():
            data['results'][model] = {
                'avg_score': model_data['avg_score'],
                'weighted_score': model_data['weighted_score'],
                'avg_time': model_data['avg_time'],
                'avg_tps': model_data['avg_tps'],
                'task_results': [r.to_dict() for r in model_data['results']]
            }
        
        # Создаем сводку
        sorted_models = sorted(results.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        data['summary'] = {
            'top_model': sorted_models[0][0] if sorted_models else None,
            'ranking': [{'model': m, 'score': d['weighted_score'], 'raw_score': d['avg_score']} 
                       for m, d in sorted_models],
            'total_models': len(results),
            'timestamp': timestamp
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты бенчмарка сохранены в {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Ошибка сохранения результатов бенчмарка: {e}")
        return None

def format_benchmark_results(results: Dict) -> str:
    """Форматирование результатов бенчмарка для отображения"""
    if not results:
        return "❌ Нет результатов для отображения"
    
    lines = ["📊 *Результаты бенчмарка моделей*\n"]
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
    
    for i, (model, data) in enumerate(sorted_models):
        score = data['weighted_score']
        score_stars = "⭐" * int(score * 5)
        
        lines.append(f"\n*{i+1}. {model}*")
        lines.append(f"   Оценка: *{score:.2f}/1.0* {score_stars}")
        lines.append(f"   Среднее время: {data['avg_time']:.2f}с")
        lines.append(f"   Токенов/сек: {data['avg_tps']:.1f}")
        
        if 'raw_score' in data:
            lines.append(f"   Сырая оценка: {data['raw_score']:.2f}")
        
        lines.append(f"   *Детали по задачам:*")
        for task_result in data['results']:
            status = "✅" if task_result.score >= 0.7 else "⚠️" if task_result.score >= 0.3 else "❌"
            weight_marker = f" ⚖️{task_result.task_weight}" if task_result.task_weight != 1.0 else ""
            timeout_info = f" (⏱{task_result.timeout_used}с)" if hasattr(task_result, 'timeout_used') else ""
            
            lines.append(f"   {status}{weight_marker} {task_result.task_name}:")
            lines.append(f"     - Оценка: {task_result.score:.1f}{timeout_info}")
            lines.append(f"     - Время: {task_result.response_time:.1f}с")
            lines.append(f"     - Вердикт: {task_result.evaluation}")
    
    # Сводная таблица
    lines.append("\n*📈 Сводная таблица:*")
    lines.append("```")
    lines.append(f"{'Модель':<20} {'Оценка':<8} {'Время':<8} {'T/s':<8}")
    lines.append("-" * 50)
    for model, data in sorted_models:
        lines.append(f"{model:<20} {data['weighted_score']:.2f}     {data['avg_time']:.2f}с    {data['avg_tps']:.1f}")
    lines.append("```")
    
    # Статистика таймаутов
    timeout_stats = adaptive_benchmark.get_task_stats()
    if timeout_stats:
        lines.append("\n*⏱ Статистика таймаутов:*")
        for task_id, stats in timeout_stats.items():
            task_name = next((t['name'] for t in CONFIG['BENCHMARK_TASKS'] if t['id'] == task_id), task_id)
            lines.append(f"  • {task_name}: ср. {stats['avg_time']:.1f}с, таймаут {stats['current_timeout']}с")
    
    return "\n".join(lines)

async def get_model_leaderboard(min_tests: int = 0) -> str:
    """Получение таблицы лидеров моделей"""
    # Получаем все модели, которые проходили бенчмарк
    tested_models = {}
    
    # Сначала проверяем benchmark_results
    for model, results in benchmark_results.items():
        if results:
            # Берем последний результат
            latest_result = max(results, key=lambda x: x.get('timestamp', 0))
            score = latest_result.get('score', 0)
            tests = len(results)
            
            if tests >= min_tests:
                tested_models[model] = {
                    'benchmark_score': score,
                    'benchmark_tested': True,
                    'benchmark_runs': tests,
                    'last_benchmark': latest_result.get('timestamp'),
                    'usage_count': model_info_cache.get(model, {}).get('usage_count', 0)
                }
    
    # Затем добавляем из model_info_cache
    for model, info in model_info_cache.items():
        if info.get('benchmark_tested', False):
            if model not in tested_models:
                score = info.get('benchmark_score', 0)
                tests = info.get('benchmark_runs', 1)
                
                if tests >= min_tests:
                    tested_models[model] = {
                        'benchmark_score': score,
                        'benchmark_tested': True,
                        'benchmark_runs': tests,
                        'last_benchmark': info.get('last_benchmark'),
                        'usage_count': info.get('usage_count', 0)
                    }
    
    if not tested_models:
        return "📊 *Таблица лидеров*\n\nПока нет протестированных моделей. Запустите бенчмарк командой /benchmark!"
    
    # Сортируем модели по оценке
    sorted_models = sorted(
        tested_models.items(),
        key=lambda x: x[1].get('benchmark_score', 0),
        reverse=True
    )
    
    lines = ["🏆 *Таблица лидеров моделей*\n"]
    
    # Общая статистика
    total_tests = sum(info.get('benchmark_runs', 0) for _, info in tested_models.items())
    avg_score = statistics.mean([info.get('benchmark_score', 0) for _, info in tested_models.items()])
    lines.append(f"Всего моделей: {len(tested_models)} | Всего тестов: {total_tests} | Средняя оценка: {avg_score:.2f}\n")
    
    lines.append("```")
    lines.append(f"{'Место':<6} {'Модель':<25} {'Оценка':<8} {'Тестов':<8} {'Исп.':<6} {'Последний':<10}")
    lines.append("-" * 70)
    
    for i, (model, data) in enumerate(sorted_models[:15]):  # Показываем топ-15
        score = data.get('benchmark_score', 0)
        tests = data.get('benchmark_runs', 0)
        uses = data.get('usage_count', 0)
        last_test = data.get('last_benchmark')
        
        last_str = ""
        if last_test:
            last_dt = datetime.fromtimestamp(last_test)
            last_str = last_dt.strftime('%d.%m %H:%M')
        
        medal = ""
        if i == 0:
            medal = "🥇 "
        elif i == 1:
            medal = "🥈 "
        elif i == 2:
            medal = "🥉 "
        elif i < 10:
            medal = f"{i+1} "
        
        score_str = f"{score:.2f}"
        tests_str = f"{tests}"
        uses_str = f"{uses}"
        
        lines.append(f"{medal:<4} {model:<25} {score_str:<8} {tests_str:<8} {uses_str:<6} {last_str:<10}")
    
    lines.append("```")
    
    # Дополнительная информация
    if len(sorted_models) > 15:
        lines.append(f"\n... и еще {len(sorted_models) - 15} моделей")
    
    # Лучшая и худшая модель
    if len(sorted_models) >= 2:
        best_model, best_data = sorted_models[0]
        worst_model, worst_data = sorted_models[-1]
        lines.append(f"\n🎯 *Лучшая модель:* {best_model} ({best_data.get('benchmark_score', 0):.2f})")
        lines.append(f"📉 *Худшая модель:* {worst_model} ({worst_data.get('benchmark_score', 0):.2f})")
    
    # Статистика по тестам
    task_stats = adaptive_benchmark.get_task_stats()
    if task_stats:
        lines.append(f"\n📋 *Статистика задач:*")
        for task_id, stats in task_stats.items():
            task_name = next((t['name'] for t in CONFIG['BENCHMARK_TASKS'] if t['id'] == task_id), task_id)
            lines.append(f"  • {task_name}: {stats['count']} тестов, ср. время {stats['avg_time']:.1f}с")
    
    return "\n".join(lines)

# Обработчики команд
@handle_errors
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    
    status = "✅ Вкл" if user_data[user_id]['preferences']['save_context'] else "❌ Выкл"
    stream_status = "✅ Вкл" if user_data[user_id]['preferences']['stream_responses'] else "❌ Выкл"
    
    keyboard = [
        [InlineKeyboardButton("📝 Выбрать модель", callback_data='show_models')],
        [InlineKeyboardButton("🆕 Новый диалог", callback_data='new_chat')],
        [InlineKeyboardButton(f"🧠 Контекст: {status}", callback_data='toggle_context')],
        [InlineKeyboardButton(f"🌀 Поток: {stream_status}", callback_data='toggle_stream')],
        [InlineKeyboardButton("⚙️ Настройки", callback_data='settings')],
        [InlineKeyboardButton("📊 Статистика", callback_data='show_stats')]
    ]
    
    if user_data[user_id]['is_admin']:
        keyboard.append([InlineKeyboardButton("🛠 Админ-панель", callback_data='admin_panel')])
        keyboard.append([InlineKeyboardButton("🏆 Бенчмарк", callback_data='benchmark_menu')])
    
    await update.message.reply_text(
        "🤖 *Добро пожаловать в Ollama Telegram Bot!*\n\n"
        "Я предоставляю доступ к различным языковым моделям через Ollama.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

@handle_errors
@rate_limit_check
async def benchmark_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для запуска бенчмарка"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    
    if not user_data[user_id]['is_admin']:
        await update.message.reply_text("🚫 Эта команда доступна только администраторам")
        return
    
    await show_benchmark_menu(update.callback_query if update.callback_query else None)

@handle_errors
async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для отображения таблицы лидеров"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    
    # Можно указать минимальное количество тестов
    min_tests = 0
    if context.args and context.args[0].isdigit():
        min_tests = int(context.args[0])
    
    leaderboard = await get_model_leaderboard(min_tests)
    
    # Добавляем время для уникальности
    current_time = datetime.now().strftime('%H:%M:%S')
    leaderboard_text = f"{leaderboard}\n\n_⏱ Обновлено в {current_time}_"
    
    await update.message.reply_text(leaderboard_text, parse_mode='Markdown')

@handle_errors
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для отображения статистики бенчмарка"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    
    total_benchmarks = sum(u['stats']['benchmarks_run'] for u in user_data.values())
    total_tested_models = len([m for m, info in model_info_cache.items() if info.get('benchmark_tested', False)])
    
    task_stats = adaptive_benchmark.get_task_stats()
    total_task_runs = sum(stats['count'] for stats in task_stats.values()) if task_stats else 0
    
    stats_text = (
        f"📈 *Статистика бенчмарка*\n\n"
        f"• Всего запусков бенчмарка: {total_benchmarks}\n"
        f"• Протестировано моделей: {total_tested_models}\n"
        f"• Всего выполнено задач: {total_task_runs}\n"
        f"• Активных тестов: {sum(1 for u in user_data.values() if u['benchmark_status'] == 'running')}\n"
    )
    
    if task_stats:
        stats_text += f"\n*Статистика по задачам:*\n"
        for task_id, stats in task_stats.items():
            task_name = next((t['name'] for t in CONFIG['BENCHMARK_TASKS'] if t['id'] == task_id), task_id)
            stats_text += f"• {task_name}: {stats['count']} тестов, ср. {stats['avg_time']:.1f}с\n"
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')

@handle_errors
@rate_limit_check
@cancel_previous_request
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    current_model = user_data[user_id]['current_model']
    message_text = update.message.text
    
    if CONFIG['MODEL_HEALTH_CHECK'] and not await get_model_info(current_model):
        await update.message.reply_text(f"⚠️ Модель {current_model} недоступна. Попробуйте другую модель.")
        return
    
    user_data[user_id]['stats']['messages_sent'] += 1
    user_data[user_id]['stats']['last_active'] = time.time()
    user_data[user_id]['stats']['total_requests'] += 1
    
    if current_model not in user_data[user_id]['stats']['models_used']:
        user_data[user_id]['stats']['models_used'][current_model] = 0
    user_data[user_id]['stats']['models_used'][current_model] += 1
    
    if user_data[user_id]['preferences']['save_context']:
        if current_model not in context_memory[user_id]:
            context_memory[user_id][current_model] = []
        
        context_memory[user_id][current_model].append({'role': 'user', 'content': message_text})
        context_memory[user_id][current_model] = trim_context(context_memory[user_id][current_model])
    
    if user_data[user_id]['preferences']['typing_indicator']:
        await asyncio.sleep(CONFIG['TYPING_DELAY'])
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    notification = None
    if user_data[user_id]['preferences']['notifications']:
        notification = await update.message.reply_text("🔄 Генерация ответа...")
    
    try:
        messages = context_memory[user_id][current_model] if user_data[user_id]['preferences']['save_context'] else []
        messages.append({'role': 'user', 'content': message_text})
        
        if user_data[user_id]['preferences']['markdown_formatting']:
            header = (
                f"*Модель:* {current_model}\n"
                f"*Запрос:* `{message_text[:100]}{'...' if len(message_text) > 100 else ''}`\n\n"
                f"*Ответ:*\n"
            )
            parse_mode = 'Markdown'
        else:
            header = (
                f"Модель: {current_model}\n"
                f"Запрос: {message_text[:100]}{'...' if len(message_text) > 100 else ''}\n\n"
                f"Ответ:\n"
            )
            parse_mode = None
        
        if user_data[user_id]['preferences']['stream_responses']:
            full_response = ""
            message = await update.message.reply_text(header + "⌛ Начинаю генерацию...", parse_mode=parse_mode)
            
            async def stream_response():
                nonlocal full_response
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=current_model,
                    messages=messages,
                    options=CONFIG['MODEL_OPTIONS'],
                    stream=True
                )
                
                for chunk in response:
                    if shutdown_event.is_set():
                        raise asyncio.CancelledError()
                    
                    content = chunk['message']['content']
                    full_response += content
                    
                    if len(full_response) % 50 == 0:
                        try:
                            await message.edit_text(header + full_response, parse_mode=parse_mode)
                        except:
                            pass
                
                return full_response
            
            try:
                answer = await asyncio.wait_for(
                    stream_response(),
                    timeout=CONFIG['MODEL_TIMEOUT']
                )
                await message.edit_text(header + answer, parse_mode=parse_mode)
                
                if 'eval_count' in response:
                    user_data[user_id]['stats']['total_tokens'] += response['eval_count']
                
                if user_data[user_id]['preferences']['save_context']:
                    context_memory[user_id][current_model].append({'role': 'assistant', 'content': answer})
                
            except asyncio.TimeoutError:
                await message.edit_text(header + full_response + "\n\n⚠️ Достигнут таймаут генерации", parse_mode=parse_mode)
                raise
                
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=current_model,
                    messages=messages,
                    options=CONFIG['MODEL_OPTIONS']
                ),
                timeout=CONFIG['MODEL_TIMEOUT']
            )
            
            answer = response['message']['content']
            message_parts = split_long_message(answer, CONFIG['MAX_MESSAGE_LENGTH'] - len(header))
            
            first_part = header + message_parts[0]
            if notification:
                try:
                    await notification.edit_text(first_part, parse_mode=parse_mode)
                except Exception as e:
                    logger.warning(f"Ошибка редактирования сообщения: {e}")
                    await update.message.reply_text(first_part, parse_mode=parse_mode)
            else:
                await update.message.reply_text(first_part, parse_mode=parse_mode)
            
            for part in message_parts[1:]:
                await update.effective_chat.send_message(part, parse_mode=parse_mode)
            
            if 'eval_count' in response:
                user_data[user_id]['stats']['total_tokens'] += response['eval_count']
            
            if user_data[user_id]['preferences']['save_context']:
                context_memory[user_id][current_model].append({'role': 'assistant', 'content': answer})
        
        model_info_cache[current_model]['last_used'] = time.time()
        model_info_cache[current_model]['usage_count'] = model_info_cache[current_model].get('usage_count', 0) + 1
    
    except asyncio.TimeoutError:
        error_msg = (
            "⏳ Модель долго генерирует ответ. Попробуйте:\n"
            "1. Сократить запрос\n"
            "2. Отправить его снова\n"
            "3. Использовать более легкую модель (/model)\n"
            "4. Отключить сохранение контекста (/settings)"
        )
        logger.warning(f"Таймаут для пользователя {user_id} с моделью {current_model}")
        if notification:
            await notification.edit_text(error_msg)
        else:
            await update.message.reply_text(error_msg)
    
    except ResponseError as e:
        if "Message too long" in str(e):
            error_msg = "⚠️ Ответ модели слишком длинный. Попробуйте более конкретный запрос."
        else:
            error_msg = f"⚠️ Ошибка модели: {str(e)}"
        
        logger.error(f"Ошибка модели для пользователя {user_id}: {error_msg}")
        if notification:
            await notification.edit_text(error_msg)
        else:
            await update.message.reply_text(error_msg)
    
    except Exception as e:
        error_msg = "⚠️ Произошла непредвиденная ошибка"
        logger.error(f"Ошибка обработки для пользователя {user_id}: {e}", exc_info=True)
        if notification:
            await notification.edit_text(error_msg)
        else:
            await update.message.reply_text(error_msg)

@handle_errors
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик inline-кнопок"""
    query = update.callback_query
    
    try:
        await query.answer()
    except Exception as e:
        if "Query is too old" in str(e):
            logger.warning(f"Просроченный callback: {query.data}")
            return
        raise
    
    user_id = query.from_user.id
    initialize_user(user_id)
    
    # Обработка кнопок бенчмарка
    if query.data == 'benchmark_menu':
        await show_benchmark_menu(query)
    
    elif query.data == 'run_full_benchmark':
        await run_full_benchmark_handler(update, context)
    
    elif query.data == 'show_leaderboard':
        await show_leaderboard_handler(update, context)
    
    elif query.data == 'show_benchmark_tasks':
        await show_benchmark_tasks_handler(update, context)
    
    elif query.data == 'test_single_model':
        await test_single_model_handler(update, context)
    
    elif query.data.startswith('benchmark_model_'):
        await benchmark_model_handler(update, context)
    
    elif query.data == 'benchmark_stats':
        await benchmark_stats_handler(update, context)
    
    elif query.data == 'clear_benchmark_cache':
        await clear_benchmark_cache_handler(update, context)
    
    # ... остальные обработчики кнопок
    elif query.data == 'toggle_context':
        user_data[user_id]['preferences']['save_context'] = not user_data[user_id]['preferences']['save_context']
        status = "✅ Вкл" if user_data[user_id]['preferences']['save_context'] else "❌ Выкл"
        await show_main_menu(query, f"🧠 Контекст: {status}")
    
    elif query.data == 'toggle_stream':
        user_data[user_id]['preferences']['stream_responses'] = not user_data[user_id]['preferences']['stream_responses']
        status = "✅ Вкл" if user_data[user_id]['preferences']['stream_responses'] else "❌ Выкл"
        await show_main_menu(query, f"🌀 Потоковая генерация: {status}")
    
    elif query.data == 'show_models':
        await show_models_menu(query)
    
    elif query.data == 'refresh_models':
        await query.edit_message_text("🔄 Обновление списка моделей...")
        await show_models_menu(query, refresh=True)
    
    elif query.data == 'new_chat':
        context_memory[user_id] = {}
        await show_main_menu(query, "🆕 Новый диалог начат. Контекст очищен.")
    
    elif query.data.startswith('model_'):
        model_name = query.data[6:]
        if CONFIG['MODEL_HEALTH_CHECK'] and not await get_model_info(model_name):
            await query.answer(f"⚠️ Модель {model_name} недоступна", show_alert=True)
            return
        
        user_data[user_id]['current_model'] = model_name
        model_info_cache[model_name]['last_used'] = time.time()
        model_info_cache[model_name]['usage_count'] = model_info_cache[model_name].get('usage_count', 0) + 1
        await show_main_menu(query, f"✅ Выбрана модель: {model_name}")
    
    elif query.data == 'settings':
        await show_settings_menu(query)
    
    elif query.data == 'toggle_markdown':
        user_data[user_id]['preferences']['markdown_formatting'] = not user_data[user_id]['preferences']['markdown_formatting']
        await show_settings_menu(query)
    
    elif query.data == 'toggle_notifications':
        user_data[user_id]['preferences']['notifications'] = not user_data[user_id]['preferences']['notifications']
        await show_settings_menu(query)
    
    elif query.data == 'toggle_typing':
        user_data[user_id]['preferences']['typing_indicator'] = not user_data[user_id]['preferences']['typing_indicator']
        await show_settings_menu(query)
    
    elif query.data == 'show_stats':
        await show_stats_menu(query)
    
    elif query.data == 'admin_panel':
        if user_data[user_id]['is_admin']:
            await show_admin_panel(query)
        else:
            await query.answer("🚫 У вас нет прав администратора", show_alert=True)
    
    elif query.data == 'back_to_menu':
        await show_main_menu(query)
    
    elif query.data == 'force_backup':
        if user_data[user_id]['is_admin']:
            await query.answer("🔄 Создание бэкапа...", show_alert=False)
            await backup_data()
            await query.answer("✅ Бэкап создан", show_alert=False)
        else:
            await query.answer("🚫 У вас нет прав администратора", show_alert=True)

@handle_errors
async def show_main_menu(query, text: str = None):
    """Отображение главного меню"""
    user_id = query.from_user.id
    status = "✅ Вкл" if user_data[user_id]['preferences']['save_context'] else "❌ Выкл"
    stream_status = "✅ Вкл" if user_data[user_id]['preferences']['stream_responses'] else "❌ Выкл"
    
    keyboard = [
        [InlineKeyboardButton("📝 Выбрать модель", callback_data='show_models')],
        [InlineKeyboardButton("🆕 Новый диалог", callback_data='new_chat')],
        [InlineKeyboardButton(f"🧠 Контекст: {status}", callback_data='toggle_context')],
        [InlineKeyboardButton(f"🌀 Поток: {stream_status}", callback_data='toggle_stream')],
        [InlineKeyboardButton("⚙️ Настройки", callback_data='settings')],
        [InlineKeyboardButton("📊 Статистика", callback_data='show_stats')]
    ]
    
    if user_data[user_id]['is_admin']:
        keyboard.append([InlineKeyboardButton("🛠 Админ-панель", callback_data='admin_panel')])
        keyboard.append([InlineKeyboardButton("🏆 Бенчмарк", callback_data='benchmark_menu')])
    
    try:
        await query.edit_message_text(
            text=text or "Главное меню",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.warning(f"Ошибка обновления меню: {e}")

@handle_errors
async def show_models_menu(query, refresh: bool = False):
    """Отображение меню выбора моделей"""
    user_id = query.from_user.id
    models = await get_available_models(refresh=refresh)
    
    if not models:
        await query.edit_message_text(
            "❌ Нет доступных моделей. Убедитесь, что Ollama запущен.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Обновить", callback_data='refresh_models')],
                [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')]
            ])
        )
        return
    
    buttons = []
    for model in models:
        model_info = model_info_cache.get(model, {})
        last_used = model_info.get('last_used')
        usage_count = model_info.get('usage_count', 0)
        benchmark_score = model_info.get('benchmark_score', 0)
        
        score_text = f" ⭐{benchmark_score:.1f}" if benchmark_score > 0 else ""
        
        if last_used:
            last_used_str = datetime.fromtimestamp(last_used).strftime('%d.%m')
            label = f"{model}{score_text} ({last_used_str}, {usage_count}x)"
        else:
            label = f"{model}{score_text} (новый)"
        
        buttons.append(InlineKeyboardButton(label, callback_data=f"model_{model}"))
    
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    keyboard.append([InlineKeyboardButton("🔄 Обновить список", callback_data='refresh_models')])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')])
    
    await query.edit_message_text(
        "Выберите модель (⭐оценка, дата последнего использования, количество использований):",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

@handle_errors
async def show_settings_menu(query):
    """Отображение меню настроек"""
    user_id = query.from_user.id
    prefs = user_data[user_id]['preferences']
    
    settings_text = (
        "⚙️ *Настройки*\n\n"
        f"• 🧠 Сохранение контекста: {'✅ Вкл' if prefs['save_context'] else '❌ Выкл'}\n"
        f"• 📝 Markdown форматирование: {'✅ Вкл' if prefs['markdown_formatting'] else '❌ Выкл'}\n"
        f"• 🔔 Уведомления: {'✅ Вкл' if prefs['notifications'] else '❌ Выкл'}\n"
        f"• ✍️ Индикатор набора: {'✅ Вкл' if prefs['typing_indicator'] else '❌ Выкл'}\n"
        f"• 🌀 Потоковая генерация: {'✅ Вкл' if prefs['stream_responses'] else '❌ Выкл'}"
    )
    
    keyboard = [
        [InlineKeyboardButton("Переключить контекст", callback_data='toggle_context')],
        [InlineKeyboardButton("Переключить Markdown", callback_data='toggle_markdown')],
        [InlineKeyboardButton("Переключить уведомления", callback_data='toggle_notifications')],
        [InlineKeyboardButton("Переключить индикатор", callback_data='toggle_typing')],
        [InlineKeyboardButton("Переключить поток", callback_data='toggle_stream')],
        [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')]
    ]
    
    await query.edit_message_text(
        settings_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

@handle_errors
async def show_stats_menu(query):
    """Отображение статистики"""
    user_id = query.from_user.id
    stats = user_data[user_id]['stats']
    
    models_used = "\n".join(
        f"• {model}: {count} сообщ." 
        for model, count in sorted(
            stats['models_used'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    )
    
    stats_text = (
        f"📊 *Статистика*\n\n"
        f"• Всего сообщений: {stats['messages_sent']}\n"
        f"• Всего запросов: {stats['total_requests']}\n"
        f"• Использованные модели:\n{models_used}\n"
        f"• Всего токенов: {stats.get('total_tokens', 0)}\n"
        f"• Последняя активность: {datetime.fromtimestamp(stats['last_active']).strftime('%d.%m %H:%M')}\n"
        f"• Текущая модель: {user_data[user_id]['current_model']}\n"
        f"• Активен с: {datetime.fromtimestamp(user_data[user_id]['created_at']).strftime('%d.%m.%Y')}"
    )
    
    if user_data[user_id]['is_admin']:
        stats_text += f"\n• Запусков бенчмарка: {stats.get('benchmarks_run', 0)}"
    
    await query.edit_message_text(
        stats_text,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')]
        ]),
        parse_mode='Markdown'
    )

@handle_errors
async def show_admin_panel(query):
    """Админ-панель"""
    user_id = query.from_user.id
    if not user_data[user_id]['is_admin']:
        await query.answer("🚫 У вас нет прав администратора", show_alert=True)
        return
    
    total_users = len(user_data)
    active_users = sum(1 for uid, data in user_data.items() 
                      if time.time() - data['stats']['last_active'] < 86400)
    total_tokens = sum(u['stats'].get('total_tokens', 0) for u in user_data.values())
    total_models = len(model_info_cache)
    
    tested_models = [m for m, info in model_info_cache.items() if info.get('benchmark_tested', False)]
    top_benchmark_models = sorted(
        [(m, model_info_cache[m]) for m in tested_models],
        key=lambda x: x[1].get('benchmark_score', 0),
        reverse=True
    )[:3]
    
    benchmark_info = ""
    if top_benchmark_models:
        benchmark_info = "\n*🏆 Топ моделей по бенчмарку:*\n"
        for model, info in top_benchmark_models:
            score = info.get('benchmark_score', 0)
            runs = info.get('benchmark_runs', 0)
            benchmark_info += f"• {model}: {score:.2f}/1.0 ({runs} тестов)\n"
    
    admin_text = (
        "🛠 *Админ-панель*\n\n"
        f"• Всего пользователей: {total_users}\n"
        f"• Активных за сутки: {active_users}\n"
        f"• Загружено моделей: {total_models}\n"
        f"• Использовано токенов: {total_tokens}\n"
        f"• Протестировано моделей: {len(tested_models)}\n"
        f"{benchmark_info}"
    )
    
    keyboard = [
        [InlineKeyboardButton("🏆 Бенчмарк моделей", callback_data='benchmark_menu')],
        [InlineKeyboardButton("📈 Статистика бенчмарка", callback_data='benchmark_stats')],
        [InlineKeyboardButton("🔄 Обновить данные", callback_data='admin_panel')],
        [InlineKeyboardButton("💾 Создать бэкап", callback_data='force_backup')],
        [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')]
    ]
    
    await query.edit_message_text(
        admin_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# Новые обработчики для бенчмарка
@handle_errors
async def benchmark_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик меню бенчмарка"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    await show_benchmark_menu(query)

async def show_benchmark_menu(query, text: str = None):
    """Отображение меню бенчмарка"""
    user_id = query.from_user.id
    
    # Получаем статистику
    tested_models = len([m for m, info in model_info_cache.items() if info.get('benchmark_tested', False)])
    total_benchmarks = sum(u['stats']['benchmarks_run'] for u in user_data.values())
    
    menu_text = text or (
        f"🏆 *Меню бенчмаркинга*\n\n"
        f"Протестировано моделей: {tested_models}\n"
        f"Всего запусков: {total_benchmarks}\n\n"
        f"Здесь вы можете тестировать и сравнивать производительность разных моделей."
    )
    
    keyboard = [
        [InlineKeyboardButton("🚀 Запустить полный бенчмарк", callback_data='run_full_benchmark')],
        [InlineKeyboardButton("📊 Показать лидерборд", callback_data='show_leaderboard')],
        [InlineKeyboardButton("📋 Список тестовых заданий", callback_data='show_benchmark_tasks')],
        [InlineKeyboardButton("🧪 Протестировать одну модель", callback_data='test_single_model')],
        [InlineKeyboardButton("📈 Статистика бенчмарка", callback_data='benchmark_stats')],
        [InlineKeyboardButton("🗑️ Очистить кэш бенчмарка", callback_data='clear_benchmark_cache')],
        [InlineKeyboardButton("🔙 Назад", callback_data='admin_panel')]
    ]
    
    await query.edit_message_text(
        menu_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

@handle_errors
async def run_full_benchmark_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запуска полного бенчмарка"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    if user_data[user_id]['benchmark_status'] == BenchmarkStatus.RUNNING.value:
        await query.answer("⚠️ Бенчмарк уже запущен", show_alert=True)
        return
    
    models = await get_available_models(refresh=True)
    if not models:
        await query.edit_message_text(
            "❌ Нет доступных моделей для тестирования.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data='benchmark_menu')]
            ])
        )
        return
    
    total_tasks = len(models) * len(CONFIG['BENCHMARK_TASKS'])
    estimated_time = total_tasks * 20  # Примерная оценка 20 секунд на задачу
    
    await query.edit_message_text(
        f"🚀 *Запуск полного бенчмарка*\n\n"
        f"Будут протестированы {len(models)} моделей:\n"
        + "\n".join(f"• {model}" for model in models[:10]) +
        (f"\n... и еще {len(models) - 10} моделей" if len(models) > 10 else "") +
        f"\n\nВсего заданий: {total_tasks}"
        f"\nОжидаемое время: ~{estimated_time//60} минут {estimated_time%60} секунд"
        f"\n\n*Начинаю тестирование...*",
        parse_mode='Markdown'
    )
    
    # Запускаем бенчмарк в фоне
    asyncio.create_task(run_benchmark_and_report(models, user_id, query, context))

async def run_benchmark_and_report(models: List[str], user_id: int, query, context: ContextTypes.DEFAULT_TYPE):
    """Запуск бенчмарка и отправка отчетов"""
    try:
        results = await run_full_benchmark(models, user_id, query)
        
        if results:
            report = format_benchmark_results(results)
            
            timestamp = int(time.time())
            report_filename = f'benchmarks/report_{user_id}_{timestamp}.txt'
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            if len(report) > 4000:
                parts = split_long_message(report, 4000)
                for i, part in enumerate(parts):
                    if i == 0:
                        await query.edit_message_text(
                            f"📊 *Отчет о бенчмарке* (часть {i+1}/{len(parts)})\n\n{part}",
                            parse_mode='Markdown'
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=query.message.chat_id,
                            text=f"📊 *Отчет о бенчмарке* (часть {i+1}/{len(parts)})\n\n{part}",
                            parse_mode='Markdown'
                        )
            else:
                await query.edit_message_text(
                    f"📊 *Отчет о бенчмарке*\n\n{report}",
                    parse_mode='Markdown',
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🏆 Лидерборд", callback_data='show_leaderboard')],
                        [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
                    ])
                )
        else:
            await query.edit_message_text(
                "❌ Бенчмарк завершился без результатов. Возможно, все модели недоступны.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
                ])
            )
            
    except Exception as e:
        logger.error(f"Ошибка при выполнении бенчмарка: {e}", exc_info=True)
        await query.edit_message_text(
            f"❌ Ошибка при выполнении бенчмарка: {str(e)}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
            ])
        )

@handle_errors
async def show_leaderboard_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик отображения лидерборда"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    # Получаем лидерборд
    leaderboard = await get_model_leaderboard()
    
    # Добавляем время обновления для уникальности
    current_time = datetime.now().strftime('%H:%M:%S')
    leaderboard_text = f"{leaderboard}\n\n_⏱ Обновлено в {current_time}_"
    
    await query.edit_message_text(
        leaderboard_text,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data='show_leaderboard')],
            [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
        ])
    )

@handle_errors
async def show_benchmark_tasks_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик отображения тестовых заданий"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    tasks_text = "📋 *Тестовые задания бенчмарка*\n\n"
    for i, task in enumerate(CONFIG['BENCHMARK_TASKS']):
        weight = task.get('weight', 1.0)
        weight_text = f" ⚖️{weight}" if weight != 1.0 else ""
        timeout = task.get('timeout', CONFIG['BENCHMARK_DEFAULT_TIMEOUT'])
        
        tasks_text += f"*{i+1}. {task['name']}{weight_text}*\n"
        tasks_text += f"   Таймаут: {timeout}с | Токены: {task.get('max_tokens', CONFIG['BENCHMARK_MAX_TOKENS'])}\n"
        tasks_text += f"   Запрос: `{task['prompt']}`\n"
        tasks_text += f"   Ожидаемый ответ: {task['expected_answer']}\n\n"
    
    # Статистика адаптивных таймаутов
    task_stats = adaptive_benchmark.get_task_stats()
    if task_stats:
        tasks_text += "*📊 Статистика выполнения:*\n"
        for task_id, stats in task_stats.items():
            task_name = next((t['name'] for t in CONFIG['BENCHMARK_TASKS'] if t['id'] == task_id), task_id)
            tasks_text += f"  • {task_name}: {stats['count']} тестов, ср. {stats['avg_time']:.1f}с\n"
    
    # Добавляем время для уникальности
    current_time = datetime.now().strftime('%H:%M:%S')
    tasks_text_with_time = f"{tasks_text}\n\n_⏱ Просмотрено в {current_time}_"
    
    await query.edit_message_text(
        tasks_text_with_time,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
        ])
    )

@handle_errors
async def test_single_model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик тестирования одной модели"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    models = await get_available_models()
    if not models:
        await query.edit_message_text(
            "❌ Нет доступных моделей.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
            ])
        )
        return
    
    buttons = []
    for model in models[:20]:
        model_info = model_info_cache.get(model, {})
        score = model_info.get('benchmark_score', 0)
        runs = model_info.get('benchmark_runs', 0)
        
        score_text = f" ⭐{score:.1f}" if score > 0 else ""
        runs_text = f" ({runs} тестов)" if runs > 0 else ""
        
        buttons.append(
            InlineKeyboardButton(f"{model}{score_text}{runs_text}", 
                               callback_data=f"benchmark_model_{model}")
        )
    
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    keyboard.append([InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')])
    
    await query.edit_message_text(
        "🧪 *Тестирование одной модели*\n\nВыберите модель для тестирования:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

@handle_errors
async def benchmark_model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик тестирования конкретной модели"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    model_name = query.data.replace('benchmark_model_', '')
    
    await query.edit_message_text(
        f"🧪 *Тестирование модели: {model_name}*\n\nПодготовка к тестированию...",
        parse_mode='Markdown'
    )
    
    # Создаем задачи
    tasks = []
    for task_config in CONFIG['BENCHMARK_TASKS']:
        task = BenchmarkTask(
            id=task_config['id'],
            name=task_config['name'],
            prompt=task_config['prompt'],
            expected_answer=task_config['expected_answer'],
            timeout=task_config.get('timeout', CONFIG['BENCHMARK_DEFAULT_TIMEOUT']),
            max_tokens=task_config.get('max_tokens', CONFIG['BENCHMARK_MAX_TOKENS']),
            weight=task_config.get('weight', 1.0)
        )
        tasks.append(task)
    
    total_tasks = len(tasks)
    progress_tracker = BenchmarkProgressTracker(query, total_tasks)
    await progress_tracker.update_display()
    
    results = []
    
    for task in tasks:
        progress_tracker.increment(model_name, task.name)
        
        result = await run_benchmark_task(model_name, task)
        if result:
            results.append(result)
            
            # Сохраняем в benchmark_results
            if model_name not in benchmark_results:
                benchmark_results[model_name] = []
            benchmark_results[model_name].append(result.to_dict())
        
        await progress_tracker.update_display()
    
    await progress_tracker.final_update()
    
    if results:
        # Взвешенная средняя оценка
        total_weight = sum(r.task_weight for r in results)
        weighted_sum = sum(r.score * r.task_weight for r in results)
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        avg_time = statistics.mean([r.response_time for r in results])
        avg_tps = statistics.mean([r.tokens_per_second for r in results])
        
        # Обновляем информацию о модели
        if model_name in model_info_cache:
            model_info_cache[model_name]['benchmark_score'] = avg_score
            model_info_cache[model_name]['benchmark_tested'] = True
            model_info_cache[model_name]['benchmark_runs'] = model_info_cache[model_name].get('benchmark_runs', 0) + 1
            model_info_cache[model_name]['last_benchmark'] = time.time()
        
        # Формируем отчет
        report = f"📊 *Результаты тестирования: {model_name}*\n\n"
        report += f"Средняя оценка: *{avg_score:.2f}/1.0*\n"
        report += f"Среднее время ответа: *{avg_time:.2f}с*\n"
        report += f"Скорость генерации: *{avg_tps:.1f} токенов/сек*\n\n"
        report += "*Детали по задачам:*\n"
        
        for result in results:
            status = "✅" if result.score >= 0.7 else "⚠️" if result.score >= 0.3 else "❌"
            weight_marker = f" ⚖️{result.task_weight}" if result.task_weight != 1.0 else ""
            timeout_info = f" (таймаут: {result.timeout_used}с)"
            
            report += f"\n{status}{weight_marker} *{result.task_name}:* {result.score:.1f}{timeout_info}\n"
            report += f"   Время: {result.response_time:.1f}с\n"
            report += f"   Вердикт: {result.evaluation}\n"
            report += f"   Ответ: _{result.raw_response}_\n"
        
        # Сохраняем результат
        await save_benchmark_results({model_name: {
            'avg_score': avg_score,
            'weighted_score': avg_score,
            'avg_time': avg_time,
            'avg_tps': avg_tps,
            'results': results
        }}, user_id)
        
        # Показываем таблицу лидеров с этой моделью
        leaderboard = await get_model_leaderboard()
        leaderboard_lines = leaderboard.split('\n')
        short_leaderboard = '\n'.join(leaderboard_lines[:20])  # Первые 20 строк
        
        report += f"\n\n🏆 *Таблица лидеров (фрагмент):*\n{short_leaderboard}"
        
        await query.edit_message_text(
            report,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏆 Полный лидерборд", callback_data='show_leaderboard')],
                [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
            ])
        )
    else:
        await query.edit_message_text(
            f"❌ Не удалось протестировать модель {model_name}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
            ])
        )

@handle_errors
async def benchmark_stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик статистики бенчмарка"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    total_benchmarks = sum(u['stats']['benchmarks_run'] for u in user_data.values())
    total_tested_models = len([m for m, info in model_info_cache.items() if info.get('benchmark_tested', False)])
    
    task_stats = adaptive_benchmark.get_task_stats()
    total_task_runs = sum(stats['count'] for stats in task_stats.values()) if task_stats else 0
    
    # Собираем историю бенчмарков пользователя
    user_benchmarks = user_data[user_id].get('benchmark_history', [])
    user_benchmarks_text = ""
    if user_benchmarks:
        user_benchmarks_text = "\n*📅 Ваши последние бенчмарки:*\n"
        for hist in user_benchmarks[-5:]:  # Последние 5
            dt = datetime.fromtimestamp(hist['timestamp'])
            models = hist.get('models_tested', [])
            user_benchmarks_text += f"• {dt.strftime('%d.%m %H:%M')}: {len(models)} моделей\n"
    
    stats_text = (
        f"📈 *Статистика бенчмарка*\n\n"
        f"• Всего запусков бенчмарка: {total_benchmarks}\n"
        f"• Протестировано моделей: {total_tested_models}\n"
        f"• Всего выполнено задач: {total_task_runs}\n"
        f"• Активных тестов: {sum(1 for u in user_data.values() if u['benchmark_status'] == 'running')}\n"
        f"{user_benchmarks_text}"
    )
    
    if task_stats:
        stats_text += f"\n*📊 Статистика по задачам:*\n"
        for task_id, stats in task_stats.items():
            task_name = next((t['name'] for t in CONFIG['BENCHMARK_TASKS'] if t['id'] == task_id), task_id)
            current_timeout = adaptive_benchmark.task_timeouts.get(task_id, 'N/A')
            stats_text += f"• {task_name}: {stats['count']} тестов, ср. {stats['avg_time']:.1f}с, таймаут {current_timeout}с\n"
    
    # Топ моделей
    tested_models = [(m, info) for m, info in model_info_cache.items() if info.get('benchmark_tested', False)]
    if tested_models:
        sorted_models = sorted(tested_models, key=lambda x: x[1].get('benchmark_score', 0), reverse=True)
        stats_text += f"\n*🏆 Топ-5 моделей:*\n"
        for i, (model, info) in enumerate(sorted_models[:5]):
            score = info.get('benchmark_score', 0)
            runs = info.get('benchmark_runs', 0)
            stats_text += f"{i+1}. {model}: {score:.2f} ({runs} тестов)\n"
    
    # Добавляем время обновления для уникальности
    current_time = datetime.now().strftime('%H:%M:%S')
    stats_text_with_time = f"{stats_text}\n\n_⏱ Обновлено в {current_time}_"
    
    await query.edit_message_text(
        stats_text_with_time,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data='benchmark_stats')],
            [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
        ])
    )

@handle_errors
async def clear_benchmark_cache_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик очистки кэша бенчмарка"""
    query = update.callback_query
    user_id = query.from_user.id
    
    await query.answer()
    initialize_user(user_id)
    
    # Сбрасываем адаптивные таймауты
    adaptive_benchmark.task_times.clear()
    adaptive_benchmark.task_timeouts.clear()
    
    # Очищаем результаты бенчмарка (но не информацию о моделях)
    benchmark_results.clear()
    
    # Сбрасываем статус тестирования, но оставляем оценки
    for model in model_info_cache:
        if 'benchmark_tested' in model_info_cache[model]:
            model_info_cache[model]['benchmark_tested'] = False
            model_info_cache[model]['benchmark_runs'] = 0
    
    await query.edit_message_text(
        "✅ Кэш бенчмарка очищен. Адаптивные таймауты сброшены.\n\n"
        "Оценки моделей сохранены, но статус тестирования сброшен.",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 В меню", callback_data='benchmark_menu')]
        ])
    )

async def post_init(application):
    """Пост-инициализация"""
    logger.info("Инициализация бота завершена")
    if application.job_queue:
        application.job_queue.run_repeating(
            backup_task,
            interval=CONFIG['BACKUP_INTERVAL'],
            first=10
        )

async def shutdown():
    """Корректное завершение работы"""
    logger.info("Завершение работы...")
    shutdown_event.set()
    
    for task in active_requests.values():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    await backup_data()
    logger.info("Финальный бэкап создан")
    
    application = ApplicationBuilder().build()
    await application.stop()
    await application.shutdown()

def handle_signal(signum, frame):
    """Обработчик сигналов завершения"""
    logger.info(f"Получен сигнал {signum}, завершаем работу...")
    asyncio.create_task(shutdown())

async def main():
    """Основная функция"""
    setup_logging()
    setup_directories()
    
    try:
        await load_backup()
        
        application = ApplicationBuilder() \
            .token(CONFIG['TOKEN']) \
            .post_init(post_init) \
            .build()
        
        # Обработчики команд
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('help', start))
        application.add_handler(CommandHandler('model', lambda u, c: show_models_menu(u.callback_query)))
        application.add_handler(CommandHandler('stats', lambda u, c: show_stats_menu(u.callback_query)))
        application.add_handler(CommandHandler('settings', lambda u, c: show_settings_menu(u.callback_query)))
        application.add_handler(CommandHandler('benchmark', benchmark_command))
        application.add_handler(CommandHandler('leaderboard', leaderboard_command))
        application.add_handler(CommandHandler('benchmark_stats', stats_command))
        
        # Обработчики сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(button_handler))
        
        logger.info("Бот запускается...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}", exc_info=True)
    finally:
        await shutdown()
        logger.info("Бот завершил работу")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
