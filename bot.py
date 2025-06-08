import logging
import subprocess
import asyncio
import signal
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ContextTypes
)
import ollama
from ollama import ResponseError
import os
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from datetime import datetime
from functools import wraps

# Конфигурация
CONFIG = {
    'TOKEN': 'YOU_TOKENS,
    'MAX_CONTEXT_MESSAGES': 8,
    'MAX_CONTEXT_LENGTH': 3000,
    'MODEL_TIMEOUT': 740,
    'DEFAULT_MODEL': 'llama3',
    'BACKUP_INTERVAL': 300,
    'MAX_MESSAGE_LENGTH': 3900,  # 4000 с запасом
    'LOG_LEVEL': 'INFO',
    'ADMIN_IDS': [ADM_IDS],
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
    'MODEL_HEALTH_CHECK': True
}

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

async def backup_data():
    """Асинхронное создание резервных копий"""
    try:
        timestamp = int(time.time())
        async with asyncio.Lock():
            with open(f'backups/user_data_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            with open(f'backups/context_memory_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(context_memory, f, ensure_ascii=False, indent=2)
            
            # Очистка старых бэкапов
            backup_files = sorted([f for f in os.listdir('backups') if f.startswith('user_data_')])
            for old_file in backup_files[:-5]:
                os.remove(f'backups/{old_file}')
            
            logger.info("Резервное копирование данных завершено")
    except Exception as e:
        logger.error(f"Ошибка резервного копирования: {e}")

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
        
        # Обновление кэша
        for model in models:
            if model not in model_info_cache:
                model_info_cache[model] = {
                    'name': model, 
                    'last_used': None,
                    'usage_count': 0
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
        
        # Пытаемся разбить по естественным разделителям
        for delimiter in ['\n\n', '\n', '. ', '! ', '? ', ' ', '']:
            split_pos = text.rfind(delimiter, 0, max_length)
            if split_pos > 0:
                parts.append(text[:split_pos + len(delimiter)])
                text = text[split_pos + len(delimiter):]
                break
        else:
            # Если разделителей нет, разбиваем по max_length
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
                'total_requests': 0
            },
            'is_admin': user_id in CONFIG['ADMIN_IDS'],
            'created_at': time.time()
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
    
    await update.message.reply_text(
        "🤖 *Добро пожаловать в Ollama Telegram Bot!*\n\n"
        "Я предоставляю доступ к различным языковым моделям через Ollama.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

@handle_errors
@rate_limit_check
@cancel_previous_request
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    initialize_user(user_id)
    current_model = user_data[user_id]['current_model']
    message_text = update.message.text
    
    # Проверка доступности модели
    if CONFIG['MODEL_HEALTH_CHECK'] and not await get_model_info(current_model):
        await update.message.reply_text(f"⚠️ Модель {current_model} недоступна. Попробуйте другую модель.")
        return
    
    # Обновление статистики
    user_data[user_id]['stats']['messages_sent'] += 1
    user_data[user_id]['stats']['last_active'] = time.time()
    user_data[user_id]['stats']['total_requests'] += 1
    
    if current_model not in user_data[user_id]['stats']['models_used']:
        user_data[user_id]['stats']['models_used'][current_model] = 0
    user_data[user_id]['stats']['models_used'][current_model] += 1
    
    # Сохранение контекста
    if user_data[user_id]['preferences']['save_context']:
        if current_model not in context_memory[user_id]:
            context_memory[user_id][current_model] = []
        
        context_memory[user_id][current_model].append({'role': 'user', 'content': message_text})
        context_memory[user_id][current_model] = trim_context(context_memory[user_id][current_model])
    
    # Индикатор набора
    if user_data[user_id]['preferences']['typing_indicator']:
        await asyncio.sleep(CONFIG['TYPING_DELAY'])
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # Уведомление о генерации
    notification = None
    if user_data[user_id]['preferences']['notifications']:
        notification = await update.message.reply_text("🔄 Генерация ответа...")
    
    try:
        messages = context_memory[user_id][current_model] if user_data[user_id]['preferences']['save_context'] else []
        messages.append({'role': 'user', 'content': message_text})
        
        # Форматирование ответа
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
            # Потоковая генерация
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
                    
                    # Обновляем сообщение каждые 50 символов
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
                
                # Обновление статистики токенов
                if 'eval_count' in response:
                    user_data[user_id]['stats']['total_tokens'] += response['eval_count']
                
                # Сохранение ответа в контекст
                if user_data[user_id]['preferences']['save_context']:
                    context_memory[user_id][current_model].append({'role': 'assistant', 'content': answer})
                
            except asyncio.TimeoutError:
                await message.edit_text(header + full_response + "\n\n⚠️ Достигнут таймаут генерации", parse_mode=parse_mode)
                raise
                
        else:
            # Стандартная генерация
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
            
            # Разбивка и отправка сообщения
            message_parts = split_long_message(answer, CONFIG['MAX_MESSAGE_LENGTH'] - len(header))
            
            # Первая часть с заголовком
            first_part = header + message_parts[0]
            if notification:
                try:
                    await notification.edit_text(first_part, parse_mode=parse_mode)
                except Exception as e:
                    logger.warning(f"Ошибка редактирования сообщения: {e}")
                    await update.message.reply_text(first_part, parse_mode=parse_mode)
            else:
                await update.message.reply_text(first_part, parse_mode=parse_mode)
            
            # Остальные части
            for part in message_parts[1:]:
                await update.effective_chat.send_message(part, parse_mode=parse_mode)
            
            # Обновление статистики токенов
            if 'eval_count' in response:
                user_data[user_id]['stats']['total_tokens'] += response['eval_count']
            
            # Сохранение ответа в контекст
            if user_data[user_id]['preferences']['save_context']:
                context_memory[user_id][current_model].append({'role': 'assistant', 'content': answer})
        
        # Обновление информации о модели
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
    user_id = query.from_user.id
    
    try:
        await query.answer()
    except Exception as e:
        if "Query is too old" in str(e):
            logger.warning(f"Просроченный callback: {query.data}")
            return
        raise
    
    initialize_user(user_id)
    
    if query.data == 'toggle_context':
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
        
        if last_used:
            last_used_str = datetime.fromtimestamp(last_used).strftime('%d.%m')
            label = f"{model} ({last_used_str}, {usage_count}x)"
        else:
            label = f"{model} (новый)"
        
        buttons.append(InlineKeyboardButton(label, callback_data=f"model_{model}"))
    
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    keyboard.append([InlineKeyboardButton("🔄 Обновить список", callback_data='refresh_models')])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')])
    
    await query.edit_message_text(
        "Выберите модель (дата последнего использования, количество использований):",
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
    
    # Топ моделей
    top_models = sorted(
        model_info_cache.items(),
        key=lambda x: x[1].get('usage_count', 0),
        reverse=True
    )[:5]
    
    models_info = "\n".join(
        f"• {model}: {info.get('usage_count', 0)} запросов, "
        f"посл. исп.: {datetime.fromtimestamp(info.get('last_used', 0)).strftime('%d.%m %H:%M') if info.get('last_used') else 'никогда'}"
        for model, info in top_models
    )
    
    admin_text = (
        "🛠 *Админ-панель*\n\n"
        f"• Всего пользователей: {total_users}\n"
        f"• Активных за сутки: {active_users}\n"
        f"• Загружено моделей: {total_models}\n"
        f"• Использовано токенов: {total_tokens}\n\n"
        f"*Топ моделей:*\n{models_info}"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔄 Обновить данные", callback_data='admin_panel')],
        [InlineKeyboardButton("💾 Создать бэкап", callback_data='force_backup')],
        [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')]
    ]
    
    await query.edit_message_text(
        admin_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def backup_task(context: ContextTypes.DEFAULT_TYPE):
    """Периодическое резервное копирование"""
    if shutdown_event.is_set():
        return
    
    await backup_data()

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
    
    # Отменяем все активные запросы
    for task in active_requests.values():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Создаем финальный бэкап
    await backup_data()
    logger.info("Финальный бэкап создан")
    
    # Закрываем приложение
    application = ApplicationBuilder().build()  # Создаем временный экземпляр
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
        # Загрузка бэкапов
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
        
        # Обработчики сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(button_handler))
        
        # Запуск бота
        logger.info("Бот запускается...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        
        # Ожидаем завершения
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
