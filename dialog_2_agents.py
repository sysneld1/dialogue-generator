# app.py - Веб-приложение на Flask для генерации диалогов с использованием SocketIO и LLM.
# Это приложение позволяет пользователям инициировать и контролировать диалоги между двумя персонажами,
# основанные на темах и описаниях ролей, с интеграцией модели llama_cpp для генерации ответов.

from flask import Flask, render_template, request  # Flask для веб-фреймворка, render_template для шаблонов, request для SID
from flask_socketio import SocketIO, emit  # SocketIO для realtime коммуникации, emit для отправки сообщений
import time  # Для временных задержек и таймстампов
import llama_cpp  # Библиотека для работы с LLM моделями (GGUF формат)
import os  # Для работы с файловой системой
import ssl  # Для настройки HTTPS
import GPUtil  # Для мониторинга GPU
import threading  # Для многопоточности (лок, потоки)
import logging  # Для логирования событий и ошибок

# Настройка логирования для отладки и мониторинга
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация Flask app и SocketIO с включенными CORS для всех доменов (для демонстрации; в продакшене ограничить)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Глобальная переменная для загруженной модели LLM
model = None

# Глобальный словарь для флагов остановки диалогов, ключ - SID пользователя, чтобы каждый мог останавливать только свой диалог
stop_flags = {}

# Глобальный словарь для хранения историй диалогов по SID (только текст, метаданные добавляются при сохранении)
dialogs = {}

# Блокировка для предотвращения одновременного запуска нескольких диалогов (глобальная очередь)
dialog_lock = threading.Lock()

def check_gpu_status():
    """
    Проверяет статус первого GPU и возвращает строку: 'free', 'busy', 'no_gpu' или 'error'.
    Использует загрузку GPU > 80% как критерий занятости.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 'no_gpu'
        # Проверяем первую GPU
        gpu = gpus[0]
        # Если загрузка процессоров > 80%, считаем GPU занятым
        if gpu.load * 100 > 80:
            return 'busy'
        else:
            return 'free'
    except Exception as e:
        print(f"Ошибка при проверке GPU: {e}")
        return 'error'

def load_model():
    """
    Загружает модель LLM с использованием llama_cpp, если она еще не загружена.
    Возвращает объект модели или None в случае ошибки.
    Данный пример тестировался с локальной моделью Grok-3-reasoning-gemma3.
    При тестировании использовалась бытовая видеокарта RTX 3090ti.
    Время на генерацию ответа составляло 8-17 сек.
    Это дает представление о возможности использовать LLM в закрытом контуре
    с использованием чувствительных данных.
    """
    global model
    if model is None:
        try:
            model = llama_cpp.Llama(
                model_path=r"G:\LLM_models2\Grok-3-reasoning-gemma3-12B-distilled-HF.Q8_0.gguf",  # Путь к модели GGUF
                n_ctx=8192,  # Максимальный контекст
                chat_format="gemma",  # Формат чата для Gemma
                n_threads=4,  # Количество CPU потоков
                n_gpu_layers=47,  # Количество слоев на GPU (снижено для стабильности)
                temperature=0.7,  # Температура генерации
                max_tokens=8192,  # Максимум токенов в ответе
                verbose=False  # Отключить подробный вывод
            )

        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            model = None
    return model

# Загрузка модели при запуске приложения
load_model()

def save_dialog_to_file(sid, dialog_history, topic, role1_name, role1_description, role2_name, role2_description):
    """
    Сохраняет историю диалога с метаданными в файл в папке logs.
    Файл имеет имя: dialog_{sid}_{timestamp}.txt
    Включает тему, роли, описание ролей, количество шагов и время.
    """
    os.makedirs('logs', exist_ok=True)
    timestamp = int(time.time())
    filename = f"logs/dialog_{sid}_{timestamp}.txt"
    try:
        # Добавляем метаданные в начало файла
        metadata = f"""=== Метаданные диалога ===
SID: {sid}
Тема диалога: {topic}
Роль 1: {role1_name}
Описание роли 1: {role1_description}
Роль 2: {role2_name}
Описание роли 2: {role2_description}
Количество шагов: {len([line for line in dialog_history.split('\n') if line.strip() and ':' in line]) // 2} (примерно)
Время создания: {time.strftime('%Y-%m-%d %H:%M:%S')}
=== История диалога ===

{dialog_history}
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(metadata)
        logging.info(f"Диалог для SID {sid} сохранён в файл {filename} с метаданными (тема: {topic}, роли: {role1_name}, {role2_name})")
    except Exception as e:
        logging.error(f"Ошибка сохранения диалога для SID {sid}: {e}")

def update_gpu_status(app_instance, sock):
    """
    Фоновая функция для периодической проверки статуса GPU и отправки обновлений клиентам через SocketIO.
    Запускается в отдельном потоке, обновляет каждые 5 секунд.
    """
    while True:
        status = check_gpu_status()
        message = {
            'busy': "GPU занят",
            'free': "GPU свободен"
        }.get(status, "Отсутствует GPU или ошибка")
        with app_instance.app_context():
            sock.emit('gpu_status', {'status': status, 'message': message})
        time.sleep(5)  # Проверка каждые 5 секунд

# Запуск фонового потока для обновления статуса GPU
status_thread = threading.Thread(target=update_gpu_status, args=(app, socketio), daemon=True)
status_thread.start()

def generate_response(topic, personality_description, conversation_history, max_tokens=300):
    """
    Генерирует ответ от имени персонажа с использованием LLM.
    Строит промпт на основе темы, роли и истории.
    В случае ошибки или пустого ответа использует резервный промпт.
    Возвращает строку ответа или сообщение об ошибке.
    """
    global model
    if model is None:
        return "Ошибка: Модель не загружена."
    try:
        prompt = f"<start_of_turn>user\nТема диалога: {topic}\n\nВаша роль: {personality_description}\n\nИстория диалога (предыдущие реплики):\n{conversation_history}\n\nТеперь ваша очередь. Сгенерируйте одну реплику от своего имени, кратко и соответствуя роли. Не превышайте одно предложение.\n<end_of_turn>\n<start_of_turn>model\n"

        output = model(
            prompt,
            max_tokens=max_tokens,
            stop=["<end_of_turn>", "\n\n", "\n", "Ученый:", "Студент:"],
            echo=False
        )
        response = output.get("choices", [{}])[0].get("text", "").strip()

        if not response or len(response) < 5:
            prompt_retry = f"<start_of_turn>user\n{personality_description}. Ответь на тему '{topic}' одной репликой.\n<end_of_turn>\n<start_of_turn>model\n"
            output = model(prompt_retry, max_tokens=100, stop=["<end_of_turn>", "\n"], echo=False)
            response = output.get("choices", [{}])[0].get("text", "").strip()

        return response.strip()
    except Exception as e:
        logging.error(f"Ошибка генерации ответа: {e}")
        return "Ошибка: Не удалось сгенерировать ответ."

# Главный маршрут приложения, возвращает HTML шаблон
@app.route('/')
def index():
    return render_template('index.html')  # Шаблон должен существовать

# Обработчик подключения клиента через SocketIO
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    logging.info(f"Пользователь подключился: SID {sid}")
    # Отправляет текущий статус GPU при подключении
    status = check_gpu_status()
    message = {
        'busy': "GPU занят",
        'free': "GPU свободен"
    }.get(status, "Отсутствует GPU или ошибка")
    emit('gpu_status', {'status': status, 'message': message})

# Обработчик остановки диалога
@socketio.on('stop_dialog')
def handle_stop_dialog():
    sid = request.sid
    logging.info(f"Пользователь остановил диалог: SID {sid}")
    stop_flags[sid] = True
    # Для сохранения диалога вызывается в handle_start_dialog при завершении
    emit('dialog_stopped', to=sid)

# Основной обработчик запуска диалога
@socketio.on('start_dialog')
def handle_start_dialog(data):
    sid = request.sid
    logging.info(f"Пользователь начал диалог: SID {sid}, тема: {data.get('topic', 'Не указана')}, роли: {data.get('role1_name', 'Не указана')}/{data.get('role2_name', 'Не указана')}, шагов: {data.get('num_steps', 'Не указана')}")
    stop_flags[sid] = False  # Сбрасываем флаг остановки для этого SID

    # Инициализируем историю диалога для SID
    dialogs[sid] = ""

    # Пытаемся получить блокировку для запуска диалога
    if not dialog_lock.acquire(blocking=False):
        logging.warning(f"Диалог заблокирован другим пользователем: SID {sid}")
        emit('dialog_error', {'message': 'Ошибка: Диалог уже запущен другим пользователем. Пожалуйста, подождите.'}, to=sid)
        return

    # Извлекаем данные из запроса с дефолтами
    topic = data.get('topic', 'Спор о форме Земли')
    role1_name = data.get('role1_name', 'Иван')
    role1_description = data.get('role1_description', '''
    Ты ученый физик. Объясни решение задачи понятным и доступным для школьника 10 класса способом.
    Дополняй предыдущие объяснения, приводи примеры и аналогии. Не здоровайся. 
    Не называй имена и роли участников диалога.
    ''')
    role2_name = data.get('role2_name', 'Петр')
    role2_description = data.get('role2_description', '''
    Ты веселый учитель физики. Объясни решение задачи простым для ребенка способом.
    Дополняй предыдущие объяснения, приводи примеры и аналогии. 
    Не здоровайся. Не называй имена и роли участников диалога.
    ''')
    num_steps = int(data.get('num_steps', 7))

    try:  # try-finally для гарантии освобождения блокировки
        if model is None:
            logging.error(f"Модель не загружена для SID {sid}")
            emit('new_line', {'step': 0, 'line': 'Ошибка: Модель не загружена.'}, to=sid)
            return

        conversation_history = ""

        # Цикл генерации шагов диалога
        for step in range(1, num_steps + 1):
            if stop_flags.get(sid, False):
                break

            # Определяем говорящего на основе шага (нечетный - роль1, четный - роль2)
            if step % 2 == 1:
                speaker = role1_name
                personality = role1_description
            else:
                speaker = role2_name
                personality = role2_description

            # Отправляем сигнал ожидания клиенту
            emit('waiting', {
                'step': step,
                'speaker': speaker
            }, to=sid)

            # Небольшая пауза для визуализации
            time.sleep(0.3)

            if stop_flags.get(sid, False):
                break

            # Фиксированная первая реплика без LLM
            if step == 1 and speaker == role1_name:
                response = "Привет, давай поспорим?"
            else:
                response = generate_response(topic, personality, conversation_history)

            if stop_flags.get(sid, False):
                break

            # Добавляем реплику в историю
            new_line = f"{speaker}: {response}"
            conversation_history += f"{new_line}\n"
            dialogs[sid] += f"{new_line}\n"

            # Отправляем новую реплику клиенту
            emit('new_line', {
                'step': step,
                'line': new_line
            }, to=sid)

        # После цикла: если не остановлен, завершаем и сохраняем
        if not stop_flags.get(sid, False):
            logging.info(f"Диалог завершён: SID {sid}, тема: {topic}, шагов: {step}")
            save_dialog_to_file(sid, dialogs[sid], topic, role1_name, role1_description, role2_name, role2_description)
            emit('dialog_completed', {'steps': step}, to=sid)
            del dialogs[sid]  # Очистка памяти
        else:
            logging.info(f"Диалог остановлен: SID {sid}, тема: {topic}, шагов: {step}")
            save_dialog_to_file(sid, dialogs[sid], topic, role1_name, role1_description, role2_name, role2_description)
            emit('dialog_stopped', {'steps': step}, to=sid)
            del dialogs[sid]  # Очистка памяти

    finally:
        dialog_lock.release()  # Всегда освобождаем блокировку

# Запуск приложения в блоке __main__
if __name__ == '__main__':
    # Попытка запуска с SSL сертификатами (HTTPS на порту 443)
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain('cert.pem', 'key.pem')

        socketio.run(
            app,
            host='0.0.0.0',
            port=443,  # Стандартный HTTPS порт
            debug=True,
            ssl_context='adhoc',  # Автоматическое создание SSL
            allow_unsafe_werkzeug=True
        )
    else:
        # Фallback на HTTP без SSL
        print("SSL сертификаты не найдены. Запускаю в HTTP режиме.")
        print("Для HTTPS создайте сертификаты командой:")
        print("openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
        # Для продакшена можно использовать готовый SSL контекст, например: ('/path/to/fullchain.pem', '/path/to/privkey.pem')

        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            allow_unsafe_werkzeug=True
        )
