import logging
from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from sklearn.feature_extraction.text import CountVectorizer  # Импортируем CountVectorizer
import numpy as np
import sqlite3

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Определяем состояния для ConversationHandler
SURVEY = range(1)


# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('"Путь к БД"')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Загрузка вопросов и ответов из базы данных
def load_faq():
    conn = sqlite3.connect('"Путь к БД"')
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM questions")
    faq = cursor.fetchall()
    conn.close()
    return faq


# Функция обработки команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Привет! Я обучающий бот. Задайте свой вопрос или напишите /survey для участия в опросе.',
        reply_markup=ForceReply(selective=True),
    )


# Функция обработки команды /survey
async def survey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Пожалуйста, дайте свою обратную связь о нашем чат-боте.')
    return SURVEY  # Переход к следующему состоянию


# Функция сохранения обратной связи
async def save_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    feedback = update.message.text

    # Получаем имя пользователя (если оно доступно)
    username = update.message.from_user.username if update.message.from_user.username else "Не указано"

    # Сохранение обратной связи в базу данных
    conn = sqlite3.connect('"Путь к БД"')
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO feedback (username, feedback) VALUES (?, ?)", (username, feedback))
        conn.commit()
        await update.message.reply_text('Спасибо за вашу обратную связь!')
    except Exception as e:
        logging.error(f"Ошибка при сохранении обратной связи: {e}")
        await update.message.reply_text('Произошла ошибка при сохранении вашей обратной связи.')
    finally:
        conn.close()

    return ConversationHandler.END  # Завершение разговора


# Функция обработки сообщений от пользователей
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    faq = load_faq()

    if not faq:
        await update.message.reply_text('К сожалению, у нас нет базы данных вопросов.')
        return

    # Создание векторизатора и преобразование FAQ
    questions = [q[0] for q in faq]

    # Векторизация пользовательского сообщения
    vectorizer = CountVectorizer().fit(questions)

    user_vector = vectorizer.transform([user_message]).toarray()
    faq_vectors = vectorizer.transform(questions).toarray()

    # Нахождение самого подходящего вопроса
    similarities = np.dot(faq_vectors, user_vector.T)

    # Проверка на случай отсутствия похожих вопросов
    if np.all(similarities == 0):
        await update.message.reply_text("Извините, я не смог найти подходящий ответ.")
        return

    best_match_index = np.argmax(similarities)

    # Ответ на основе найденного вопроса
    await update.message.reply_text(faq[best_match_index][1])


# Главная функция
def main():
    init_db()  # Инициализация базы данных

    application = ApplicationBuilder().token("token tg-bota").build()

    # Определяем ConversationHandler для обработки опроса
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("survey", survey)],
        states={
            SURVEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_feedback)]
        },
        fallbacks=[],
        allow_reentry=True,
    )

    application.add_handler(CommandHandler("start", start))

    # Добавляем ConversationHandler для обработки опроса
    application.add_handler(conv_handler)

    # Обработка текстовых сообщений (не команд)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == '__main__':
    main()
