# Мини-RAG-сервис

HTTP-сервис на FastAPI для создания RAG (Retrieval-Augmented Generation) системы. Сервис умеет загружать документы (PDF, Markdown, TXT), создавать их векторные представления и отвечать на вопросы на основе содержимого документов.

## Возможности

- 📄 **Загрузка документов**: Поддержка PDF, Markdown и TXT файлов
- 🔍 **Векторный поиск**: Использование OpenAI embeddings и Qdrant для хранения
- 🤖 **Умные ответы**: Генерация ответов с помощью GPT-4o/4.1-mini
- 📚 **Источники**: Возврат списка использованных фрагментов-источников
- 🏷️ **Теги**: Возможность добавления тегов к документам
- 🗑️ **Удаление**: Удаление документов и их фрагментов

## API Endpoints

| Метод    | URL                   | Описание                                                |
| -------- | --------------------- | ------------------------------------------------------- |
| `POST`   | `/documents/ingest`   | Загрузка документа (multipart: `file`, `tags`(opt))     |
| `POST`   | `/chat/ask`           | Задать вопрос (JSON: `{"question": str, "top_k": int}`) |
| `GET`    | `/health`             | Проверка состояния сервиса                              |
| `DELETE` | `/documents/{doc_id}` | Удаление документа                                      |

## Быстрый старт

### Локальное развертывание

1. **Клонируйте репозиторий:**

```bash
git clone <repository-url>
cd test
```

2. **Установите зависимости:**

```bash
uv sync
```

3. **Создайте файл .env:**

```bash
cp .env.example .env
# Отредактируйте .env с вашими API ключами
```

4. **Запустите Qdrant:**

```bash
docker-compose up qdrant -d
```

5. **Запустите приложение:**

```bash
uv run main.py
```

### Docker развертывание

1. **Создайте .env файл с вашими API ключами**

2. **Запустите весь стек:**

```bash
docker-compose up --build
```

Сервис будет доступен по адресу: http://localhost:8000

## Примеры использования

### 1. Загрузка документа

```bash
curl -X POST "http://localhost:8000/documents/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "tags=research,important"
```

**Ответ:**

```json
{
  "document_id": "abc123def456",
  "status": "ingested"
}
```

### 2. Задать вопрос

```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое машинное обучение?",
    "top_k": 3
  }'
```

**Ответ:**

```json
{
  "result": "Машинное обучение - это подраздел искусственного интеллекта...",
  "source_documents": [
    {
      "id": "chunk_123",
      "document_id": "abc123def456",
      "text": "Машинное обучение представляет собой..."
    }
  ]
}
```

### 3. Проверка здоровья сервиса

```bash
curl http://localhost:8000/health
```

**Ответ:**

```json
{
  "status": "alive"
}
```

### 4. Удаление документа

```bash
curl -X DELETE "http://localhost:8000/documents/abc123def456"
```

**Ответ:**

```json
{
  "document_id": "abc123def456",
  "status": "deleted"
}
```

## Тестирование

Запустите тесты с покрытием:

```bash
pytest --cov=src --cov-report=term-missing
```

## Структура проекта

```
src/
├── chat/           # API для вопросов и ответов
├── documents/      # API для загрузки документов
├── llm/           # Интеграции с LLM и векторным хранилищем
└── settings.py    # Конфигурация приложения
```

## Технологии

- **FastAPI** - веб-фреймворк
- **OpenAI** - LLM и embeddings
- **Qdrant** - векторное хранилище
- **LlamaParse** - парсинг документов
- **Pydantic** - валидация данных
- **Docker** - контейнеризация

## AI-инструменты

При разработке этого проекта использовались следующие AI-инструменты:

- **Cursor** - основной (со второго дня) IDE с AI-ассистентом для написания кода, рефакторинга и отладки
- **ChatGPT** - помощь с архитектурными решениями и объяснение концепций
- **GitHub Copilot** - автодополнение кода и предложения
