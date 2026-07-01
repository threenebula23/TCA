# Module: Agent.tools.ocr_tool

## Purpose

Трёхуровневое извлечение текста: файлы (мягко), изображения/скриншоты (средне), фото (жёсткий OCR).

---

## Responsibilities

- Трёхуровневое извлечение текста: файлы (мягко), изображения/скриншоты (средне), фото (жёсткий OCR).

---

## Public API

|name|description|
|---|---|
|ocr_read_file_soft|Уровень 1 (мягкий): текстовые файлы и PDF с текстовым слоем (без тяжёлого OCR по растру).
Не подходит для .png/.jpg — для них `ocr_read_image_medium` или `ocr_read_photo_strong`.|
|ocr_read_image_medium|Уровень 2 (средний): скриншоты, UI, чёткие диаграммы, отсканированные документы с хорошим контрастом.
Сначала вызывай этот инструмент для изображений; при слабом тексте — `ocr_read_photo_strong` или vision.|
|ocr_read_photo_strong|Уровень 3 (жёсткий): фото с камеры, шум, блики, мелкий шрифт — сильная предобработка + PSM auto.
Если и это не даёт точный текст — используй multimodal/vision по вложенному изображению.|

---

## Dependencies

- `Agent.path_utils`
- `PIL`
- `__future__`
- `fitz`
- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `pytesseract`
- `re`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `Agent/tools/compact_tools.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/ocr_tool.py`

---

## Entry Points


---

## API / route hints

