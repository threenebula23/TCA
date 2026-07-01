# Участие в разработке TCA

## Окружение

```bash
./install.sh
source .venv/bin/activate   # или путь к venv из install.sh
python -m pytest tests/ -q
```

## Pull request

1. Один логический набор изменений; без массовых несвязанных рефакторингов.
2. **Документация**: при изменении публичного поведения тула, режима, prefs или путей данных — обновить соответствующую страницу в **`wiki/`** (см. [wiki/developer/ADDING_TOOLS.md](wiki/developer/ADDING_TOOLS.md)) в том же PR.
3. Добавьте или обновите тесты под новое поведение.
4. Убедитесь, что внутренние ссылки на `wiki/` в README и `docs/` не битые.

## Добавление тула

См. **[wiki/developer/ADDING_TOOLS.md](wiki/developer/ADDING_TOOLS.md)**.

## Стиль кода

Соответствуйте существующему стилю файла; не удаляйте несвязанные комментарии и код.
