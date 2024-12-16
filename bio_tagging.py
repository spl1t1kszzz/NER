def create_bio_labels(text, entities):
    """
    Создаёт BIO-разметку для исходного текста без токенизации.

    Args:
        text (str): Исходный текст.
        entities (list): Список сущностей [(start, end, entity_type)].

    Returns:
        words (list): Список слов исходного текста.
        labels (list): BIO-теги, соответствующие словам.
    """
    words = text.split()  # Разделение текста на слова по пробелам
    labels = ["O"] * len(words)  # Инициализация всех меток как "O"

    # Сопоставляем сущности с словами
    for start, end, entity_type in entities:
        entity_started = False
        for i, word in enumerate(words):
            word_start = text.find(word, start if not entity_started else end)  # Найти начало слова
            word_end = word_start + len(word)

            # Если слово частично или полностью пересекается с сущностью
            if (word_start >= start and word_start < end) or (word_end > start and word_end <= end):
                if not entity_started:
                    labels[i] = f"B-{entity_type}"  # Первая часть сущности
                    entity_started = True
                else:
                    labels[i] = f"I-{entity_type}"  # Остальная часть сущности

    return words, labels


# Пример текста и сущностей
text = "1 January 2010 at 07:59 Заметки об NLP (часть 2) Artificial Intelligence Natural Language Processing."
entities = [
    (35, 38, "Science"),  # Иван Петров
    (49, 72, "Science"), # Москву
    (73, 100, "Science")  # Москву
]

words, labels = create_bio_labels(text, entities)

print("Слова:", words)
print("BIO-разметка:", labels)
