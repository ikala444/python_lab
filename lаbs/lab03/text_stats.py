import sys
from libs import text

def text_info(table: bool = True):
    text = sys.stdin.readline().strip()
    tokens = tokenize(normalize(text))
    top_words = top_n(count_freq(tokens), 5)

    print(f"Всего слов: {len(tokens)}")
    print(f"Уникальных слов: {len(set(tokens))}")