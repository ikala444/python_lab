import sys
from text import normalize, tokenize, count_freq, top_n

text = sys.stdin.read()
norm = normalize(text)
token = tokenize(norm)
cf = count_freq(token)
tn = top_n(cf, 5)
print(f"Всего слов: {len(token)}")
print(f"Уникальных слов: {len(set(token))}")
print("Топ 5:")
for word, value in tn:
    print(f"{word}: {value}")
