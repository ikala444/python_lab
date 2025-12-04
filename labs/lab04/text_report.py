import sys

sys.path.append(r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\lаbs")
from mylabs.text import normalize, tokenize, top_n, count_freq
from lab04.io_txt_csv import read_text, write_csv


def main():
    input_text = read_text(
        r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\lab04\input.txt"
    )
    tokens = tokenize(normalize(input_text))
    freq = count_freq(tokens)
    top_5 = top_n(freq, 5)
    top_all = top_n(freq, 100)

    print(f"Всего слов: {len(tokens)}")
    print(f"Уникальных слов: {len(freq)}")
    print("Топ-5:")
    for word, count in top_5:
        print(f"{word}: {count}")

    write_csv(top_all, path=r"data\lab04\report.csv", header=("word", "count"))


if __name__ == "__main__":
    main()
