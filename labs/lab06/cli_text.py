import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CLI‑утилиты лабораторной №6")
    subparsers = parser.add_subparsers(dest="command")

    cat_parser = subparsers.add_parser("cat", help="Вывести содержимое файла")
    cat_parser.add_argument("--input", required=True)
    cat_parser.add_argument("-n", action="store_true", help="Нумеровать строки")

    stats_parser = subparsers.add_parser("stats", help="Частоты слов")
    stats_parser.add_argument("--input", required=True)
    stats_parser.add_argument("--top", type=int, default=5)

    args = parser.parse_args()

    if args.command == "cat":
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if args.n:
                        print(f"{i:4} {line.rstrip()}")
                    else:
                        print(line.rstrip())
        except FileNotFoundError:
            print(f"Ошибка: файл {args.input} не найден")
            sys.exit(1)

    elif args.command == "stats":
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from mylabs.text import normalize, tokenize, count_freq, top_n

            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = tokenize(normalize(text))
            freq = count_freq(tokens)
            top_words = top_n(freq, args.top)

            print(f"Всего слов: {len(tokens)}")
            print(f"Уникальных слов: {len(freq)}")
            print(f"Топ-{args.top}:")
            for word, count in top_words:
                print(f"{word}: {count}")

        except FileNotFoundError:
            print(f"Ошибка: файл {args.input} не найден")
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
cat:
python lаbs\lab06\cli_text.py cat --input data/samples/people.csv -n

stats:
python lаbs\lab06\cli_text.py stats --input data/samples/text.txt --top 5
"""
