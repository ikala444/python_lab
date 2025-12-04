import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from lab05.json_csv import json_to_csv, csv_to_json
from lab05.csv_xlsx import csv_to_xlsx


def main():
    parser = argparse.ArgumentParser(description="Конвертеры данных")
    sub = parser.add_subparsers(dest="cmd")
    p1 = sub.add_parser("json2csv")
    p1.add_argument("--in", dest="input", required=True)
    p1.add_argument("--out", dest="output", required=True)
    p2 = sub.add_parser("csv2json")
    p2.add_argument("--in", dest="input", required=True)
    p2.add_argument("--out", dest="output", required=True)
    p3 = sub.add_parser("csv2xlsx")
    p3.add_argument("--in", dest="input", required=True)
    p3.add_argument("--out", dest="output", required=True)
    args = parser.parse_args()
    if args.cmd == "json2csv":
        json_to_csv(args.input, args.output)
        print(f"Успешно: {args.input} -> {args.output}")
    elif args.cmd == "csv2json":
        csv_to_json(args.input, args.output)
        print(f"Успешно: {args.input} -> {args.output}")
    elif args.cmd == "csv2xlsx":
        csv_to_xlsx(args.input, args.output)
        print(f"Успешно: {args.input} -> {args.output}")
    else:
        print("Ошибка: неизвестная команда")


if __name__ == "__main__":
    main()


"""
json2csv:
python lаbs\lab06\cli_converter.py json2csv --in data/samples/people.json --out data/out/people.csv

csv2json:
python lаbs\lab06\cli_converter.py csv2json --in data/samples/people.csv --out data/out/people.json

csv2xlsx:
python lаbs\lab06\cli_converter.py csv2xlsx --in data/samples/people.csv --out data/out/people.xlsx
"""
