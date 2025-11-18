import json
import csv
import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Импортируем из корневой папки
from lаbs.lab05.json_csv import json_to_csv, csv_to_json


def test_json_to_csv_roundtrip(tmp_path: Path):
    src = tmp_path / "people.json"
    dst = tmp_path / "people.csv"
    data = [
        {"name": "Alice", "age": 22},
        {"name": "Bob", "age": 25},
    ]
    src.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    json_to_csv(str(src), str(dst))

    with dst.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(data)
    assert {"name", "age"} <= set(rows[0].keys())


def test_csv_to_json_roundtrip(tmp_path: Path):
    src = tmp_path / "people.csv"
    dst = tmp_path / "people_out.json"
    rows = [
        {"name": "Alice", "age": "22"},
        {"name": "Bob", "age": "25"},
    ]
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age"])
        writer.writeheader()
        writer.writerows(rows)

    csv_to_json(str(src), str(dst))

    data_out = json.loads(dst.read_text(encoding="utf-8"))

    assert len(data_out) == len(rows)
    assert all("name" in rec and "age" in rec for rec in data_out)


@pytest.mark.parametrize(
    "func, input_file, error",
    [
        ("json_to_csv", "invalid.json", ValueError),
        ("csv_to_json", "invalid.csv", ValueError),
    ],
)
def test_invalid_content_raises(func, input_file, error, tmp_path: Path):
    # Создаем файл с некорректным содержимым
    fpath = tmp_path / input_file
    fpath.write_text("this is not valid json or csv", encoding="utf-8")

    # Подготавливаем путь назначения
    dst = tmp_path / "out.file"

    # Выбираем функцию для теста
    f = json_to_csv if func == "json_to_csv" else csv_to_json

    with pytest.raises(error):
        f(str(fpath), str(dst))


@pytest.mark.parametrize("func", [json_to_csv, csv_to_json])
def test_file_not_found_raises(func, tmp_path: Path):
    non_existent_path = tmp_path / "no_such_file.non"
    dst = tmp_path / "out.file"
    with pytest.raises(FileNotFoundError):
        func(str(non_existent_path), str(dst))
