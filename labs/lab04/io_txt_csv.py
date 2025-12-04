import csv
from pathlib import Path
from typing import Iterable, Sequence


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(path)
    return p.read_text(encoding=encoding)


def write_csv(
    rows: Iterable[Sequence], path: str | Path, header: tuple[str, ...] | None = None
) -> None:
    p = Path(path)
    rows = list(rows)
    if rows:
        first_len = len(rows[0])
        if any(len(row) != first_len for row in rows):
            raise ValueError("Все строки должны иметь одинаковую длину")
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


print(
    read_text(r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\lab04\input.txt")
)
write_csv(
    [("word", "count"), ("test", 3)],
    r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\lab04\check.csv",
)
