import os
import csv
from openpyxl import Workbook


def csv_to_xlsx(csv_path: str, xlsx_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл не найден: {csv_path}")
    if os.path.getsize(csv_path) == 0:
        raise ValueError("Файл полностью пуст")

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            ws.append(row)

    for column_cells in ws.columns:
        max_length = 0
        column_l = column_cells[0].column_letter
        for cell in column_cells:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[column_l].width = max(max_length + 2, 8)
    wb.save(xlsx_path)


csv_to_xlsx(
    r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\samples\citties.csv",
    r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\out\people.xlsx",
)
