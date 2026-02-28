#Kod biorący i wypisujący klasy które są w danych

from pathlib import Path
from collections import Counter
from datasetHelpers import get_xml_files, create_image_dict

BASE = Path(__file__).resolve().parents[2]   # jeśli uruchamiasz z dataset_yolo/labels/
DATA_DIR = BASE / "data"

file_list = get_xml_files(
    root_dir=str(DATA_DIR),
    selected_folders=None,
    testing_percentage=10,
    validation_percentage=10,
    load_from_file=False
)

images = create_image_dict(file_list, str(DATA_DIR))

counter = Counter()
for rec in images.values():
    for obj in rec["objects"]:
        counter[obj["name"]] += 1

print("Klasa -> liczba wystąpień (alfabetycznie)\n")
for cls in sorted(counter.keys()):
    print(f"{cls}: {counter[cls]}")

#Odciski palców?