# main.py
# Uruchom to z poziomu projektu, gdzie masz datasetHelpers.py obok main.py
#
# WYMAGANE:
# pip install pyyaml opencv-python numpy beautifulsoup4 lxml
#
# CO ROBI:
# 1) bierze XML-e i zdjęcia z folderu data/
# 2) sprawdza podstawowe błędy
# 3) tworzy gotowy dataset w formacie YOLO w folderze dataset_yolo/
#    dataset_yolo/images/{train,val,test}/...
#    dataset_yolo/labels/{train,val,test}/...txt
#    dataset_yolo/data.yaml

#TO JEST KOD Z CHATA KTÓRY BIERZE OBRAZKI I PRZYGOTOWUJE JE DO WALNIĘCIA ICH DO YOLO I TRENINGU MODELU
#Ale nie jest to gotowe bo klasy tych danych trzeba jeszcze wyczyścić
import os
import shutil
from collections import Counter

import yaml

from datasetHelpers import get_xml_files, create_image_dict


# ----------------------------
# USTAWIENIA
# ----------------------------
SOURCE_DATA_DIR = "data"          # tu masz zdjęcia + XML
OUT_DIR = "dataset_yolo"          # tu powstanie dataset dla YOLO
TEST_PCT = 10
VAL_PCT = 10
SELECTED_FOLDERS = None           # np. ["folder1", "folder2"] albo None
COPY_MODE = "copy"                # "copy" albo "symlink" (symlink działa najlepiej na Linux/Mac; na Windows bywa różnie)


# ----------------------------
# Pomocnicze
# ----------------------------
def ensure_dirs(base_dir: str):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)


def split_name(set_type: str) -> str:
    # w Twoich funkcjach bywa "validate", YOLO standardowo używa "val"
    return "val" if set_type == "validate" else set_type


def yolo_line_from_bbox(bb, img_w: int, img_h: int, class_id: int) -> str:
    xmin, ymin, xmax, ymax = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]

    # minimalne zabezpieczenie (nie psuje, jeśli dane są ok)
    xmin = max(0, min(xmin, img_w))
    xmax = max(0, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h))
    ymax = max(0, min(ymax, img_h))

    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h

    # format: class x_center y_center width height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def copy_or_link(src: str, dst: str):
    if COPY_MODE == "symlink":
        # usuń, jeśli istnieje
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def validate_record(rec: dict):
    issues = []
    img_path = rec["filename"]
    w = rec["size"]["width"]
    h = rec["size"]["height"]

    if not os.path.exists(img_path):
        issues.append(f"Brak obrazu: {img_path}")

    for i, obj in enumerate(rec["objects"]):
        bb = obj["bndbox"]
        xmin, ymin, xmax, ymax = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]

        if xmin >= xmax or ymin >= ymax:
            issues.append(f"Zła geometria bbox (obiekt {i}): {bb}")

        if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
            issues.append(f"Bbox poza obrazem (obiekt {i}): {bb} przy (w,h)=({w},{h})")

    return issues


# ----------------------------
# MAIN
# ----------------------------
def main():
    # 1) zbierz listę XML-i
    file_list = get_xml_files(
        root_dir=SOURCE_DATA_DIR,
        selected_folders=SELECTED_FOLDERS,
        testing_percentage=TEST_PCT,
        validation_percentage=VAL_PCT,
        load_from_file=False
    )

    # 2) wczytaj wszystko do dict
    images = create_image_dict(file_list, SOURCE_DATA_DIR)

    # 3) podsumowanie klas + walidacja
    class_counter = Counter()
    all_issues = []

    for xml_name, rec in images.items():
        for obj in rec["objects"]:
            class_counter[obj["name"]] += 1
        issues = validate_record(rec)
        if issues:
            all_issues.append((xml_name, issues))

    print("Liczba obrazów(XML) wczytanych:", len(images))
    print("Liczba klas:", len(class_counter))
    print("Najczęstsze klasy:", class_counter.most_common(20))
    print("Pliki z problemami:", len(all_issues))

    # jeśli są problemy, wypisz kilka i przerwij (lepiej poprawić dane)
    if all_issues:
        print("\nPrzykładowe problemy (pierwsze 10):")
        for xml_name, issues in all_issues[:10]:
            print(" -", xml_name)
            for msg in issues:
                print("    ", msg)
        print("\nPopraw dane albo zdecyduj, że ignorujesz te problemy i usuń ten return.")
        return

    # 4) zrób mapowanie klas -> id
    classes = sorted(class_counter.keys())
    class_to_id = {c: i for i, c in enumerate(classes)}

    # 5) przygotuj foldery wyjściowe
    ensure_dirs(OUT_DIR)

    # 6) buduj YOLO dataset
    for xml_name, rec in images.items():
        split = split_name(rec["type"])  # train/val/test

        src_img = rec["filename"]
        img_base = os.path.basename(src_img)
        img_stem, _ = os.path.splitext(img_base)

        dst_img = os.path.join(OUT_DIR, "images", split, img_base)
        dst_lbl = os.path.join(OUT_DIR, "labels", split, img_stem + ".txt")

        # skopiuj/zalinkuj obraz
        copy_or_link(src_img, dst_img)

        # zapisz label
        w = rec["size"]["width"]
        h = rec["size"]["height"]
        lines = []
        for obj in rec["objects"]:
            cid = class_to_id[obj["name"]]
            lines.append(yolo_line_from_bbox(obj["bndbox"], w, h, cid))

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # 7) zapisz data.yaml dla YOLO
    data_yaml = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }

    with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print("\nGotowe.")
    print("Dataset YOLO:", os.path.abspath(OUT_DIR))
    print("Plik konfiguracyjny:", os.path.join(os.path.abspath(OUT_DIR), "data.yaml"))
    print("Mapowanie klas:", class_to_id)


if __name__ == "__main__":
    main()