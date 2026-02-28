#Ten kod po prostu sprawdzał plik pkl który z jakiegoś powodu był w danych, ale on był pusty XD

from pathlib import Path
import os
import pickle

PKL_PATH = Path(r"data/images.pkl")  # albo BASE/"data"/"nazwa.pkl"

print("Working dir:", os.getcwd())
print("PKL path:", PKL_PATH)
print("Exists:", PKL_PATH.exists())

if PKL_PATH.exists():
    size = PKL_PATH.stat().st_size
    print("Size (bytes):", size)

    # pokaż pierwsze bajty pliku
    with open(PKL_PATH, "rb") as f:
        head = f.read(32)
    print("First 32 bytes:", head)

    if size == 0:
        print("PLIK JEST PUSTY -> nie da się go wczytać pickle.")
    else:
        try:
            with open(PKL_PATH, "rb") as f:
                obj = pickle.load(f)
            print("Loaded OK, type:", type(obj))
        except Exception as e:
            print("pickle.load ERROR:", type(e).__name__, str(e))