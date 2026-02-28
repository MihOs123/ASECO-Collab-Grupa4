#To są funkcje od mateuszka, lekko zmieniłem pierwszą funkcję reszta jest bez zmian

import os
import numpy as np
from bs4 import BeautifulSoup

def load_annotation(file_path, infos, use_gtin=False):
    # 1) wczytaj cały XML jako tekst
    with open(file_path, "r", encoding="utf-8") as f:
        xml = f.read()

    # 2) powiedz wprost, że to XML (żeby nie zgadywał)
    bs = BeautifulSoup(xml, "xml")

    # 3) folder, w którym leży XML
    image_dir = os.path.dirname(file_path)

    # 4) nazwa pliku obrazu z XML
    filename_tag = bs.find("filename")
    if filename_tag is None or filename_tag.string is None:
        raise ValueError(f"Brak tagu <filename> w pliku: {file_path}")
    filename = filename_tag.string.strip().replace("\t", "").replace(" ", "")
    image_path = os.path.join(image_dir, filename)

    # 5) rozmiar obrazu z XML
    size_tag = bs.find("size")
    if size_tag is None or size_tag.width is None or size_tag.height is None:
        raise ValueError(f"Brak tagu <size>/<width>/<height> w pliku: {file_path}")

    image = {
        "filename": image_path,
        "size": {
            "width": int(size_tag.width.string),
            "height": int(size_tag.height.string)
        },
        "objects": []
    }

    # 6) doklej dodatkowe info
    image.update(infos)

    # helper: bezpieczne pobranie tekstu z taga
    def _text(tag, default=""):
        return tag.string.strip() if tag and tag.string else default

    # helper: poprawne "0/1" -> False/True
    def _bool01(tag, default=False):
        s = _text(tag, None)
        if s is None:
            return default
        try:
            return bool(int(s))
        except ValueError:
            return default

    # 7) obiekty
    for ob in bs.find_all("object"):
        name = _text(ob.find("name")).replace("\t", "").replace(" ", "")

        if use_gtin:
            parts = name.split("_")
            if len(parts) == 6:
                name = "gtin" + parts[-1]

        bnd = ob.find("bndbox")
        if bnd is None:
            raise ValueError(f"Brak <bndbox> w pliku: {file_path}")

        obj = {
            "name": name,
            "bndbox": {
                "xmin": int(_text(bnd.find("xmin"))),
                "xmax": int(_text(bnd.find("xmax"))),
                "ymin": int(_text(bnd.find("ymin"))),
                "ymax": int(_text(bnd.find("ymax")))
            },
            "pose": _text(ob.find("pose"), None),
            "truncated": _bool01(ob.find("truncated"), False),  # TERAZ liczy się poprawnie
            "difficult": _bool01(ob.find("difficult"), False)   # TERAZ liczy się poprawnie
        }
        image["objects"].append(obj)

    return image
  
def get_xml_files(root_dir, selected_folders,
                  testing_percentage, validation_percentage,
                  load_from_file = True):
  """Gets all xml files recursively inside the root directory
  and returns the file names with the path
  Args:
    root_dir: root directory as string
    selected_folders: list of folder names which should only be
        used for the dataset. If None will use all.
  Returns:
    list of tuples with file_name and path_name relative from
        root_dir
  """
  file_list = []
  
  if load_from_file:
    try:
      with open(FLAGS_OD.xml_list_file, "r") as f:
        for line in f:
          file_list.append(line.split(";"))
      return file_list
    except:
      print("Failed at loading file_list from file, creating new one")
  
  data_split = TrainValTestSplit(testing_percentage, validation_percentage)

  root_f = root_dir.split(os.sep)
  for path, _, files in os.walk(root_dir):
    path_f = path.split(os.sep)
    if not selected_folders is None and not \
        any(s_folder in path_f for s_folder in selected_folders):
      continue
    for file in files:
      if file.split(".")[-1] == "xml":
          rel_parts = [f for f in path_f if f not in root_f]
          rel_path = os.path.join(*rel_parts) if rel_parts else ""  # <- najważniejsze
          file_list.append([file, rel_path, data_split.next_label()])
  
  return file_list
  
class TrainValTestSplit:
  """This class helps in assigning a train, validation or test
  label while making sure the datasets have a similar distribution
  of classes. Helpful for small datasets
  """
  #TODO init with objects list to get actual dist
  def __init__(self, testing_percentage=10, validation_percentage=10):
    """
    Args:
      testing_percentage: int percentage of dataset that should go to test
      validation_percentage: int percentage of dataset that should go to validate
    """
    self.tr = testing_percentage
    self.v = validation_percentage
  #TODO leverage actual dist to adjust label if necessary  
  #TODO check that the distribution in all sets is the same
  #-- distribution over classes, shelf_type
  def next_label(self):
    percentage = np.random.randint(1,100)
    if percentage < self.tr:
      return "test"
    elif percentage < (self.tr + self.v):
      return "validate"
    else:
      return "train"
	 
def create_image_dict(file_list, image_dir):
  """Update the list of images and object annotations from the file system.
  Split them into stable training, testing, and validation sets.
  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.
    images: a dict containing all the information per image
    objects: a dict containing all the information per object
    classes: a dict containing all the information per class
  Returns:
    an updated or new images, objects & classes dict
  """

  images = dict()
  num_files = 0
  for file_name, rel_path, set_type in file_list:
    infos = {"type": set_type,
             "shelf_type": rel_path.split(os.sep)[0]}
    images[file_name] = load_annotation(os.path.join(image_dir,rel_path,file_name), infos)
    num_files += 1
    
    if (num_files) % 20 == 0:
      print("images found sofar: {}".format(num_files))
  print("images found total: {}".format(num_files))    
  
  return images
  
 
## Usage 
#file_list = get_xml_files(image_dir,
#                          selected_folders,
#                          testing_percentage,
#                          validation_percentage,
#                          False)
#
#images = create_image_dict(file_list,
#                           image_dir)