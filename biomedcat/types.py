from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional
from biomedcat.stages.biolink_yml_processor import biolink_yml_processor


BASE_DIR = Path(__file__).parent.absolute()
BIOLINK_FILE_PATH = BASE_DIR / "data/biolink_classes_flat.json"

if BIOLINK_FILE_PATH.is_file():
    # checks if file exists
    print ("Loading biolink info...")
    with open(str(BIOLINK_FILE_PATH)) as json_data:
        biolink_info_flat