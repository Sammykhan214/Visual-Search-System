import os
import pandas as pd
import json
from pathlib import Path

def build_metadata(raw_dir: str, output_csv: str, output_json: str) -> pd.DataFrame:
    """
    Scans the GPR1200 image folder, exrtacts category IDs from filenames,
    and creates metadata CSV and category_mapping.json.
    Assumes filename format: {category_id}_{image_id}.jpg
    """ 
    image_paths = list(Path(raw_dir).glob('*.jpg'))
    records = []
    category_set = set()

    for img_path in image_paths:
        stem = img_path.stem
        parts = stem.split('_')
        if len(parts) >= 2:
            cat_id = parts[0]
            cat_set.add(cat_id)
            records.append({
                'image_name': img_path.name,
                'image_path': str(img_path),
                'category_id': cat_id
            })
    
    df = pd.DataFrame(records)
    #create mapping (ID to readable name - here we just use ID as name for simplicity)
    cat_mapping = {cid: f"Category_{cid}" for cid in sorted(category_set)}

    #Save outputs
    df.to_csv(output_csv, index=False)
    with open(output_json, 'w') as f:
        json.dump(cat_mapping, f, indent=2)
    
    return df, cat_mapping