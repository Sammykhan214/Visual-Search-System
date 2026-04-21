import faiss
import numpy as np
import pandas as pd
import os

def build_faiss_index(metadata_df, embeddings_dir, index_save_path, dimension, metric='cosine'):
    """
    Load all .npy embeddings, build FAISS index, and save it.
    Also saves an ID mapping for retrieval.
    """

    image_paths = metadata_df['image_path'].tolist()
    vectors = []
    valid_indices = []

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        npy_path = os.path.join(embeddings_dir, f"{os.path.splitext(img_name)[0]}.npy")
        if os.path.exists(npy_path):
            vec = np.load(npy_path).astype('float32')
            vectors.append(vec)
            valid_indices.append(idx)
        else:
            print(f"Warning: Embedding not found for {img_path}, skipping.")
    
    vectors = np.vstack(vectors).astype('float32')

    if metric == 'cosine':
        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    index.add(vectors)
    faiss.write_index(index, index_save_path)

    # Save mapping from FAISS index to metadata index
    mapping_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    mapping_df.to_csv(index_save_path.replace('.bin', '_mapping.csv'), index=False)

    print(f"Index built with {index.ntotal} vectors and saved to {index_save_path}")