import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from .model_factory import get_extractor

# Set number of threads to match CPU cores (4 on typical i3/i5)
torch.set_num_threads(4)
# Disable gradient computation globally (saves memory)
torch.set_grad_enabled(False)

def generate_embeddings(metadata_df, output_dir, model_name='mobilenetv3', batch_size=8, device='cpu'):
    """
    CPU-optimized: small batch size, no GPU.
    """
    extractor = get_extractor(model_name)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = metadata_df['image_path'].tolist()


    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []

        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                tensor = extractor.preprocess(img)
                batch_images.append(tensor)
                valid_indices.append(i + idx)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        batch_tensors = torch.stack(batch_images).to(device)
        features = extractor.extract_batch(batch_tensors).cpu().numpy()

        # Save embeddings
        for rel_idx, global_idx in enumerate(valid_indices):
            img_name = os.path.basename(image_paths[global_idx])
            save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.npy")
            np.save(save_path, features[rel_idx])
        
        #Explicityly free memory
        del batch_tensors, features, batch_images
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"Embeddings saved to {output_dir}")