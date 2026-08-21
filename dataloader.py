from torch.utils.data import DataLoader
from dataset import DegradedFacesDataset

def get_dataloader(root_dir, batch_size=8, num_workers=2, shuffle=True):
    """
    Returns a configured PyTorch DataLoader for the DegradedFacesDataset.
    Optimized for local edge hardware (<8GB VRAM). 
    A batch size of 8-16 is typically recommended for 8GB VRAM when training 
    FPN and GAN models, but can be adjusted based on the specific architecture's memory footprint.
    
    :param root_dir: Directory containing high-res faces.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses for data loading.
    :param shuffle: Whether to shuffle the data at every epoch.
    :return: DataLoader instance
    """
    dataset = DegradedFacesDataset(root_dir=root_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Speeds up transfer to GPU
        drop_last=True
    )
    
    return dataloader
