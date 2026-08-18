"""
Dataset manager for Bacpipe.
Scans the audio directory and manages dataset selection.
"""

import os
from pathlib import Path


def get_datasets_dir():
    """Get the path to the audio datasets directory."""
    current_dir = Path(__file__).parent
    audio_dir = current_dir / "public" / "assets" / "audio"
    print(f"🔍 Dataset manager: Looking for datasets in {audio_dir}")
    return audio_dir


def get_available_datasets():
    """
    Get list of available datasets from the audio directory.
    
    Returns:
        list: List of dicts with 'name' and 'path' keys
    """
    audio_dir = get_datasets_dir()
    datasets = []
    
    print(f"📁 Checking if audio directory exists: {audio_dir.exists()}")
    
    if not audio_dir.exists():
        print(f"⚠️  Audio directory does not exist: {audio_dir}")
        return datasets
    
    try:
        # Get all subdirectories in the audio folder
        items = list(audio_dir.iterdir())
        print(f"📂 Found {len(items)} items in {audio_dir}")
        
        for item in sorted(audio_dir.iterdir()):
            print(f"  - Checking: {item.name} (is_dir: {item.is_dir()})")
            if item.is_dir():
                dataset_dict = {
                    'name': item.name,
                    'path': str(item),
                    'display_name': item.name.replace('_', ' ').title()
                }
                datasets.append(dataset_dict)
                print(f"    ✅ Added dataset: {dataset_dict['display_name']}")
        
        print(f"✨ Total datasets found: {len(datasets)}")
    except Exception as e:
        print(f"❌ Error scanning datasets: {e}")
        import traceback
        traceback.print_exc()
    
    return datasets


def set_dataset(dataset_name):
    """
    Set the active dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset directory
        
    Returns:
        dict: Status info with 'success', 'dataset_name', and 'path' keys
    """
    print(f"\n🎵 Attempting to set dataset: {dataset_name}")
    datasets = get_available_datasets()
    
    # Find the dataset
    for dataset in datasets:
        if dataset['name'] == dataset_name:
            dataset_path = dataset['path']
            
            try:
                # Update bacpipe config
                import bacpipe
                bacpipe.config.audio_dir = dataset_path
                print(f"✅ Successfully set bacpipe.config.audio_dir to: {dataset_path}")
                
                return {
                    'success': True,
                    'dataset_name': dataset_name,
                    'path': dataset_path,
                    'message': f'Dataset switched to {dataset_name}'
                }
            except Exception as e:
                print(f"❌ Failed to set dataset: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(e),
                    'message': f'Failed to set dataset: {str(e)}'
                }
    
    print(f"❌ Dataset '{dataset_name}' not found in available datasets")
    return {
        'success': False,
        'error': 'Dataset not found',
        'message': f'Dataset "{dataset_name}" not found'
    }

