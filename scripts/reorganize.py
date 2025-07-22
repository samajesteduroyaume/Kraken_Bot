import os
import shutil
from pathlib import Path

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def move_files(mapping):
    """Move files according to the mapping."""
    for src, dst in mapping.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Skip if source doesn't exist
        if not src_path.exists():
            print(f"Source not found: {src_path}")
            continue
            
        # Create destination directory if needed
        if dst_path.suffix:  # If it's a file
            ensure_dir(dst_path.parent)
        else:  # If it's a directory
            ensure_dir(dst_path)
            
        # Move the file/directory
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        print(f"Moved: {src_path} -> {dst_path}")

# Define the file mapping
file_mapping = {
    # Core functionality
    'src/core/analysis/technical.py': 'src/kraken_bot/core/indicators/technical.py',
    'src/core/api/kraken.py': 'src/kraken_bot/api/kraken.py',
    'src/core/backtesting/backtester.py': 'src/kraken_bot/core/backtesting.py',
    'src/core/risk/manager.py': 'src/kraken_bot/risk/manager.py',
    'src/core/trading/execution.py': 'src/kraken_bot/core/trading/execution.py',
    'src/core/trading/risk/manager.py': 'src/kraken_bot/risk/manager.py',
    'src/core/trading/signals/__init__.py': 'src/kraken_bot/core/trading/signals.py',
    'src/core/trading/strategy.py': 'src/kraken_bot/core/trading/strategy.py',
    'src/core/trading/trader.py': 'src/kraken_bot/core/trading/trader.py',
    
    # ML components
    'src/ml/predictor.py': 'src/kraken_bot/ml/predictor.py',
    'src/ml/trainer.py': 'src/kraken_bot/ml/training/trainer.py',
    'src/ml/models/__init__.py': 'src/kraken_bot/ml/models/__init__.py',
    
    # Utils
    'src/utils/helpers.py': 'src/kraken_bot/utils/helpers.py',
    'src/utils/logger.py': 'src/kraken_bot/utils/logger.py',
    
    # Config
    'config/settings.py': 'config/settings.py',
    'config/logging_config.py': 'config/logging.yaml',
    
    # Scripts
    'services/auto_train.py': 'scripts/auto_train.py',
}

if __name__ == "__main__":
    print("Starting project reorganization...")
    move_files(file_mapping)
    print("Reorganization complete!")
