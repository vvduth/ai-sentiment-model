import os
import argparse
import torchaudio
import librosa
import torch
from training.meld_training import prepare_dataloaders, CSV_TRAIN_PATH, VIDEO_TRAIN_DIR
from models import MultimodalSentimentModel

# AWS sagemaker training script

# Get the model directory from the environment variable set by SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION  = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST  = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")


# Configure PyTorch to use expandable segments for CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def pargs_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # Data directories
    parser.add_argument('--dir', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--val_dir', type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument('--test_dir', type=str, default=SM_CHANNEL_TEST)
    parser.add_argument('--model_dir', type=str, default=SM_MODEL_DIR)
    
    return parser.parse_args()

def main():
    # install ffmpeg
    print("available audio backend:")
    print(str(torchaudio.list_audio_backends()))
    
    args = pargs_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # trackiung initialization GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated()/1024**3
        print(f"Initial GPU memory allocated: {memory_used:.2f} GB")
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )
    
    print(f""" traning csv path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}""")
    print(f""" training video path: {os.path.join(args.train_dir, 'train_splits')}""")
    
    model = MultimodalSentimentModel().to(device)
        
        
    
    
    
if __name__ == "__main__":
    args = pargs_args()
    main()