from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
import librosa


CSV_DEV_PATH = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/dev/dev_sent_emo.csv'
VIDEO_DEV_DIR = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/dev/dev_splits_complete'

CSV_TEST_PATH = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/test/test_sent_emo.csv'
VIDEO_TEST_DIR = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/test/output_repeated_splits_test'

CSV_TRAIN_PATH = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/train/train_sent_emo.csv'
VIDEO_TRAIN_DIR = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/train/train_splits'

def collate_fn(batch):
    """
    Custom collate function to handle None values in the batch
    """
    # filter out None values
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        # tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_map = {
            'neutral': 4,
            'joy': 3,
            'sadness': 5,
            'anger': 0,
            'fear': 2,
            'surprise': 6,
            'disgust': 1
        }
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
    def _load_video_frames(self, video_path):
        """
        Load exactly 30 frames from video, resize to 224x224, normalize to 0-1.
        Output shape: (30, 3, 224, 224)
        """
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            print("Loading video frames from:", video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # try and read first frame to validate the video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read video file: {video_path}")
            
            # reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # extract exactly 30 frames to match the model input
            while (len(frames)) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
                frame = frame / 255.0  # Normalize between 0 and 1
                frames.append(frame)
        except Exception as e:
            print(f"Error loading video frames: {str(e)}")
            return None
        finally:
            cap.release()
            
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # pad with zeros if less than 30 frames, truncate if more
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
            
        # Convert from [frames, height, width, channels] to [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, video_path):
        """
        Extract audio from video using ffmpeg, convert to mel spectrogram.
        Settings: 16kHz sample rate, mono channel, pcm_s16le codec.
        """
        # create audio path from video path but with .wav extension
        audio_path = video_path.replace('.mp4', '.wav')
        
        try: 
            print(f"Extracting audio features from: {audio_path}")
            
            # extract audio using ffmpeg CLI
            subprocess.run(['ffmpeg', 
                            '-i', video_path,
                            '-vn',  # output only audio
                            '-acodec', 'pcm_s16le',  # PCM signed 16-bit little-endian
                            '-ar', '16000',  # 16kHz sample rate
                            '-ac', '1',  # mono channel
                            audio_path
                            ], check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            
            # load audio using librosa (more reliable than torchaudio.load on Windows)
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            waveform = torch.FloatTensor(waveform).unsqueeze(0)  # Add channel dimension
            
            # extract mel spectrogram features
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512)
            mel_spec = mel_spectrogram(waveform)
            
            # normalize 
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            # pad or truncate to ensure consistent length of 300
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else: 
                # truncate time dimension to 300
                mel_spec = mel_spec[:, :, :300]
            return mel_spec
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio features with ffmpeg: {str(e)}")
            return None
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None
        finally:
            # clean up the temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return a single data point (text, video frames, audio features, labels)
        """
        # if idx is a tensor, convert to int
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        try:
            # construct video filename
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # tokenize text input (max length 128)
            text_input = self.tokenizer(row['Utterance'], 
                                        padding='max_length',
                                        truncation=True,
                                        max_length=128,
                                        return_tensors='pt')
            
            # load video frames and extract audio features
            video_frames = self._load_video_frames(video_path)   
            audio_features = self._extract_audio_features(video_path)
            
            # map sentiment and emotion to labels
            emotion_label = self.emotion_map.get(row['Emotion'].lower())
            sentiment_label = self.sentiment_map.get(row['Sentiment'].lower())
            
            return {
                'text_inputs': {
                    # squeeze to remove batch dimension , we need squeeze because tokenizer returns batch dimension
                    'input_ids': text_input['input_ids'].squeeze(),  # remove batch dimension
                    # attention mask is also squeezed, mask is useful for transformer models to ignore padding tokens
                    'attention_mask': text_input['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {video_path} index {idx}: {str(e)}")
            return None
        
def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir,
                        batch_size=32):
    """
    load train, dev, test datasets and create dataloaders
    with the specified batch size
    """
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader
           

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        CSV_TRAIN_PATH, VIDEO_TRAIN_DIR,
        CSV_DEV_PATH, VIDEO_DEV_DIR,
        CSV_TEST_PATH, VIDEO_TEST_DIR,
        batch_size=32
    )
    
    for batch in train_loader:
        print(batch['text_inputs']['input_ids'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break