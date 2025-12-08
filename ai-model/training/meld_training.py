from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
CSV_DEV_PATH = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/dev/dev_sent_emo.csv'
VIDEO_DEV_DIR = 'C:/Users/ducth/git_repos/ai-sentiment-model/ai-model/dataset/dev/dev_splits_complete'

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
        Load video frames from the given video path.
        This is a placeholder method and should be implemented to extract frames from the video.
        """
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            print("Loading video frames from:", video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # try and read first fame to validate the video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read video file: {video_path}")
            
            # reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # brake the video into 30 frames to match the model input
            while (len(frames)) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame= cv2.resize(frame, (224, 224))  # Resize to match model input size, keep consistent
                frame = frame / 255.0  # Normalize the frame between 0 and 1
                frames.append(frame)
        except Exception as e:
            print(f"Error loading video frames: {str(e)}")
            return None
        finally:
            cap.release()
            
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        ## if frames has less than 30 frames, pad with zeros, if more, truncate
        if len(frames) < 30:
            # take the 1st frame , copy the shape and fill with zeros then make it 30 frames in total
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
            
        # before permuting : [frames, height, width, channels]
        # after permuting : [frames, channels, height, width]
        # we permute the dimensions to match PyTorch's expected input format for Conv3D layers
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
    
    def _extract_audio_features(self, video_path):
        """
            get audio features from the video file
        """
        # create audio path from video path but with diff extension
        audio_path = video_path.replace('.mp4', '.wav')
        
        try: 
            print(f"Extracting audio features from: {audio_path}")
            
            # run this command in the termil with code
            subprocess.run(['ffmpeg', 
                            '-i', video_path,
                            '-vn',  # output only audio
                            '-acodec', 'pcm_s16le', # PCM signed 16-bit little-endian for audio
                            '-ar', '16000', # set -ar for audio sample rate
                            '-ac', '1', # set -ac for number of audio channels
                            audio_path
                            ], check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            
            # load the file into memory
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # extract mel spectrogram features for audio
            meld_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                                                    n_mels=64,
                                                    n_fft=1024,
                                                    hop_length=512)
            mel_spec = meld_spectrogram(waveform)
            
            # normalize 
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            # pad or truncate to ensure consistent length
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else: 
                # keep all the channels and frequency bins but truncate the time dimension
                mel_spec = mel_spec[:,:, :300]
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio features with ffmpeg: {str(e)}")
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
        
        
        
        
    # override len method
    def __len__(self):
        """
        return the length of the dataset
        """
        return len(self.data)
    def __getitem__(self, idx):
        # Implement this method to return a single data point at the given index
        row = self.data.iloc[idx]
        # Get text data
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        
        path = os.path.join(self.video_dir, video_filename )
        video_path_exist = os.path.exists(path)
        
        if video_path_exist == False:
            raise FileNotFoundError(f"Video file not found for filename: {path}")
        
        # tokenize the text input
        text_input = self.tokenizer(row['Utterance'], 
                                    padding='max_length',
                                    truncation=True,
                                    max_length=128,
                                    return_tensors='pt')
        
        # load the video fram from the video file
        video_frames = self._load_video_frames(path)         
        print(video_frames)

if __name__ == "__main__":
    meld = MELDDataset(CSV_DEV_PATH, VIDEO_DEV_DIR)
    print(meld[0])
    