import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch
from meld_training import MELDDataset, VIDEO_TRAIN_DIR, CSV_TRAIN_PATH


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters for feature extraction
        for param in self.bert.parameters():
            # use it , not train it
            param.requires_grad = False  # Freeze BERT parameters
        
        # Project BERT output to 128 dimensions
        self.projection = nn.Linear(768, 128)
    
    def forward(self, input_ids, attention_mask):
        """
        This method encodes the input text using BERT and projects it to a 128-dimensional space.
        input_ids: Tensor of shape [batch_size, seq_length]
        attention_mask: Tensor of shape [batch_size, seq_length]
        Returns:
            Tensor of shape [batch_size, 128]
        """
        # extract bert embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # use the [CLS] token representation
        pooler_output = outputs.pooler_output  # shape: [batch_size, 768]
        
        return self.projection(pooler_output)  # shape: [batch_size, 128]

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pre-trained 3D ResNet model for video feature extraction
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze ResNet parameters
        
        # Modify the final fully connected layer to project to 128 dimensions
        num_fts = self.backbone.fc.in_features
        # replace the fc layer with Sequential containing Linear layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x shape: [batch_size, frames, channels, height, width] => need to transpose to [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)  
        return self.backbone(x)  # shape: [batch_size, 128]
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # lower level feature extraction using conv1d layers
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher level feature extraction
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output size will be (batch_size, 128, 1)
        )
        
        for param in self.conv_layer.parameters():
            param.requires_grad = False  # Freeze Conv1d parameters, no need to train from scratch
        
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
            ) # trainable layer to project to 128 dimensions
        
    def forward(self, x):
        # x shape: [batch_size, 64, 300]
        x = x.squeeze(1)
        
        features = self.conv_layer(x)  
        # shape: [batch_size, 128, 1]
        return self.projection(features.squeeze(-1))  
    
class MultimodalSentimentModel(nn.Module):
    def __init__(self ):
        super().__init__()
        
        # encoder
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        # fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # classification layer
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # 7 emotion classes 
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 sentiment classes
        )

    def forward(self, text_inputs, video_frames, audio_features):
        """_summary_

        Args:
            text_inputs (_type_): _description_
            video_frames (_type_): _description_
            audio_features (_type_): _description_
        """
        text_features = self.text_encoder(
            text_inputs['input_ids'], 
            text_inputs['attention_mask']
        )
        
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        
        # concatenate features
        combined_features = torch.cat([
            text_features, 
            video_features, 
            audio_features
             ], dim=1)  # shape: [batch_size, 128*3]
        
        # fusion layer 
        fused_features = self.fusion_layer(combined_features)  # shape: [batch_size, 256]
        emotion_outputs = self.emotion_classifier(fused_features)  # shape: [batch_size, 7]
        sentiment_outputs = self.sentiment_classifier(fused_features)  # shape: [batch_size, 3]
        
        return {
            'emotions': emotion_outputs,
            'sentiments': sentiment_outputs
        }

if __name__ == "__main__":
    dataset = MELDDataset(CSV_TRAIN_PATH, VIDEO_TRAIN_DIR)
    sample = dataset[0]
    
    model = MultimodalSentimentModel()
    model.eval()
    
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)  # add batch dimension
    audio_features = sample['audio_features'].unsqueeze(0)  # add batch dimension
    
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        # get emotion probabilities
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]
        
    emotion_map = {
             0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'joy',
            4: 'neutral',
            5: 'sadness',
            6: 'surprise'
        }
    sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
    for i, prob in enumerate(emotion_probs):
            print(f"Emotion: {emotion_map[i]}, Probability: {prob.item():.2f}")
    for i, prob in enumerate(sentiment_probs):
            print(f"Sentiment: {sentiment_map[i]}, Probability: {prob.item():.2f}")
    
    print("Predicted Emotion Probabilities:", emotion_probs)
        