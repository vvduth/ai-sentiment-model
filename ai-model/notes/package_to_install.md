## first letr install some lib wthatwe need
* tourch:  for tensor computation and deep learning
  ```bash
  pip install torch torchvision torchaudio librosa
  ```

  ```python
  return {
            'text_input': {
                # squeeze to remove batch dimension , we need squeeze because tokenizer returns batch dimension
                'input_ids': text_input['input_ids'].squeeze(),  # remove batch dimension
                'attention_mask': text_input['attention_mask'].squeeze()
            },
            'video_frames': video_frames,
        }
  ```

  why do we need to squeeze the batch dimension here?
    - The batch dimension is typically added by tokenizers to accommodate multiple inputs at once. However, if you're processing a single input (like a single sentence or a single video), you may want to remove this extra dimension to simplify your data structure and make it compatible with models that expect inputs without a batch dimension. Squeezing the batch dimension helps in aligning the input shape with the model's expected input shape, ensuring that the data is correctly processed during inference or training.
  -  before squeezing, the shape of 'input_ids' and 'attention_mask' would typically be :
  -  (1, sequence_length) , where 1 is the batch size for a single input. After squeezing, the shape becomes (sequence_length,), removing the batch dimension.
  


# Shape of 'input_ids' and 'attention_mask' Before Squeeze

## Visual Representation

### Before Squeeze: `(1, 128)` - 2D Tensor

```
Shape: [batch_size=1, sequence_length=128]

┌─────────────────────────────────────────────────────────────────┐
│ Batch 0 (the only sample)                                        │
│ ┌───┬───┬───┬───┬───┬───┬───┬─────┬───┬───┬───┬───┬───┬───┬───┐│
│ │101│123│456│789│234│567│890│ ... │345│678│901│102│102│102│102││
│ └───┴───┴───┴───┴───┴───┴───┴─────┴───┴───┴───┴───┴───┴───┴───┘│
│  ↑                                                             ↑  │
│  Token 0                                          Token 127      │
└─────────────────────────────────────────────────────────────────┘
   Dimension 0: Batch (size=1)
   Dimension 1: Sequence (size=128)
```

### After Squeeze: `(128,)` - 1D Tensor

```
Shape: [sequence_length=128]

┌───┬───┬───┬───┬───┬───┬───┬─────┬───┬───┬───┬───┬───┬───┬───┐
│101│123│456│789│234│567│890│ ... │345│678│901│102│102│102│102│
└───┴───┴───┴───┴───┴───┴───┴─────┴───┴───┴───┴───┴───┴───┴───┘
 ↑                                                             ↑
 Token 0                                          Token 127

   Only Dimension 0: Sequence (size=128)
```

## Code Example with Shapes

```python
# Before squeeze
text_input = self.tokenizer(row['Utterance'], 
                            padding='max_length',
                            truncation=True,
                            max_length=128,
                            return_tensors='pt')

print(f"input_ids shape BEFORE squeeze: {text_input['input_ids'].shape}")
# Output: torch.Size([1, 128])
#                     ↑  ↑
#                     │  └─ sequence_length (max_length=128)
#                     └──── batch_size (always 1 for single input)

print(f"attention_mask shape BEFORE squeeze: {text_input['attention_mask'].shape}")
# Output: torch.Size([1, 128])

# After squeeze
squeezed_input_ids = text_input['input_ids'].squeeze()
squeezed_attention_mask = text_input['attention_mask'].squeeze()

print(f"input_ids shape AFTER squeeze: {squeezed_input_ids.shape}")
# Output: torch.Size([128])
#                     ↑
#                     └─ only sequence_length remains

print(f"attention_mask shape AFTER squeeze: {squeezed_attention_mask.shape}")
# Output: torch.Size([128])
```

## Why Squeeze?

The tokenizer always returns tensors with a batch dimension `[batch_size, sequence_length]` because it's designed to process multiple inputs at once. When you're loading a single sample in `__getitem__`, you don't need the batch dimension yet—the DataLoader will add it back later when batching multiple samples together.

**Transformation flow:**
1. **Tokenizer output:** `(1, 128)` - includes unnecessary batch dimension
2. **After squeeze:** `(128,)` - clean 1D tensor for single sample
3. **DataLoader batching:** `(batch_size, 128)` - batch dimension added when multiple samples are loaded


## step: 
1  define data set class with __init__ , _load_video_frames,_extract_audio_features, __len__, __getitem__ methods
what hose this method do:
- __init__: Initialize dataset, load metadata, set up tokenizer and transforms
- _load_video_frames: Load and preprocess video frames from a given path
- _extract_audio_features: Extract audio features from a given path
- __len__: Return the total number of samples in the dataset
- __getitem__: Return a single data point at the given index
=> with these method implement, we are ready to load data for training


## models:

![alt text](image-5.png)
video: resnet3d 18 layer
audio:  raw spectrogram + cnn
text encodeer: bert base uncased

=> all the data get from 3 model will be concatnated into a single tensor then then put into fusionlayer for training
* the fusionlayer will learn the relationship between these 3 modal data and output the final prediction
* we wil tak ethe output of the fusion layer and then pass it to emotion classifier and sentiment classifier to get the final prediction

=> so lets build those layers

## first layer : trainable layers
* linear: fully connected layer : every neuron in this layer is connected to every neuron in the previous layer
  => one of the simplest layer , greate for classification task, ypou define inout and output neurons

* conv1d: convolutional layer for 1d data : useful for sequence data like audio signals
  => it will learn local patterns in the data by applying convolutional filters

## second : functional layers: doesnt learn,  just perform specific operations
* Relu : activation function that introduces non-linearity
  => it will output 0 for negative input and output the input itself for positive input
* dropout: prevent overfitting by randomly setting a fraction of input units to 0 during training
  => helps the model generalize better by reducing reliance on specific neurons
* maxpool1d: downsampling layer for 1d data, only keep the strongest features
  => it will reduce the dimensionality of the input by taking the maximum value over a defined window
* adaptiveavgpool1d: downsampling layer that outputs a fixed size regardless of input size
  => it will compute the average value over a defined window to reduce dimensionality, helpful for variable-length input sequences

## third :  normalization layers: help stabilize and speed up training
* batchnorm1d: normalize the input across the batch dimension
  => helps in stabilizing and speeding up training by reducing internal covariate shift


## workflow of the training process: (kind of =) )
* video => fomrat input tensor as [batch_size, channels, depth, height, width] (reorder dimension) => video frame with pre tained resnet3d , get output from it (process features) => linear layer to reduce dimension (128) (make size smaller)  => relu for non linearity => dropout for regularization (prevent overfitting) 

* text  => input ids and attention mask => bert model  (understand  each word in context of the sentence) => get pooled output (summary of the sentence) (768) => project to smaller dimension with linear layer (128) 

* audio => (inputsize 64) cov1d layer to learn local patterns in audio spectrogram (64)=> (input size 64) batchnorm1d to stabilize training => relu for non linearity => maxpool1d to downsample (output 2) => (64, k = 3) conv1d layer to learn higher level features (128)=> batchnorm1d => relu => adaptiveavgpool1d to get fixed size output [batch_size, 128, time_steps] =>linear layer to project to smaller dimension (128) => relu for non linearity => dropout for regularization  


=> all these 3 modal features will be concatnated [batch_size, 128 * 3] => fusion layer to learn relationship between these modal features => relu for non linearity => dropout for regularization => emotion classifier and sentiment classifier to get final prediction

the output will be emotion classifier (7 classes) and sentiment classifier (3 classes)

### optimizer: adam optimizer with learning rate 1e-4
* we use adam optimizer because it combines the advantages of both AdaGrad and RMSProp, making it well-suited for a wide range of deep learning tasks. Adam adapts the learning rate for each parameter individually based on the first and second moments of the gradients, which helps in faster convergence and better performance, especially in scenarios with sparse gradients or noisy data.
### loss function: cross entropy loss for both emotion and sentiment classification
* loss function is used to measure the difference between the predicted output and the true labels during training. Cross-entropy loss is particularly suitable for multi-class classification tasks, such as emotion and sentiment classification, because it quantifies the dissimilarity between the predicted probability distribution (output of the model) and the actual distribution (true labels). By minimizing this loss, the model learns to make more accurate predictions.
### learning rate scheduler: reduceLROnPlateau to reduce learning rate when validation loss plateaus
* recuse learnign rate if it does not importe for certain epochs. gradient descent will get stuck in local minima if learning rate is too high. so reducing learning rate will help the model to converge better

## A sequential container.
what is nn.Sequential in PyTorch?
`nn.Sequential` is a container module in PyTorch that allows you to create a neural network by stacking layers sequentially. It simplifies the process of building models by allowing you to define a sequence of layers in a single line of code. Each layer is added in the order it is defined, and the input is passed through each layer one after the other.