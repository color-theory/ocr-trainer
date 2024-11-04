### Basic Architecture of Our CNN

The following is an outline of the basic architecture for our Convolutional Neural Network (CNN), which we will use for character recognition:

#### 1. Input Layer
- **Input Size**: (50, 50, 1)
  - Represents a 50x50 grayscale image.

#### 2. Convolutional Layer 1
- **Filters**: 32
  - Number of filters used to extract features from the input.
- **Kernel Size**: (3, 3)
  - Size of the sliding window used to scan the image.
- **Activation Function**: ReLU
  - Applies a non-linearity by setting negative values to zero, allowing the network to learn complex features.
- **Padding**: 'same'
  - Preserves the size of the input image by adding extra pixels around the border as needed.

#### 3. Max Pooling Layer 1
- **Pooling Size**: (2, 2)
  - Reduces the size of each feature map by half while retaining the most important information.

#### 4. Convolutional Layer 2
- **Filters**: 64
  - Increases the number of filters to learn more complex features.
- **Kernel Size**: (3, 3)
- **Activation Function**: ReLU
- **Padding**: 'same'

#### 5. Max Pooling Layer 2
- **Pooling Size**: (2, 2)

#### 6. Flatten Layer
- **Flatten**: Converts the 2D feature maps into a 1D vector for use in the fully connected layers.

#### 7. Fully Connected (Dense) Layer 1
- **Units**: 128
  - Number of neurons in this layer.
- **Activation Function**: ReLU
- **Dropout**: 0.5
  - Helps prevent overfitting by randomly setting half of the neurons to zero during training.

#### 8. Output Layer
- **Units**: Number of character classes (e.g., 62 for A-Z, a-z, 0-9)
- **Activation Function**: Softmax
  - Converts the output to a probability distribution over character classes.

### Summary of CNN Architecture
- The CNN starts with two **convolutional layers** with increasing filter numbers, each followed by a **max pooling layer** to downsample the feature maps.
- After flattening, the network has a **fully connected layer** with 128 units, followed by a final output layer to classify the input image into one of the character classes.
