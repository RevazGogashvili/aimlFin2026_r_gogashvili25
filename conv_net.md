1.1 From Dense Architectures to Biological Topology
As noted by Professor Gogishvili, while fully connected (dense) networks are "universal," their universality comes at a high cost of training time and computational resources. When processing an image of 
100
×
100
100×100
, a single dense neuron requires 10,000 weights. CNNs overcome this by modeling the topology of the human eye and brain.
In the human eye, the retina contains light-sensitive cells. Biological neurons are connected to localized areas of the retina rather than the entire visual field. This "vision chain" ensures that the network focuses on spatial relationships, significantly reducing the number of parameters through Weight Sharing.
1.2 Mathematical Framework and the "Sliding" Neuron
The fundamental operation of a neuron in a convolutional layer follows the standard activation formula, but applied locally:
y
=
σ
(
∑
i
=
1
n
w
i
x
i
+
b
)
y=σ( 
i=1
∑
n
​
 w 
i
​
 x 
i
​
 +b)
Where:
x
i
x 
i
​
 
: Input signals from a localized area.
w
i
w 
i
​
 
: Weights (Kernel/Filter).
b
b
: Bias term.
σ
σ
: Activation function (e.g., ReLU, Sigmoid).
Unlike dense networks, a Weight Matrix (Filter/Kernel) is shared across the entire input. A single neuron "slides" across the image, calculating outputs step-by-step. This results in a Feature Map. For example, an input of 
12
×
14
12×14
 pixels can be mapped to an 
8
×
10
8×10
 output volume through this localized connectivity.
1.3 3D Convolution and Hyperparameters
In modern applications, we process Input Volumes rather than 2D planes.
Input Volume: Typically expressed as 
W
×
H
×
D
W×H×D
 (Width, Height, Depth/Channels).
Filters: As shown in the technical slides, a convolutional layer may have multiple filters (
W
0
,
W
1
,
.
.
.
W 
0
​
 ,W 
1
​
 ,...
). For an input volume with 3 color channels (RGB) and a 
1
×
1
1×1
 padding, a 
7
×
7
×
3
7×7×3
 input processed by 
3
×
3
×
3
3×3×3
 filters with a Stride of 2 produces a 
3
×
3
×
2
3×3×2
 output volume.
Key Terms:
Stride: The shift magnitude of the filter. A higher stride reduces the output dimensions.
Padding: Filling the edges of the input with zeros to ensure the filter fits the original image dimensions.
Dilated Convolution: A technique where the filter is modeled as a "mesh," skipping certain pixels to expand the receptive field without increasing parameters.
1.4 Hierarchical Feature Detection
CNNs learn patterns through increasing complexity across layers:
Low-Level Features: Kernels specialize in detecting edges, lines, and color gradients.
Mid-Level Features: Combinations of edges form textures and motifs.
High-Level Features: Deep layers recognize complex object parts (e.g., wheels, mirrors, or specific code structures in cybersecurity).
1.5 Optimization: Pooling and Dropout
Max Pooling: To reduce parameters and prevent overfitting, we use a 
2
×
2
2×2
 matrix to downsample the feature map, keeping only the maximal value (the "essence" of the information).
Dropout: This regularization method randomly "deactivates" neurons during training, forcing the network to learn more robust features and enhancing trainability.
2. Visualizations
Note: These images should be saved in the task_1 folder of the repository.
eye_topology.png: Illustration of the retinal connection of neurons (Slide 6).
convolution_volume.png: The 
7
×
7
×
3
7×7×3
 input volume calculation with filters 
W
0
W 
0
​
 
 and 
W
1
W 
1
​
 
 (Slide 9).
feature_hierarchy.png: Visualization of Low, Mid, and High-level features (Slide 33).
max_pooling.png: Diagram showing 
2
×
2
2×2
 filters with stride 2 reducing a grid (Slide 35).
dropout_layer.png: Comparison between a standard network and a dropout-applied network (Slide 37).
3. Cybersecurity Application: Malware Visual Classification
In cybersecurity, CNNs are utilized to identify malware families by converting binary files into grayscale images. This circumvents traditional signature-based detection, as the structural "texture" of the malware (data sections, code blocks, headers) remains consistent even if the code is obfuscated.
Implementation Code (Python)
code
Python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Simulating Malware Binary-to-Image Data
# Raw bytes of a .exe are reshaped into 64x64 pixel grayscale images
def load_malware_data():
    X = np.random.rand(200, 64, 64, 1).astype('float32') # 200 samples
    y = np.random.randint(0, 2, 200) # 0: Benign, 1: Malware
    return X, y

X_train, y_train = load_malware_data()

# 2. Designing the CNN Architecture
model = models.Sequential([
    # First Conv Layer: 32 Filters, 3x3 Kernel, ReLU Activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)), # Max Pooling to reduce dimensionality
    
    # Second Conv Layer: 64 Filters for complex pattern recognition
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fully Connected Transition
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout for regularization
    layers.Dense(1, activation='sigmoid') # Binary Classification
])

# 3. Compilation with Cybersecurity Metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 4. Model Architecture Summary
model.summary()

# 5. Training (Simulated)
# model.fit(X_train, y_train, epochs=10, batch_size=32)