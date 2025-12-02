# Deep Learning Model Guide: From Logistic Regression to Neural Networks

## Critical Issues in Your Original Code

### 1. **Parameter Reinitialization Every Epoch** ❌
```python
# WRONG - Reinitializes weights every iteration
for epoch in range(epochs):
    w = np.random.random(32)      # ← Problem: Resets w each epoch!
    b = np.random.randn()          # ← Problem: Resets b each epoch!
```

**Why it's wrong:** The model can't learn because weights are reset to random values at every epoch, discarding the gradient updates.

**Fix:**
```python
# CORRECT - Initialize once before the loop
w = np.random.randn(X_train.shape[1]) * 0.01
b = 0

for epoch in range(epochs):
    # Training code here - w and b persist across epochs
```

---

### 2. **Matrix Multiplication Error** ❌
```python
# WRONG - Element-wise multiplication
z = w * X_train[count:count+32] + b
```

**Why it's wrong:** 
- `X_train[count:count+32]` has shape (32, 8)
- `w` has shape (32,)
- Element-wise multiplication doesn't align dimensions correctly for linear regression

**Fix:**
```python
# CORRECT - Proper matrix multiplication
z = np.dot(X_batch, w) + b  # (32, 8) × (8,) = (32,)
```

---

### 3. **Batch Index Management** ❌
```python
# WRONG - count starts at 0 each epoch
for epoch in range(epochs):
    count = 0
    for batch in range(17):
        # Process same 32 samples repeatedly
    count += 32  # Only incremented after inner loop completes
```

**Why it's wrong:** Each batch processes indices [0:32], [0:32], ... instead of [0:32], [32:64], [64:96], etc.

**Fix:**
```python
# CORRECT - Proper batch iteration
for batch_start in range(0, len(X_train), batch_size):
    batch_end = min(batch_start + batch_size, len(X_train))
    X_batch = X_train[batch_start:batch_end]
    y_batch = y_train[batch_start:batch_end]
```

---

### 4. **Loss Calculation Issues** ❌
```python
# WRONG - Uses undefined variable 'count'
loss = -np.mean(y_train[count:count+32] * np.log(y_pred) + ...)
# 'count' may be out of bounds after the loop
```

**Fix:**
```python
# CORRECT - Calculate loss across all data
epoch_loss = 0
for batch_start in range(0, len(X_train), batch_size):
    # ... training ...
    batch_loss = binary_crossentropy_loss(y_batch, y_pred)
    epoch_loss += batch_loss

avg_epoch_loss = epoch_loss / num_batches
```

---

### 5. **Array Formatting in Print** ❌
```python
# WRONG - Tries to format array as float
print(f"w: {w:.4f}")  # w is array, not scalar!
```

**Fix:**
```python
# CORRECT 
print(f"w shape: {w.shape}, first weight: {w[0]:.4f}")
# Or for loss (which is scalar)
print(f"Loss: {loss:.6f}")
```

---

## Corrected Logistic Regression Code

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("Datasets/diabetes.csv")
X = df.drop("Outcome", axis=1)
Y = df["Outcome"].values
sc = StandardScaler()
X_scale = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scale, Y, random_state=42, test_size=0.3
)

# ✓ Initialize ONCE before epochs
w = np.random.randn(X_train.shape[1]) * 0.01
b = 0
lr = 0.01
epochs = 1000
batch_size = 32

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_crossentropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ✓ Training loop
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    # ✓ Proper batch iteration
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = min(batch_start + batch_size, len(X_train))
        X_batch = X_train_shuffled[batch_start:batch_end]
        y_batch = y_train_shuffled[batch_start:batch_end]
        
        # Forward pass
        z = np.dot(X_batch, w) + b  # ✓ Correct matrix multiplication
        y_pred = sigmoid(z)
        
        # Backward pass
        dw = np.dot(X_batch.T, (y_pred - y_batch)) / len(y_batch)
        db = np.mean(y_pred - y_batch)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
        
        # ✓ Calculate loss
        batch_loss = binary_crossentropy_loss(y_batch, y_pred)
        epoch_loss += batch_loss
        num_batches += 1
    
    avg_epoch_loss = epoch_loss / num_batches
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}, Loss: {avg_epoch_loss:.6f}")

# Evaluate
z_test = np.dot(X_test, w) + b
y_pred_test = sigmoid(z_test)
y_pred_test_binary = (y_pred_test > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred_test_binary)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

## Neural Network with Hidden Layers

### Key Differences from Logistic Regression

| Aspect | Logistic Regression | Neural Network |
|--------|-------------------|-----------------|
| **Layers** | 1 (input → output) | Multiple (input → hidden₁ → hidden₂ → output) |
| **Activation** | Sigmoid only | ReLU (hidden), Sigmoid (output) |
| **Parameters** | Single w, b | Multiple W, b for each layer |
| **Backprop** | Simple gradient | Chain rule through layers |
| **Capacity** | Limited (linear decision boundary) | Complex (non-linear boundaries) |

---

### Simple Neural Network Implementation

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size=1, lr=0.01):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes, e.g., [64, 32]
            output_size: Output size (1 for binary classification)
        """
        self.lr = lr
        self.weights = []
        self.biases = []
        
        # Build network layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * \
                np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU derivative for backpropagation"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid activation for output layer"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """Forward propagation through all layers"""
        self.z_values = []
        self.a_values = [X]
        
        A = X
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)
            A = self.relu(Z)
            self.a_values.append(A)
        
        # Output layer with sigmoid
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.z_values.append(Z)
        A = self.sigmoid(Z)
        self.a_values.append(A)
        
        return A
    
    def backward(self, y_true):
        """Backpropagation: compute gradients for all layers"""
        m = len(y_true)
        
        # Output layer gradient
        dZ = self.a_values[-1] - y_true.reshape(-1, 1)
        
        gradients_w = []
        gradients_b = []
        
        # Backprop through each layer
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.a_values[i].T, dZ) / m
            dB = np.mean(dZ, axis=0, keepdims=True)
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
            
            if i > 0:
                # Chain rule: backprop through ReLU
                dZ = np.dot(dZ, self.weights[i].T) * \
                     self.relu_derivative(self.z_values[i-1])
        
        return gradients_w, gradients_b
    
    def train(self, X_train, y_train, epochs=1000, batch_size=32):
        """Train the network"""
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                gradients_w, gradients_b = self.backward(y_batch)
                
                # Update parameters
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * gradients_w[i]
                    self.biases[i] -= self.lr * gradients_b[i]
                
                num_batches += 1
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss computed")

# Usage
nn = SimpleNeuralNetwork(
    input_size=8,           # 8 features
    hidden_sizes=[64, 32],  # 2 hidden layers: 64 and 32 neurons
    output_size=1           # Binary classification
)
nn.train(X_train, y_train, epochs=1000, batch_size=32)
```

---

## Architecture Visualization

```
Input Layer (8 features)
        |
Hidden Layer 1 (64 neurons, ReLU)
        |
Hidden Layer 2 (32 neurons, ReLU)
        |
Output Layer (1 neuron, Sigmoid)
        |
    Prediction (0-1)
```

---

## Key Concepts

### Weight Initialization
- **Xavier/Glorot**: `w = np.random.randn(in, out) * sqrt(1/in)`
- **He initialization**: `w = np.random.randn(in, out) * sqrt(2/in)` (for ReLU)
- Prevents vanishing/exploding gradients

### Activation Functions
- **ReLU**: Non-linear, prevents neuron saturation
- **Sigmoid**: Squashes output to [0, 1], used for binary classification output

### Backpropagation
Computes gradients using chain rule:
- Output layer gradient: `dZ = y_pred - y_true`
- Hidden layer: `dZ = (dZ @ W^T) * relu'(Z)`

### Batch Gradient Descent Variants
- **Batch**: All samples per update (slow, stable)
- **Stochastic**: One sample per update (noisy, fast)
- **Mini-batch**: subset of samples per update (balanced)

---

## Next Steps: Using TensorFlow/Keras

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=32, 
          validation_data=(X_test, y_test))
```

---

## Summary Table: Issues Fixed

| Issue | Original | Fixed |
|-------|----------|-------|
| Parameter init | Every epoch | Once before epochs |
| Matrix mult | `w * X` | `np.dot(X, w)` |
| Batch indexing | Incorrect loop | `range(0, len, batch_size)` |
| Loss calculation | Last batch only | All batches averaged |
| Architecture | None (logistic only) | Multiple hidden layers |
| Activation | Sigmoid only | ReLU + Sigmoid |
| Test evaluation | Missing | Included |
