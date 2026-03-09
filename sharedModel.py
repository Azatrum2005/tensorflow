import tensorflow as tf
from tensorflow import keras
layers=tf.keras.layers
Dense=layers.Dense
Model=tf.keras.Model
import numpy as np

# Define input tensor
input_layer = layers.Input(shape=(2,))

# Shared layers
shared = layers.Dense(32, activation='relu')(input_layer)

# Output branches
output1 = layers.Dense(1,activation='sigmoid', name='regression_output')(shared)
output2 = layers.Dense(3, activation='softmax', name='classification_output')(shared)

# Create model
model = Model(inputs=input_layer, outputs=[output1, output2])

# Compile the model with appropriate loss functions
model.compile(
    optimizer='adam',
    loss={
        'regression_output': 'mse',
        'classification_output': 'sparse_categorical_crossentropy'  # Use sparse for integer labels
    },
    metrics={
        'regression_output': ['mae'],
        'classification_output': ['accuracy']
    }
)

# Model summary
# model.summary()

# Generate sample data with proper labels
X_train = np.array([[1,2], [2,3], [3,4], [4,5]])
y_reg = np.array([1.0, 1.0, 0.0, 0.0])      # Regression targets
y_cls = np.array([0, 1, 2, 0])            # Classification labels (0-2 for 3 classes)

# Train the model
model.fit(
    X_train,
    {'regression_output': y_reg, 'classification_output': y_cls},
    epochs=10,
    # verbose=1
)

# Make predictions
test_sample = np.array([[1,3]])
predictions = model.predict(test_sample)
print(f"Regression prediction: {predictions[0][0][0]:.4f}")
print(f"Classification probabilities: {predictions[1][0]}")
