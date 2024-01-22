import numpy as np
from PIL import Image
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        try:
            label = int(filename.split('_')[0])  # Assuming the label is before the first underscore
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(np.array(img).flatten())  # Flatten the image to a 1D array
                labels.append(label)
        except ValueError:
            print(f"Ignoring file {filename} with invalid label.")
    
    return np.array(images), np.array(labels)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(X, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_input_hidden
    hidden_output = sigmoid(hidden_input)
    output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_hidden_output
    output_probs = softmax(output_layer_input)
    return hidden_output, output_probs

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

def train(X, y, learning_rate, epochs, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output):
    for epoch in range(epochs):
        # Forward pass
        hidden_output, output_probs = forward_pass(X, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output)

        # Backpropagation
        output_error = output_probs - y
        hidden_error = np.dot(output_error, weights_hidden_output.T) * (hidden_output * (1 - hidden_output))

        # Update weights and biases
        weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_error)
        bias_hidden_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
        bias_input_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

# Load training and testing data
train_folder = r'D:\\django\\ai_project\\train'
test_folder = r'D:\\django\\ai_project\\test'

X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

# Initialize weights and biases
input_size = X_train.shape[1]
hidden_size = 128
output_size = num_classes

np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_input_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden_output = np.zeros((1, output_size))

# Training the model
learning_rate = 0.01
epochs = 100
train(X_train, y_train_one_hot, learning_rate, epochs, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output)


_, train_output_probs = forward_pass(X_train, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output)
train_predictions = np.argmax(train_output_probs, axis=1)
train_accuracy = np.mean(train_predictions == y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

_, test_output_probs = forward_pass(X_test, weights_input_hidden, bias_input_hidden, weights_hidden_output, bias_hidden_output)
test_predictions = np.argmax(test_output_probs, axis=1)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
