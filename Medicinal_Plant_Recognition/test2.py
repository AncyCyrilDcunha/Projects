import numpy as np
import os
import seaborn as sns  # Import seaborn for heatmap
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define file paths and directories
dir_path = 'D:/pythoncode/train'

# Load data generators
train = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, validation_split=0.1, 
                            rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                            width_shift_range=0.1, height_shift_range=0.1)
validation_generator = train.flow_from_directory(dir_path, target_size=(100, 100), batch_size=32, 
                                                  class_mode='categorical', subset='validation')

# Load the trained model
model = load_model('trained_model.h5')

# Evaluate the model on the validation dataset
val_steps = validation_generator.samples // validation_generator.batch_size
evaluation_results = model.evaluate(validation_generator, steps=val_steps, verbose=1)

# Handle multiple values returned from evaluation
val_loss = evaluation_results[0]
val_accuracy = evaluation_results[1]

print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Collect predictions and true labels
validation_generator.reset()  # Reset the generator to start from the beginning

y_true = []
y_pred = []

for _ in range(val_steps):
    images, labels = validation_generator.next()
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    y_true.extend(true_classes)
    y_pred.extend(predicted_classes)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Debug target names
target_names = [k for k, v in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])]
print('Target Names:', target_names)  # Debug output to ensure correct class names

# Calculate and print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report with zero_division parameter
try:
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
except TypeError as e:
    print(f'Error in classification report: {e}')

# Compute and display confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix (optional)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
