import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from glob import glob
import matplotlib.pyplot as plt

# Load image paths for each class

very_mild = glob('C:/Users/anany/Desktop/VIT/RESEARCH/sem 7-3 cloud maam/project/dataset/Very_Mild_Demented/*.jpg')
mild = glob('C:/Users/anany/Desktop/VIT/RESEARCH/sem 7-3 cloud maam/project/dataset/Mild_Dementedr/*.jpg')
moderate = glob('C:/Users/anany/Desktop/VIT/RESEARCH/sem 7-3 cloud maam/project/dataset/Moderate_Demented/*.jpg')
non = glob('C:/Users/anany/Desktop/VIT/RESEARCH/sem 7-3 cloud maam/project/dataset/Non_Demented/*.jpg')

# Load and preprocess images
data_paths = very_mild + mild + moderate + non
data = []
for path in data_paths:
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize
    data.append(img_array)
data = np.array(data)

# Create labels
labels = ['very_mild'] * len(very_mild) + ['mild'] * len(mild) + ['moderate'] * len(moderate) + ['non'] * len(non)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Define a pre-trained model for transfer learning
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Create an ensemble using different classifiers
classifiers = [
    ('random_forest', RandomForestClassifier(n_estimators=100)),
    ('xgboost', XGBClassifier(n_estimators=100)),
    ('logistic_regression', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True))
]

# Create a StackingClassifier as the ensemble
ensemble = StackingClassifier(estimators=classifiers, final_estimator=model)

# Train the ensemble
ensemble.fit(X_train, y_train)

# Evaluate the ensemble
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {accuracy}")

# Calculate Specificity, Precision, Recall, F1 Score
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate AUC-ROC
y_pred_proba = ensemble.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, average='macro')
print(f"AUC-ROC Score: {roc_auc}")

# Generate SHAP values for the ensemble
explainer = shap.Explainer(ensemble)
shap_values = explainer(X_test)

# Interpretation of SHAP values and ensemble predictions
shap.summary_plot(shap_values, X_test)

# Plot Loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
