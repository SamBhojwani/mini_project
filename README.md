# mini_project

AI-Based Waste Segregation System using Python + TensorFlow

# AI-Based Waste Segregation System for Smart Cities

### Overview:

A machine learning model that classifies waste images into Organic or Recyclable categories using CNN.

### Features:

- Image classification using TensorFlow/Keras.
- Streamlit web application for easy interaction.
- Promotes sustainable waste management.

### Dataset:

Sourced from Kaggle: [Waste Classification Dataset](https://www.kaggle.com/datasets)

### How to Run:

1. Clone this repository:
   git clone https://github.com/SamBhojwani/mini_project.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py

### Folder Structure:

AI-Waste-Segregation/
│
├── app.py
├── model.py
├── preprocess.py
├── waste_classifier.h5
├── dataset/
├── docs/
└── README.md

### Future Improvements:

- Integration with Raspberry Pi for real-time classification.
- Classifying more waste types (hazardous, e-waste, etc.).
  requirements.txt

List all required libraries:
streamlit
tensorflow
numpy
pandas
opencv-python
Pillow
.gitignore

To prevent uploading unnecessary files (like large datasets, model checkpoints, environment files):

# Ignore Python cache files

**pycache**/
\*.pyc

# Ignore virtual environment

venv/
env/

# Ignore model weights or large files

waste*classifier.h5
*.h5
\_.csv

# Ignore dataset directories if large

dataset/
