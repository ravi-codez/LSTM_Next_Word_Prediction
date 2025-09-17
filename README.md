# LSTM Next Word Prediction

This repository contains a machine learning project that uses an LSTM (Long Short-Term Memory) based Recurrent Neural Network to predict the next word in a given sequence of text. The project is implemented in both Python scripts and Jupyter notebooks.

---

### `experiments.ipynb`: Model Development & Training

**Project Description:**  
This notebook details the process of building and training an LSTM and GRU model for next word prediction. It uses a sample dataset (`hemlet.txt`) for demonstration but can be adapted to other text corpora.

**Dataset:**  
- `hemlet.txt`: A plain-text file used for training the model.  

**Technologies:**  
- Python  
- Pandas  
- NumPy  
- TensorFlow / Keras  
- Scikit-learn  
- Pickle (for saving tokenizer)  

**Methodology:**  

1. **Data Preprocessing**  
   - The text is cleaned and tokenized.  
   - Token sequences are generated using Keras `Tokenizer`.  
   - Input-output pairs are created where the model learns to predict the next word.  
   - Sequences are padded to ensure uniform input length.  

2. **Model Architecture**  
   - An LSTM-based sequential model is constructed.  
   - Embedding layer for word representations.  
   - One or more LSTM layers to capture sequential dependencies.  
   - Dense output layer with softmax activation for vocabulary prediction.  

3. **Training**  
   - The model is compiled using the Adam optimizer.  
   - `CategoricalCrossentropy` is used as the loss function.  
   - Accuracy is tracked as the main evaluation metric.  
   - Early stopping is applied to prevent overfitting.  
   - Trained weights are saved as `next_word_LSTM.h5`.  
   - Tokenizer is saved as `tokenizer.pkl`.  

---

### `main.py`: Running Predictions

**Project Description:**  
This script loads the trained LSTM model and tokenizer to generate predictions on new text input.  

