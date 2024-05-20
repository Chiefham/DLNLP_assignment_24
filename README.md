# DLNLP_assignment_24

---

This document will first introduce the specific roles 
of each file or folder and how to use the project code.

---
Datasets/mlb.pkl: Binary encoding file obtained by 
using the fit() method in the MultiLabelBinarizer() 
class to convert the training set labels.

---
Datasets/muti-classification-test.txt: Test 
set file containing test data and labels.

---
Datasets/muti-classification-train.txt: 
Training set file containing training data and labels.

---
chinese_bert_wwm_L-12_H-768_A-12/: This folder 
contains the pre-trained model of BERT-wwm, 
used for invoking BERT-wwm model in the code.

---
model/bert-textcnn-loss-acc.png: Image showing the 
changes of training accuracy, training loss, 
validation accuracy, and validation loss during epochs.

---
model/bert_textcnn.h5: Trained model combining BERT-wwm and TextCNN.

---
Evaluation.py: This file contains a custom function for model evaluation.

---
Models.py: This file contains two custom functions, 
each constructing the TextCNN model and the combined 
BERT-wwm and TextCNN model.

---
config.py: Configuration file providing hyperparameter 
configurations for training and testing.

---
main.py: Main file including data loading, 
train-test data split, data encoding, model training and saving, 
training parameters plotting, model loading, and evaluation.

---
utils.py: This file contains three custom functions 
providing some tool functions for training 
and testing.

---
Before running the project code, ensure that the dependencies are 
installed according to the requirements specified in requirement.txt. 
The project code runs on Python 3.6. If you need to configure the
hyperparameters of training, you can directly configure them in 
the config.py file. If you need to adjust the model structure or 
the type of pre-trained model, you can modify them in the Models.py 
file. To train and evaluate the model, simply run the main.py file. 
If you only want to evaluate the model, comment out the training and
plotting parts, then run the main.py file.
