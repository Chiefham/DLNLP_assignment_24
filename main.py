import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from utils import load_data,encoding_text
import config
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from Models import build_bert_textcnn_model
from keras_bert import Tokenizer, load_vocabulary
from Evaluation import evaluate
from keras.models import load_model
from keras_bert import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



## load data
data_x, data_y = load_data(config.train_dataset_path)
train_content_x,val_content_x,train_label_y,val_label_y = \
    train_test_split(data_x,data_y,test_size=0.2,random_state=42)
test_content_x, test_label_y = load_data(config.test_dataset_path)

token_dict = load_vocabulary(config.bert_dict_path)
tokenizer = Tokenizer(token_dict)


# shuffle
index = [i for i in range(len(train_content_x))]
random.shuffle(index)  # shuffule index
train_content_x = [train_content_x[i] for i in index]
train_label_y = [train_label_y[i] for i in index]

# encoding data
train_x = encoding_text(train_content_x,tokenizer)
val_x = encoding_text(val_content_x,tokenizer)
test_x = encoding_text(test_content_x,tokenizer)

# encoding label
mlb = MultiLabelBinarizer()
mlb.fit(train_label_y)
pickle.dump(mlb, open('./Datasets/mlb.pkl', 'wb'))

train_y = np.array(mlb.transform(train_label_y))
val_y = np.array(mlb.transform(val_label_y))
test_y = np.array(mlb.transform(test_label_y))


# train
model = build_bert_textcnn_model(config.bert_config_path,
                                 config.bert_checkpoint_path, len(mlb.classes_))
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                    batch_size=config.batch_size, epochs=config.epochs,
                    callbacks=[early_stopping])
# model save
model.save("./model/bert_textcnn.h5")


# plot loss
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
# plot acc
plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
# save plot
plt.savefig("./bert-textcnn-loss-acc.png")


# load model
model = load_model('./model/bert_textcnn.h5', custom_objects=get_custom_objects())
mlb = pickle.load(open('./Datasets/mlb.pkl', 'rb'))

# evaluate model
evaluate(mlb,model,tokenizer)