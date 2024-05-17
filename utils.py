import config
import pickle
import numpy as np
from tqdm import tqdm
from keras_bert import get_custom_objects
from keras.models import load_model
from keras_bert import load_vocabulary, Tokenizer
from sklearn.metrics import hamming_loss, classification_report

def load_data(txt_file_path):
    text_list = []
    label_list = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            label_list.append(line[0].split('|'))
            text_list.append(line[1])
    return text_list, label_list

# 对文本编码
def encoding_text(content_list,tokenizer):
    token_ids = []
    segment_ids = []
    for line in tqdm(content_list):
        token_id, segment_id = tokenizer.encode(first=line, max_len=config.max_len)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    encoding_res = [np.array(token_ids), np.array(segment_ids)]
    return encoding_res

def predict_single_text(text,model,tokenizer,mlb):
    token_id, segment_id = tokenizer.encode(first=text, max_len=config.max_len)
    prediction = model.predict([[token_id], [segment_id]])[0]

    indices = [i for i in range(len(prediction)) if prediction[i] > 0.5]
    lables = [mlb.classes_.tolist()[i] for i in indices]
    one_hot = np.where(prediction > 0.5, 1, 0)
    return one_hot, lables