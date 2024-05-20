
# pre-trained bert
bert_config_path = './chinese_bert_wwm_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = './chinese_bert_wwm_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = './chinese_bert_wwm_L-12_H-768_A-12/vocab.txt'


# dataset location
train_dataset_path = "./Datasets/multi-classification-train.txt"
test_dataset_path = "./Datasets/multi-classification-test.txt"

# train parameter
epochs = 10
batch_size = 4
max_len = 256
learning_rate = 3e-5
