import config
import keras
from keras_bert import load_trained_model_from_checkpoint


def textcnn(inputs):

    # use differnet kernel size with 3,4,5 to extract feature
    kernel_size = [3, 4, 5]
    cnn_features = []
    for size in kernel_size:
        cnn = keras.layers.Conv1D(filters=256, kernel_size=size)(
            inputs)  # shape=[batch_size,maxlen-2,256]
        cnn = keras.layers.GlobalMaxPooling1D()(cnn)  # shape=[batch_size,256]
        cnn_features.append(cnn)
    # concat each feature
    output = keras.layers.concatenate(cnn_features,
                                      axis=-1)  # [batch_size,256*3]

    # return the extraction result of textcnn
    return output

def build_bert_textcnn_model(config_path, checkpoint_path, class_nums):
    """
    :param config_path: bert_config.json location
    :param checkpoint_path: bert_model.ckpt location
    :param class_nums: final output dim
    :return: final model
    """
    # load pre-trained bert model
    bert = load_trained_model_from_checkpoint(
        config_file=config_path,
        checkpoint_file=checkpoint_path,
        seq_len=None
    )
    # [cls]
    cls_features = keras.layers.Lambda(lambda x: x[:, 0],name='cls')(bert.output)  # shape=[batch_size,768]
    word_embedding = keras.layers.Lambda(lambda x: x[:, 1:-1],name='word_embedding')(bert.output)  # shape=[batch_size,maxlen-2,768]
    # get features from textcnn
    cnn_features = textcnn(word_embedding)  # shape=[batch_size,cnn_output_dim]
    # concat cls and textcnn features
    all_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)  # shape=[batch_size,cnn_output_dim+768]
    # drop out
    all_features = keras.layers.Dropout(0.2)(all_features)  # shape=[batch_size,cnn_output_dim+768]
    # dim reduction
    dense = keras.layers.Dense(units=256, activation='relu')(all_features)  # shape=[batch_size,256]
    # output
    output = keras.layers.Dense(units=class_nums,activation='sigmoid')(dense)  # shape=[batch_size,class_nums]
    # build model
    model = keras.models.Model(bert.input, output, name='bert-textcnn')
    model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(config.learning_rate),metrics=['accuracy'])
    return model


