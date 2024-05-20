import config
from tqdm import tqdm
from sklearn.metrics import hamming_loss, classification_report
from utils import load_data,predict_single_text


def evaluate(mlb,model,tokenizer):
    test_x, test_y = load_data(config.test_dataset_path)
    true_y_list = mlb.transform(test_y)

    pred_y_list = []
    pred_labels = []
    for text in tqdm(test_x):
        pred_y, label = predict_single_text(text,model,tokenizer,mlb)
        pred_y_list.append(pred_y)
        pred_labels.append(label)

    # cal acc
    test_len = len(test_y)
    correct_count = 0
    for i in range(test_len):
        if test_y[i] == pred_labels[i]:
            correct_count += 1
    accuracy = correct_count / test_len

    print(classification_report(true_y_list, pred_y_list, target_names=mlb.classes_.tolist(), digits=4))
    print("accuracy:{}".format(accuracy))
    print("hamming_loss:{}".format(hamming_loss(true_y_list, pred_y_list)))
