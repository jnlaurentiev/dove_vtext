"""
//******************************************************************************
// FILENAME:           dove_ml.py
// DESCRIPTION:        This Natural Language Processing Python script contains pseudocode for classical
//                     machine learning models used for the DOVE Project.
// AUTHOR(s):          John Novoa-Laurentiev
//
// For those interested in learning more about these models, contact us at BWHMTERMS@bwh.harvard.edu .
//******************************************************************************
"""


def load_data(input_file, note_column_idx, label_column_idx, id_id=2):
    with open(input_file, encoding='utf-8', newline='') as f:
        csv_reader = reader(f)
        headers = next(csv_reader)
        csv_data = [(row[note_column_idx], int(row[label_column_idx]), row[id_id]) for row in csv_reader]
    description, label, note_id = zip(*csv_data)
    input_x = description
    output_y = label
    return list(input_x), list(output_y), list(note_id)


def run_baseline(model_type, test_file):
    train_x, train_y, train_note_id = load_data(test_file, note_column_idx=1, label_column_idx=2, id_id=0)
    test_x, test_y, test_note_id = load_data(test_file, note_column_idx=1, label_column_idx=2, id_id=0)
    label2id, id2label = build_alphabet(test_y, True)
    positive_label = 1
    positive_id = label2id[positive_label]

    train_weight = x_tfidftransformer.fit_transform(x_vectorizer.fit_transform(train_x)).toarray()
    model = train(train_weight)

    test_preds, test_golds = model_predict(test_x, test_y, model, model_type, label2id)
    results = calc_metric(test_golds, test_preds)

    return results


def model_predict(test_x, test_y, model, model_type, label2id):
    test_weight = x_tfidftransformer.transform(x_vectorizer.transform(test_x)).toarray()

    test_label_id = []
    for label in test_y:
        test_label_id.append(label2id[label])

    test_preds = model.predict(test_weight)

    return test_preds, test_y


def calc_metric(labels, preds):
    ROC_AUC, roc_ci_lower, roc_ci_upper, roc_scores = score_ci(labels, preds, auroc_score)

    PRC_AUC, prc_ci_lower, prc_ci_upper, prc_scores = score_ci(labels, preds, auprc_score)

    F1, f1_ci_lower, f1_ci_upper, f1_scores = score_ci(labels, preds, f1_score)

    TPR, tpr_ci_lower, tpr_ci_upper, tpr_scores = score_ci(labels, preds, sensitivity_score)

    TNR, tnr_ci_lower, tnr_ci_upper, tnr_scores = score_ci(labels, preds, specificity_score)

    PPV, ppv_ci_lower, ppv_ci_upper, ppv_scores = score_ci(labels, preds, ppv_score)

    NPV, npv_ci_lower, npv_ci_upper, npv_scores = score_ci(labels, preds, npv_score)

    ACC, acc_ci_lower, acc_ci_upper, acc_scores = score_ci(labels, preds, accuracy_score)

    return metric_scores


def preprocess_text(text):
    text = text.lower()
    text = sub(r'regex pattern 1', '', text)
    text = sub(r'regex pattern 2', '', text)

    return text


if __name__ == '__main__':
    train_data = './data/input_train.csv'
    test_data = './data/input_test.csv'
    output_dir = './ml_output/final'
    model_list = ['logistic', 'svm', 'randforest', 'xgboost']
    for md in model_list:
        results = run_baseline(md, test_data)
        test1_result = dataframe([results])
        test1_result.columns = "F1-score\tlower\tupper\tSensitivity\tlower\tupper\tSpecificity\tlower\tupper\tPrecision\tlower\tupper\tNPV\tlower\tupper\tAccuracy\tlower\tupper\tAUROC\tlower\tuppper\tAUPRC\tlower\tupper".split("\t")
        test1_result.to_csv(path.join(output_dir, f"{md}_performance.csv"), index=False)
