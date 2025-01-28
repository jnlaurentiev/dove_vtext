"""
//******************************************************************************
// FILENAME:           dove_bioclinicalbert.py
// DESCRIPTION:        This Natural Language Processing Python script contains pseudocode for a sequence classification
//                     deep learning model used for the DOVE Project.
// AUTHOR(s):          John Novoa-Laurentiev
//
// For those interested in learning more about this model, contact us at BWHMTERMS@bwh.harvard.edu .
//******************************************************************************
"""



def auprc_score(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def auroc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


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


def init_training_args(epochs, batch_size, learning_rate):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return training_args


if __name__ == '__main__':
    val_df = read_excel("../data/input.xls")
    val_df["input"] = val_df["sentence_text"]
    val_df["label"] = val_df["label"]
    val_data = encode_(val_df.input.tolist(), val_df.label.tolist())
    output_dir = './dl_output/final'

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    training_args = init_training_args(epochs=x, batch_size=y, learning_rate=z)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )
    trainer.train()
    predictions = trainer.predict(val_data)
    pred_prob_df = pd.DataFrame(predictions.predictions)
    pred_prob_df = pred_prob_df.apply(softmax, axis=1)
    pred_prob_df.to_csv(os.path.join(output_dir, 'pred_probs.csv'))
    preds = np.argmax(predictions.predictions, axis=-1)
    calc_metric(predictions.label_ids, preds)
    labels_list = predictions.label_ids

    test1_result = calc_metric(predictions.label_ids, preds)
    test1_result = pd.DataFrame([test1_result])
    test1_result.columns = "F1-score\tlower\tupper\tSensitivity\tlower\tupper\tSpecificity\tlower\tupper\tPrecision\tlower\tupper\tNPV\tlower\tupper\tAccuracy\tlower\tupper\tAUROC\tlower\tuppper\tAUPRC\tlower\tupper".split("\t")
    test1_result.to_csv(os.path.join(output_dir, "output.csv"), index=False)
