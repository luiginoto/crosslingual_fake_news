import os
import glob
from dataloader import MetaLearningSystemDataLoader
from protomaml import ProtoMAMLFewShotClassifier
from experiment_builder import ExperimentBuilder
from utils.parser_utils import get_args
from utils.storage import save_to_json
from data_preprocessing.preprocessing_covid import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import seaborn as sns



def meta_test_preds(experiment_name):

    langs = {'en', 'es', 'it', 'pt', 'fr', 'hi'}

    test_lang = experiment_name[-2:]
    val_lang = 'fr' if test_lang != 'fr' else 'it'
    train_langs = list(langs.difference({val_lang, test_lang}))

    print(train_langs)
    print(val_lang)
    print(test_lang)

    os.environ["DATASET_DIR"] = "/Users/luiginoto/Documents/NYU/Classes/NLP with Representation Learning/Final Project/nlp-project/datasets_" + test_lang
    args, device = get_args()
    args.experiment_name = experiment_name
    print(args.experiment_name)

    data = MetaLearningSystemDataLoader
    model = ProtoMAMLFewShotClassifier(args=args, device=device)
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)

    maml_system.model.load_model(
                model_save_dir=os.path.join(experiment_name, 'saved_models'),
                model_name="train_model",
                model_idx="best",
            )

    seeds = [42 + i for i in range(2)] # range(args.num_evaluation_seeds)]

    test_scores = {}
    losses = []
    accuracies = []
    f1_scores = []

    for seed in seeds:

        train_dataloader, dev_dataloader = maml_system.data.get_finetune_dataloaders('MM-COVID_' + str(test_lang), 0, seed)

        _, best_loss, curr_loss, accuracy, is_correct_preds, student_logits_list, student_preds_list, y_true_list, teacher_preds_list = maml_system.model.finetune_epoch(
                            None,
                            maml_system.model.classifier.config,
                            train_dataloader,
                            dev_dataloader,
                            task_name='MM-COVID_' + str(test_lang),
                            epoch=maml_system.epoch,
                            eval_every=1,
                            model_save_dir=os.path.join(experiment_name, 'saved_models'),
                            best_loss=0,
                            return_student_teacher_preds=True
        )

        y_true = np.array(teacher_preds_list)
        y_pred_score = np.array(student_logits_list)[:, 1].flatten()
        y_pred_labels = np.array(student_preds_list)

        print('accuracy')
        print(accuracy)
        print(np.mean(y_true == y_pred_labels))

        accuracies.append(accuracy)
        losses.append(curr_loss)
        f1_scores.append(f1_score(y_true, y_pred_labels))
    
    test_scores["test_accuracy_mean"] = np.mean(accuracies)
    test_scores["test_accuracy_std"] = np.std(accuracies)
    test_scores["test_f1_score_mean"] = np.mean(f1_scores)
    test_scores["test_f1_score_std"] = np.std(f1_scores)
    test_scores["test_loss_mean"] = np.mean(losses)
    test_scores["test_loss_std"] = np.std(losses)

    return y_true, y_pred_score, y_pred_labels, test_scores


if __name__ == '__main__':

    experiment_names = glob.glob('covid_experiment_*')

    roc_curve_scores = {}
    result = {}

    for experiment_name in experiment_names:

        print(f'Computing test metrics on {experiment_name}')

        y_true, y_pred_score, y_pred_labels, test_scores = meta_test_preds(experiment_name)

        fpr, tpr, _ = roc_curve(y_true, y_pred_score)
        auc = roc_auc_score(y_true, y_pred_score)
        roc_curve_scores['MM-COVID_test_' + str(experiment_name[-2:])] = {}
        roc_curve_scores['MM-COVID_test_' + str(experiment_name[-2:])]['fpr'] = fpr
        roc_curve_scores['MM-COVID_test_' + str(experiment_name[-2:])]['tpr'] = tpr
        roc_curve_scores['MM-COVID_test_' + str(experiment_name[-2:])]['auc'] = auc

        result['MM-COVID_test_' + str(experiment_name[-2:])] = test_scores
    
    save_to_json('test_results.json', result)

    fig = plt.figure(figsize=(8,6))

    for experiment_name in roc_curve_scores:
        plt.plot(
            roc_curve_scores[experiment_name]['fpr'], 
            roc_curve_scores[experiment_name]['tpr'], 
            label="{}, AUC={:.3f}".format(experiment_name, roc_curve_scores[experiment_name]['auc'])
            )
        
    plt.plot([0,1], [0,1], color='black', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontsize=15)
    plt.legend(prop={'size':8}, loc='lower right')

    plt.savefig('roc_curve.pdf')