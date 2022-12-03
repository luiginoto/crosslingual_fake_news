import os
from dataloader import MetaLearningSystemDataLoader
from protomaml import ProtoMAMLFewShotClassifier
from experiment_builder import ExperimentBuilder
from utils.parser_utils import get_args
from data_preprocessing.preprocessing_covid import *
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns



def meta_test_preds(args, device, experiment_name, seed):
    args.experiment_name = experiment_name

    if os.path.exists('datasets/Dataset'):
        shutil.rmtree('datasets/Dataset')

    langs = {'en', 'es', 'it', 'pt', 'fr', 'hi'}

    test_lang = experiment_name[-2:]
    val_lang = 'fr' if test_lang != 'fr' else 'it'
    train_langs = list(langs.difference(set(val_lang) or set(test_lang)))

    preprocess_covid('../Datasets/mm_covid', 'datasets/Dataset', train_langs, [val_lang], [test_lang])

    data = MetaLearningSystemDataLoader
    model = ProtoMAMLFewShotClassifier(args=args, device=device)
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)

    maml_system.model.load_model(
                model_save_dir=os.path.join(experiment_name, 'saved_models'),
                model_name="train_model",
                model_idx="best",
            )

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

    return y_true, y_pred_score, y_pred_labels


if __name__ == '__main__':

    os.environ["DATASET_DIR"] = "/Users/luiginoto/Documents/NYU/Classes/NLP with Representation Learning/Final Project/nlp-project/datasets"
    args, device = get_args()

    seed = 12345

    experiment_names = glob.glob('covid_experiment_*')

    roc_curve_scores = {}

    for experiment_name in experiment_names:

        y_true, y_pred_score, y_pred_labels = meta_test_preds(args, device, experiment_name, seed)
        fpr, tpr, _ = roc_curve(y_true, y_pred_score)
        auc = roc_auc_score(y_true, y_pred_score)

        roc_curve_scores['MM-COVID_' + str(experiment_name[-2:])] = {}
        roc_curve_scores['MM-COVID_' + str(experiment_name[-2:])]['fpr'] = fpr
        roc_curve_scores['MM-COVID_' + str(experiment_name[-2:])]['tpr'] = tpr
        roc_curve_scores['MM-COVID_' + str(experiment_name[-2:])]['auc'] = auc

    fig = plt.figure(figsize=(8,6))

    for experiment_name in roc_curve_scores:
        plt.plot(
            roc_curve_scores[experiment_name]['fpr'], 
            roc_curve_scores[experiment_name]['tpr'], 
            label="{}, AUC={:.3f}".format(experiment_name, roc_curve_scores[experiment_name]['auc'])
            )
        
    plt.plot([0,1], [0,1], color='black', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('roc_curve.pdf')






# print(data.dataset.task_set_sizes['train'].keys())

# maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)

# maml_system.model.load_model(
#                 model_save_dir=maml_system.saved_models_filepath,
#                 model_name="train_model",
#                 model_idx="best",
#             )

# train_dataloader, dev_dataloader = maml_system.data.get_finetune_dataloaders('MM-COVID_it', 0, 12345)

# _, best_loss, curr_loss, accuracy, is_correct_preds, student_logits_list, student_preds_list, y_true_list, teacher_preds_list = maml_system.model.finetune_epoch(
#                     None,
#                     maml_system.model.classifier.config,
#                     train_dataloader,
#                     dev_dataloader,
#                     task_name='MM-COVID_it',
#                     epoch=maml_system.epoch,
#                     eval_every=1,
#                     model_save_dir=maml_system.saved_models_filepath,
#                     best_loss=0,
#                     return_student_teacher_preds=True
# )

# print(len(is_correct_preds))
# print(len(y_true_list))
# print(len(student_preds_list))
# print(len(teacher_preds_list))


# print('Preds')
# print(student_logits_list[:10])
# print(student_preds_list[:10])

# print('Labels')
# print(y_true_list[:10])
# print(teacher_preds_list[:10])

# print('is correct')
# print(is_correct_preds[:10])

# y_true_np = np.array(teacher_preds_list)
# y_pred_score_np = np.array(student_logits_list)[:, 1].flatten()
# y_pred_np = np.array(student_preds_list)

# print(y_true_np[:10])
# print(y_pred_score_np[:10])
# print(y_pred_np[:10])

# RocCurveDisplay.from_predictions(y_true_np, y_pred_score_np)

# path = os.path.join(args.experiment_name, 'plots')
# os.makedirs(path)
# plt.savefig(os.path.join(path, 'roc_curve.pdf'))
