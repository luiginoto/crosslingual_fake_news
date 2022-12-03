import os
from dataloader import MetaLearningSystemDataLoader
from protomaml import ProtoMAMLFewShotClassifier
from experiment_builder import ExperimentBuilder
from utils.parser_utils import get_args
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

os.environ["DATASET_DIR"] = "/Users/luiginoto/Documents/NYU/Classes/NLP with Representation Learning/Final Project/nlp-project/datasets"

args, device = get_args()

data = MetaLearningSystemDataLoader

model = ProtoMAMLFewShotClassifier(args=args, device=device)

# print(data.dataset.task_set_sizes['train'].keys())

maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)

maml_system.model.load_model(
                model_save_dir=maml_system.saved_models_filepath,
                model_name="train_model",
                model_idx="best",
            )

train_dataloader, dev_dataloader = maml_system.data.get_finetune_dataloaders('MM-COVID_it', 0, 12345)

_, best_loss, curr_loss, accuracy, is_correct_preds, student_logits_list, student_preds_list, y_true_list, teacher_preds_list = maml_system.model.finetune_epoch(
                    None,
                    maml_system.model.classifier.config,
                    train_dataloader,
                    dev_dataloader,
                    task_name='MM-COVID_it',
                    epoch=maml_system.epoch,
                    eval_every=1,
                    model_save_dir=maml_system.saved_models_filepath,
                    best_loss=0,
                    return_student_teacher_preds=True
)

print(len(is_correct_preds))
print(len(y_true_list))
print(len(student_preds_list))
print(len(teacher_preds_list))


print('Preds')
print(student_logits_list[:10])
print(student_preds_list[:10])

print('Labels')
print(y_true_list[:10])
print(teacher_preds_list[:10])

print('is correct')
print(is_correct_preds[:10])

y_true_np = np.array(teacher_preds_list)
y_pred_score_np = np.array(student_logits_list)[:, 1].flatten()
y_pred_np = np.array(student_preds_list)

print(y_true_np[:10])
print(y_pred_score_np[:10])
print(y_pred_np[:10])

RocCurveDisplay.from_predictions(y_true_np, y_pred_score_np)

path = os.path.join(args.experiment_name, 'plots')
os.makedirs(path)
plt.savefig(os.path.join(path, 'roc_curve.pdf'))
