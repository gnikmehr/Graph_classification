import pandas as pd
import json
import optuna
import numpy as np
import random
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn.naive_bayes
import sklearn.tree
import sklearn.linear_model
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
from sklearn.metrics import classification_report


seed = 42 #52, 99, 120


X_train = pd.read_csv("final_dataset/X_train216_14_1.csv")
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)
X_train = X_train.loc[:, ~X_train.columns.str.contains("^Unnamed")]
X_train = X_train.drop('out_degree', axis=1)
X_train = X_train.drop('neighborhood', axis=1)
X_train = X_train.drop('leverage', axis=1)
X_train = X_train.drop('clustering_coefficient', axis=1)
X_train = X_train.drop('in_degree', axis=1)
X_train = X_train.drop('harmonic', axis=1)
X_train = X_train.drop('closeness', axis=1)
X_train = X_train.drop('pagerank', axis=1)
X_train = X_train.drop('hub_score', axis=1)



X_val = pd.read_csv("final_dataset/X_val216_14_1.csv")
y_val = X_val['label']
X_val = X_val.drop('label', axis=1)
X_val = X_val.loc[:, ~X_val.columns.str.contains("^Unnamed")]
X_val = X_val.drop('out_degree', axis=1)
X_val = X_val.drop('neighborhood', axis=1)
X_val = X_val.drop('leverage', axis=1)
X_val = X_val.drop('clustering_coefficient', axis=1)
X_val = X_val.drop('in_degree', axis=1)
X_val = X_val.drop('harmonic', axis=1)
X_val = X_val.drop('closeness', axis=1)
X_val = X_val.drop('pagerank', axis=1)
X_val = X_val.drop('hub_score', axis=1)



X_test = pd.read_csv("final_dataset/X_test216_14_1.csv")
y_test = X_test['label']
X_test = X_test.drop('label', axis=1)
X_test = X_test.loc[:, ~X_test.columns.str.contains("^Unnamed")]
X_test = X_test.drop('out_degree', axis=1)
X_test = X_test.drop('neighborhood', axis=1)
X_test = X_test.drop('leverage', axis=1)
X_test = X_test.drop('clustering_coefficient', axis=1)
X_test = X_test.drop('in_degree', axis=1)
X_test = X_test.drop('harmonic', axis=1)
X_test = X_test.drop('closeness', axis=1)
X_test = X_test.drop('pagerank', axis=1)
X_test = X_test.drop('hub_score', axis=1)




def objective(trial):
    classifier_name = trial.suggest_categorical(
        "classifier", ["SVC", "RandomForest", "NaiveBayes", "CART", "MLPClassifier", "XGBoost", "Perceptron"]
    )
    
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        classifier_obj = sklearn.svm.SVC(C=svc_c, kernel=kernel, gamma="auto", probability=True, random_state=seed)
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("n_estimators", 10, 100)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, random_state=seed, 
        )
    elif classifier_name == "NaiveBayes":
        classifier_obj = sklearn.naive_bayes.GaussianNB()
    elif classifier_name == "CART":
        max_depth = trial.suggest_int("cart_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    elif classifier_name == "MLPClassifier":
        hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 50, 100)
        classifier_obj = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=500, random_state=seed)
    elif classifier_name == "Perceptron":
        alpha = trial.suggest_float("perceptron_alpha", 1e-5, 1e-1, log=True)
        classifier_obj = sklearn.linear_model.Perceptron(alpha=alpha, max_iter=1000, random_state=seed)
    elif classifier_name == "XGBoost":
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 32, log=True)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 1.0, log=True)
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
        classifier_obj = XGBClassifier(
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            n_estimators=xgb_n_estimators,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
        )
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    all_metrics = {key: [] for key in scoring.keys()}

    # Train on train set and evaluate on validation set
    classifier_obj.fit(X_train, y_train)

    for metric_name, metric_func in scoring.items():
        scorer = sklearn.metrics.get_scorer(metric_func)
        metric_value = scorer(classifier_obj, X_val, y_val)
        all_metrics[metric_name].append(metric_value)

    # Compute per-class precision & recall
    report = classification_report(y_val, classifier_obj.predict(X_val), output_dict=True)

    # Store precision & recall for each class
    all_metrics["precision_per_class"] = {cls: values["precision"] for cls, values in report.items() if isinstance(values, dict)}
    all_metrics["recall_per_class"] = {cls: values["recall"] for cls, values in report.items() if isinstance(values, dict)}

     # Log metrics (optional)
    trial.set_user_attr("metrics", all_metrics)

    if classifier_name not in best_results or best_results[classifier_name]['accuracy'] < all_metrics['accuracy']:
        best_results[classifier_name] = {
            'params': trial.params,
            'accuracy': all_metrics['accuracy'],
            'metrics': all_metrics,
           
        }
    
    # Return the primary metric to optimize
    return all_metrics['accuracy']
best_results = {}
# Run hyperparameter tuning
study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
study.optimize(objective, n_trials=100, n_jobs=-1)

print("--------------------------tunning finished------------------------------")


Perceptron_dic = {}
NaiveBayes_dic = {}
SVC_dic = {}
CART_dic = {}
RandomForest_dic = {}
XGBoost_dic = {}
MLPClassifier_dic = {}

def makedic(temp_dic):
   if 'params' not in temp_dic:
            temp_dic['params'] = [result['params']]
   else:
         temp_dic['params'].append(result['params'])
   if 'accuracy' not in temp_dic:
            temp_dic['accuracy'] = result['accuracy']
   else:
         temp_dic['accuracy'].append(result['accuracy'][0])
   for metric, value in result['metrics'].items():
      if metric.capitalize() == "Precision": 
            if 'Precision' not in temp_dic:
               temp_dic['Precision'] = value
            else:
               temp_dic['Recall'].append(value[0])
      if metric.capitalize() == "Recall":
            if 'Recall' not in temp_dic:
               temp_dic['Recall'] = value
            else:
               temp_dic['Recall'].append(value[0])
      if metric.capitalize() == "F1":
            if 'F1' not in temp_dic:
               temp_dic['F1'] = value
            else:
               temp_dic['F1'].append(value[0])
      if metric.capitalize() == "Roc_auc":
            if 'Roc_auc' not in temp_dic:
               temp_dic['Roc_auc'] = value
            else:
               temp_dic['Roc_auc'].append(value[0])
      if metric.capitalize() == "Precision_per_class":
            if 'Precision_0' not in temp_dic:
               temp_dic['Precision_0'] = [value['0']]
            else:
               temp_dic['Precision_0'].append(value['0'])
            if 'Precision_1' not in temp_dic:
               temp_dic['Precision_1'] = [value['1']]
            else:
               temp_dic['Precision_1'].append(value['1'])
      if metric.capitalize() == "Recall_per_class":
            if 'Recall_0' not in temp_dic:
               temp_dic['Recall_0'] = [value['0']]
            else:
               temp_dic['Recall_0'].append(value['0'])
            if 'Recall_1' not in temp_dic:
               temp_dic['Recall_1'] = [value['1']]
            else:
               temp_dic['Recall_1'].append(value['1'])
   return temp_dic
    

for classifier, result in best_results.items():
   if classifier == "Perceptron":
      Perceptron_dic = makedic(Perceptron_dic)
   if classifier == "NaiveBayes":
      NaiveBayes_dic = makedic(NaiveBayes_dic)
   if classifier == "SVC":
      SVC_dic = makedic(SVC_dic)
   if classifier == "CART":
      CART_dic = makedic(CART_dic)
   if classifier == "RandomForest":
      RandomForest_dic = makedic(RandomForest_dic)
   if classifier == "XGBoost":
      XGBoost_dic = makedic(XGBoost_dic)
   if classifier == "MLPClassifier":
      MLPClassifier_dic = makedic(MLPClassifier_dic)

methods = {"Perceptron": Perceptron_dic,
 "NaiveBayes": NaiveBayes_dic,
 "SVC": SVC_dic,
 "CART": CART_dic,
 "RandomForest":RandomForest_dic,
 "XGBoost":XGBoost_dic,
 "MLPClassifier": MLPClassifier_dic}


with open("final_dataset/val_results216_5.json", "r") as file:
    val_results = json.load(file)

for m in list(methods.keys()):
    val_results[m]['params'].append(methods[m]['params'][0])
    val_results[m]['accuracy'].append(methods[m]['accuracy'][0])
    val_results[m]['Precision'].append(methods[m]['Precision'][0])
    val_results[m]['Recall'].append(methods[m]['Recall'][0])
    val_results[m]['F1'].append(methods[m]['F1'][0])
    val_results[m]['Roc_auc'].append(methods[m]['Roc_auc'][0])
    val_results[m]['Precision_0'].append(methods[m]['Precision_0'][0])
    val_results[m]['Precision_1'].append(methods[m]['Precision_1'][0])
    val_results[m]['Recall_0'].append(methods[m]['Recall_0'][0])
    val_results[m]['Recall_1'].append(methods[m]['Recall_1'][0])

# print(val_results)
with open("final_dataset/val_results.json", "w") as file:
    json.dump(val_results, file, indent=4)

print("--------------------------val results saved------------------------------")


def makedic_test(temp_dic,scoring, b_model,  report):

   for metric_name, metric_func in scoring.items():
      scorer = sklearn.metrics.get_scorer(metric_func)
      metric_value = scorer(b_model, X_test, y_test)

      if metric_name == "accuracy":
         if 'accuracy' not in temp_dic:
            temp_dic['accuracy'] = [metric_value]
         else:
               temp_dic['accuracy'].append(metric_value)
      if metric_name == "precision":
         if 'precision' not in temp_dic:
            temp_dic['precision'] = [metric_value]
         else:
            temp_dic['precision'].append(metric_value)
      if metric_name == "recall":
         if 'recall' not in temp_dic:
            temp_dic['recall'] = [metric_value]
         else:
            temp_dic['recall'].append(metric_value)
      if metric_name == "f1":
         if 'f1' not in temp_dic:
            temp_dic['f1'] = [metric_value]
         else:
            temp_dic['f1'].append(metric_value)
      if metric_name == "roc_auc":
         if 'roc_auc' not in temp_dic:
            temp_dic['roc_auc'] = [metric_value]
         else:
            temp_dic['roc_auc'].append(metric_value)
   

   pv = {cls: values["precision"] for cls, values in report.items() if isinstance(values, dict)}
   rv = {cls: values["recall"] for cls, values in report.items() if isinstance(values, dict)}

   if 'precision0' not in temp_dic:
      temp_dic['precision0'] = [pv['0']]
   else:
      temp_dic['precision0'].append(pv['0'])
   if 'precision1' not in temp_dic:
      temp_dic['precision1'] = [pv['1']]
   else:
      temp_dic['precision1'].append(pv['1'])

   if 'recall0' not in temp_dic:
      temp_dic['recall0'] = [rv['0']]
   else:
      temp_dic['recall0'].append(rv['0'])
   if 'recall1' not in temp_dic:
      temp_dic['recall1'] = [rv['1']]
   else:
      temp_dic['recall1'].append(rv['1'])
         

   return temp_dic


scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }

Perceptron_dic_test = {}
NaiveBayes_dic_test = {}
SVC_dic_test = {}
CART_dic_test = {}
RandomForest_dic_test = {}
XGBoost_dic_test = {}
MLPClassifier_dic_test = {}

classifier_mapping = {
    "SVC": sklearn.svm.SVC,
    "RandomForest": sklearn.ensemble.RandomForestClassifier,
    "NaiveBayes": sklearn.naive_bayes.GaussianNB,
    "CART": sklearn.tree.DecisionTreeClassifier,
    "MLPClassifier": MLPClassifier,
    "Perceptron": sklearn.linear_model.Perceptron,
    "XGBoost": XGBClassifier,
}

# Evaluate best models on test set
print("\nBest results per classifier:")
for classifier, result in best_results.items():
    classifier_class = classifier_mapping.get(classifier)
    print(classifier_class)
    if classifier_class:
        b_model = classifier_class(**{
            k.replace("perceptron_alpha", "alpha")
            .replace("svc_c", "C")
            .replace("rf_max_depth", "max_depth")
            .replace("cart_max_depth", "max_depth")
            .replace("xgb_max_depth", "max_depth")
            .replace("xgb_learning_rate", "learning_rate")
            .replace("xgb_n_estimators", "n_estimators")  # Fix XGBoost param names
            : v for k, v in result["params"].items() if k != "classifier"
        })
    b_model.fit(X_train, y_train)

    if classifier == "RandomForest":
        print("---------------------------")
        importances = b_model.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
        print("---------------------------")
    # Compute per-class precision & recall
    report = classification_report(y_test, b_model.predict(X_test), output_dict=True)

    if classifier == "Perceptron":
      Perceptron_dic_test = makedic_test(Perceptron_dic_test,scoring, b_model, report)
    if classifier == "NaiveBayes":
      NaiveBayes_dic_test = makedic_test(NaiveBayes_dic_test,scoring, b_model, report)
    if classifier == "SVC":
      SVC_dic_test = makedic_test(SVC_dic_test,scoring, b_model, report)
    if classifier == "CART":
      CART_dic_test = makedic_test(CART_dic_test,scoring, b_model, report)
    if classifier == "RandomForest":
      RandomForest_dic_test = makedic_test(RandomForest_dic_test,scoring, b_model, report)
    if classifier == "XGBoost":
      XGBoost_dic_test = makedic_test(XGBoost_dic_test,scoring, b_model, report)
    if classifier == "MLPClassifier":
      MLPClassifier_dic_test = makedic_test(MLPClassifier_dic_test,scoring, b_model, report)


methods_test = {"Perceptron": Perceptron_dic_test,
 "NaiveBayes": NaiveBayes_dic_test,
 "SVC": SVC_dic_test,
 "CART": CART_dic_test,
 "RandomForest":RandomForest_dic_test,
 "XGBoost":XGBoost_dic_test,
 "MLPClassifier": MLPClassifier_dic_test}

with open("final_dataset/test_results.json", "r") as file:
    test_results = json.load(file)


for m in list(methods_test.keys()):
    test_results[m]['accuracy'].append(methods_test[m]['accuracy'][0])
    test_results[m]['precision'].append(methods_test[m]['precision'][0])
    test_results[m]['recall'].append(methods_test[m]['recall'][0])
    test_results[m]['f1'].append(methods_test[m]['f1'][0])
    test_results[m]['roc_auc'].append(methods_test[m]['roc_auc'][0])
    test_results[m]['precision0'].append(methods_test[m]['precision0'][0])
    test_results[m]['precision1'].append(methods_test[m]['precision1'][0])
    test_results[m]['recall0'].append(methods_test[m]['recall0'][0])
    test_results[m]['recall1'].append(methods_test[m]['recall1'][0])

# print(test_results)

with open("final_dataset/test_results216_5.json", "w") as file:
    json.dump(test_results, file, indent=4)

print("--------------------------test results saved------------------------------")

