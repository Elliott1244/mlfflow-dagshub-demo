import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Elliott1244', repo_name='mlfflow-dagshub-demo', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

mlflow.set_tracking_uri('https://dagshub.com/Elliott1244/mlfflow-dagshub-demo.mlflow')

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_depth = 15

mlflow.set_experiment('iris_dt')

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_artifact('iris_dt.py')

    mlflow.sklearn.log_model(dt, 'Decision Tree')

    mlflow.set_tag('author', 'harsha')
    mlflow.set_tag('model', 'Decision tree')


    


    print('accuracy', accuracy)