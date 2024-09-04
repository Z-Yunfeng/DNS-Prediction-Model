import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix


def input_data(filepath):
    """导入数据集，去除重复行、含缺失值行，返回特征集和标签"""

    data = pd.read_excel(filepath)
    data.dropna(axis=0, how='any', inplace=True)
    data.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    y = data['DNS']
    X = data.drop(['DNS'], axis=1)

    return data, X, y

def scale_num_feature(X, X_train, X_test):
    """标准化数值型特征"""
    
    scaler = StandardScaler()

    feature_names = X.columns.tolist()
    unique_values = X.nunique()
    non_binary_features = unique_values[unique_values>2].index.tolist() # 非二值特征（即数值型特征）的索引

    X_train[non_binary_features] = scaler.fit_transform(X_train[non_binary_features])
    X_test[non_binary_features] = scaler.transform(X_test[non_binary_features])
    
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test

def feature_selection(model, num_features, X_train, y_train, X_test):
    """基于嵌入法选择特征"""
    
    selector = SelectFromModel(model, max_features=num_features).fit(X_train, y_train)
    feature_names = selector.get_feature_names_out().tolist()

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    features_selected = pd.DataFrame(feature_names, columns=['Features'])

    return X_train, X_test, features_selected

def evaluate_model(ppl, X_train, y_train, X_test, y_test, kf=5, name=''):
    """评估模型在训练集和测试集上的性能"""

    val_acc = cross_val_score(ppl, X_train, y_train, cv=kf)
    ppl.fit(X_train, y_train)
    y_pred = ppl.predict(X_test)
    reports = classification_report(y_test, y_pred)
    mc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = ['Non-DNS','DNS']
    conf_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    print(f'{name}\'s results of 10-fold cross-validation are as follows: \n {val_acc} \n')
    print(f'{name}\'s mean accuracy of 10-fold cross-validation is {val_acc.mean():.3g}')
    print(f'{name}\'s Matthews Correlation Coefficient is {mc:.3g} \n')
    print(f'{name}\'s performance on test set is as follows:\n{reports}')

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size":10}, cmap="Blues")
    plt.title(f'{name}', fontsize=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()