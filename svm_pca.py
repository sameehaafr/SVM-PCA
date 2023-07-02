import streamlit as st
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import precision_score, recall_score

# Load the data

def load_data():
    tumor_df = load_breast_cancer()
    df = pd.DataFrame(tumor_df.data, columns=tumor_df.feature_names)
    df['target'] = tumor_df.target
    return df

# Split the data

def split_data(df):
    X = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

default_c = 1.0
default_gamma = 'scale'
default_kernel = 'rbf'

c1 = 10
gamma1 = 0.01
kernel1 = 'rbf'

# Build basic SVM model
def build_basic_svm(X_train_scaled, y_train):
    clf_svm = SVC()
    clf_svm.fit(X_train_scaled,y_train)
    return clf_svm

def build_svm(C, gamma, kernel, X_train_scaled, y_train):
    clf_svm = SVC(C=C, gamma=gamma, kernel=kernel)
    clf_svm.fit(X_train_scaled,y_train)
    return clf_svm

# Evaluate the basic SVM model
def show_confusion_matrix(clf_svm, X_test_scaled, y_test):
    class_labels = ['Malignant', 'Benign']
    cm = confusion_matrix(y_test, clf_svm.predict(X_test_scaled))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Use GridSearchCV to find the best parameters
def find_best_params(X_train_scaled, y_train):
    param_grid = [
        {'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']}
        ]

    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=0
    )

    optimal_params.fit(X_train_scaled, y_train)
    c = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']
    kernel = optimal_params.best_params_['kernel']

    return c, gamma, kernel

# # Plot Scree Plot - PCA to reduce the number of features
# def scree_plot(X_train_scaled):
#     pca = PCA().fit(X_train_scaled)
#     #X_train_pca = pca.fit_transform(X_train_scaled)
#     per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
#     #labels = [str(x) for x in range(1, len(per_var)+1)]

#     fig = plt.bar(x=range(1, len(per_var)+1), height=per_var)
#     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#     plt.ylabel('Percentage of Explained Variance')
#     plt.xlabel('Principal Component')
#     plt.title('Scree Plot')
#     return plt.show()

def scree_plot(X_train_scaled):
    pca = PCA().fit(X_train_scaled)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    fig, ax = plt.subplots()
    ax.bar(x=range(1, len(per_var) + 1), height=per_var)
    ax.set_ylabel('Percentage of Explained Variance')
    ax.set_xlabel('Principal Component')
    ax.set_title('Scree Plot')

    return fig

# Build the model with the optimal parameters and the reduced number of features
def pca(X_train_scaled, X_test_scaled, y_train):
    pca = PCA(n_components = 2)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    param_grid = [
        {'C': [0.5, 1, 10, 100],
            'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']}
    ]

    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=0
    )

    optimal_params.fit(X_train_pca, y_train)
    c = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']
    kernel = optimal_params.best_params_['kernel']

    return c, gamma, kernel, X_train_pca, X_test_pca

# ---------------------------------------------DISPLAY------------------------------------------------------------- #

st.title('SVM for Classifying Tumors')
st.caption('The objective of this project is to build an SVM model that can classify tumor characteristics as either Malignant (non-cancerous) or Benign (cancerous). The data used for this project was obtained from the UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic')
st.caption('The code for this project can be found here: https://github.com/sameehaafr/SVM-PCA')


df = load_data()
st.dataframe(df)

X_train_scaled, X_test_scaled, y_train, y_test = split_data(df)

# Build basic SVM model
st.header('Basic SVM Model')
st.caption("Default SVM Parameters: C = 1.0, gamma = 'scale', kernel = 'rbf'")
st.caption("Read more about the parameters and SVC function here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html")
clf_svm = build_basic_svm(X_train_scaled, y_train)
st.subheader('Confusion Matrix and Metrics for Basic SVM Model')
accuracy = clf_svm.score(X_test_scaled, y_test)
y_pred = clf_svm.predict(X_test_scaled)
st.write("Accuracy: ", accuracy.round(2))
st.write("Precision: ", precision_score(y_test, y_pred, labels=['Malignant', 'Benign']).round(2))
st.write("Recall: ", recall_score(y_test, y_pred, labels=['Malignant', 'Benign']).round(2)) 
show_confusion_matrix(clf_svm, X_test_scaled, y_test)


# Use GridSearchCV to find the best parameters
c, gamma, kernel = find_best_params(X_train_scaled, y_train)

# Build the model with the optimal parameters
st.header('SVM Model with Optimal Parameters')
st.caption('Optimal Parameters: C = {}, gamma = {}, kernel = {}'.format(c, gamma, kernel))
clf_svm = build_svm(c, gamma, kernel, X_train_scaled, y_train)
st.subheader('Confusion Matrix and Metrics for SVM with Optimal Parameters')
y_pred = clf_svm.predict(X_test_scaled)
class_names = ['Malignant', 'Benign']
st.write("Accuracy: ", accuracy.round(2))
st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
show_confusion_matrix(clf_svm, X_test_scaled, y_test)
accuracy = clf_svm.score(X_test_scaled, y_test)

# Plot Scree Plot - PCA to reduce the number of featuress
st.header('Plotting Scree Plot - PCA to reduce the number of features')
#scree_plot = scree_plot(X_train_scaled) #returns scree plot
fig = scree_plot(X_train_scaled)
st.pyplot(fig)
# c, gamma, kernel, X_train_pca, X_test_pca = pca(X_train_scaled, X_test_scaled, y_train) #returns c, gamma, kernel, X_train_pca, X_test_pca
# st.caption('Optimal Parameters determined by PCA: C = {}, gamma = {}, kernel = {}'.format(c, gamma, kernel))

# # Build the model with the optimal parameters and the reduced number of features
# clf_svm_pca = build_svm(c, gamma, kernel, X_train_pca, y_train)
# accuracy = clf_svm.score(X_test_scaled, y_test)
# y_pred = clf_svm.predict(X_test_scaled)
# class_names = ['Malignant', 'Benign']
# st.write("Accuracy: ", accuracy.round(2))
# st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
# st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 