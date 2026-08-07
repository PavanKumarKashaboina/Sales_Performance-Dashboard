from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# FIGS and TAO from imodels library as per the notebook
from imodels import FIGSClassifier, TaoTreeClassifier  
import os
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Global variables for metrics
accuracy = []
precision = []
recall = []
fscore = []
# Updated labels based on the notebook target: is_malicious (0: safe, 1: malicious)
categories = ['safe', 'malicious']
target_name = 'is_malicious'
model_folder = "model"

if not os.path.exists(model_folder):
    os.mkdir(model_folder)

def uploadDataset(): 
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head()) + "\n\n")

def Preprocess_Dataset():
    global dataset, X, y
    text.delete('1.0', END)
    
    # Handling nulls
    dataset = dataset.dropna()
    text.insert(END, "Null values check:\n" + str(dataset.isnull().sum()) + "\n\n")
    
    # Label Encoding categorical columns (Department, Campus, Position, Origin Country)
    le = LabelEncoder()
    for col in dataset.columns:
        if dataset[col].dtype == 'object' or dataset[col].dtype == 'str':
            dataset[col] = le.fit_transform(dataset[col])
    
    y = dataset[target_name]
    X = dataset.drop(target_name, axis=1)
    
    text.insert(END, "Dataset Preprocessed and Label Encoded successfully.\n")
    
    # Count Plot for class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_name, data=dataset)
    plt.title("Malicious vs Safe Distribution")
    plt.show()

def Train_Test_Splitting():
    global X, y, x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    text.delete('1.0', END)
    text.insert(END, "Total records: " + str(X.shape[0]) + "\n")
    text.insert(END, "Training records: " + str(x_train.shape[0]) + "\n")
    text.insert(END, "Testing records: " + str(x_test.shape[0]) + "\n")

def Calculate_Metrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END, f"{algorithm} Accuracy  : {a:.2f}%\n")
    text.insert(END, f"{algorithm} Precision : {p:.2f}%\n")
    text.insert(END, f"{algorithm} Recall    : {r:.2f}%\n")
    text.insert(END, f"{algorithm} F1-Score  : {f:.2f}%\n\n")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title(algorithm + " Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def FIGS_classifier():
    global x_train, y_train, x_test, y_test, mlmodel
    text.delete('1.0', END)
    text.insert(END, "Training FIGS Classifier...\n")
    
    model_path = os.path.join(model_folder, "FIGS_model.pkl")
    if os.path.exists(model_path):
        mlmodel = joblib.load(model_path)
    else:
        mlmodel = FIGSClassifier(max_rules=12) # Parameter example from imodels
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_path)
        
    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("FIGS Classifier", y_pred, y_test)

def TAO_classifier():
    global x_train, y_train, x_test, y_test, mlmodel
    text.delete('1.0', END)
    text.insert(END, "Training TAO Classifier...\n")
    
    model_path = os.path.join(model_folder, "TAO_model.pkl")
    if os.path.exists(model_path):
        mlmodel = joblib.load(model_path)
    else:
        mlmodel = TaoTreeClassifier()
        mlmodel.fit(x_train.values, y_train.values)
        joblib.dump(mlmodel, model_path)
        
    y_pred = mlmodel.predict(x_test.values)
    Calculate_Metrics("TAO Classifier", y_pred, y_test)

def Prediction():
    global mlmodel
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test_data = pd.read_csv(filename)
    
    # Preprocessing test data
    le = LabelEncoder()
    for col in test_data.columns:
        if test_data[col].dtype == 'object':
            test_data[col] = le.fit_transform(test_data[col])
            
    preds = mlmodel.predict(test_data.values)
    
    text.delete('1.0', END)
    text.insert(END, "Predictions for Loaded File:\n\n")
    for i, p in enumerate(preds):
        result = "Malicious" if p == 1 else "Safe"
        text.insert(END, f"Row {i+1}: Result -> {result}\n")

def graph():
    labels = ['FIGS', 'TAO']
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots()
    ax.bar(x - width*1.5, accuracy, width, label='Accuracy')
    ax.bar(x - width/2, precision, width, label='Precision')
    ax.bar(x + width/2, recall, width, label='Recall')
    ax.bar(x + width*1.5, fscore, width, label='F1-Score')

    ax.set_ylabel('Scores')
    ax.set_title('Algorithm Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

# GUI Setup
main = Tk()
main.title("Insider Threat Detection using FIGS & TAO")
main.geometry("1300x700")

title = Label(main, text='Insider Threat Detection (FIGS vs TAO)')
title.config(bg='midnight blue', fg='white', font=('times', 18, 'bold'), height=3, width=100)
title.place(x=0, y=5)

ff = ('times', 12, 'bold')

Button(main, text="Upload Dataset", command=uploadDataset, font=ff, width=20).place(x=20, y=100)
Button(main, text="Preprocessing", command=Preprocess_Dataset, font=ff, width=20).place(x=20, y=150)
Button(main, text="Train Test Split", command=Train_Test_Splitting, font=ff, width=20).place(x=20, y=200)
Button(main, text="FIGS Classifier", command=FIGS_classifier, font=ff, width=20).place(x=20, y=250)
Button(main, text="TAO Classifier", command=TAO_classifier, font=ff, width=20).place(x=20, y=300)
Button(main, text="Predict Malicious", command=Prediction, font=ff, width=20).place(x=20, y=350)
Button(main, text="Comparison Graph", command=graph, font=ff, width=20).place(x=20, y=400)
Button(main, text="Exit", command=main.destroy, font=ff, width=20).place(x=20, y=450)

text = Text(main, height=25, width=110, font=('times', 12))
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300, y=100)

main.config(bg='light slate gray')
main.mainloop()