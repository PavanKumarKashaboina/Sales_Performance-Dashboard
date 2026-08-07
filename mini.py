import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, END
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Placeholder for TAO Classifier if using a custom implementation
class TaoTreeClassifier(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Initialize global variables
precision = []
recall = []
fscore = []
accuracy = []
categories = ['malicious', 'safe']  # Fixed naming
target_name = 'is_malicious'
model_folder = 'model'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# --- Define functions ---

def uploadDataset(): 
    global dataset
    filename = filedialog.askopenfilename(initialdir=".")
    if filename:
        text.delete('1.0', END)
        text.insert(END, filename + ' Loaded\n\n')
        dataset = pd.read_csv(filename)
        text.insert(END, str(dataset.head()) + "\n\n")

def Preprocess_Dataset():
    global dataset, X, y
    text.delete('1.0', END)
    
    dataset = dataset.dropna()
    text.insert(END, "Missing Values:\n" + str(dataset.isnull().sum()) + "\n\n")
    
    non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
    for col in non_numeric_columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
    
    y = dataset[target_name]
    X = dataset.drop(target_name, axis=1)

    sns.set(style="darkgrid")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=target_name, data=dataset, palette="Set3")
    plt.title("Class Distribution")
    plt.show()

def Train_Test_Splitting():
    global X, y, x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    text.delete('1.0', END)
    text.insert(END, f"Total records: {X.shape[0]}\n")
    text.insert(END, f"Training records: {x_train.shape[0]}\n")
    text.insert(END, f"Testing records: {x_test.shape[0]}\n")

def Calculate_Metrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END, f"{algorithm} Results:\n")
    text.insert(END, f"Accuracy: {a:.2f}%\nPrecision: {p:.2f}%\nRecall: {r:.2f}%\nF1-Score: {f:.2f}%\n")
    
    cr = classification_report(y_test, predict, target_names=categories)
    text.insert(END, "\nClassification Report:\n" + cr + "\n")

    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap="viridis")
    plt.title(f"{algorithm} Confusion Matrix")
    plt.show()

def existing_classifier():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)
    
    # --- 1. TAO Classifier ---
    tao_filename = os.path.join(model_folder, "TAO_weights.pkl")
    if os.path.exists(tao_filename):
        tao_model = joblib.load(tao_filename)
    else:
        # Initializing TAO (using DecisionTree as the engine)
        tao_model = TaoTreeClassifier(max_depth=10, random_state=42)
        tao_model.fit(x_train, y_train)
        joblib.dump(tao_model, tao_filename)

    y_pred_tao = tao_model.predict(x_test)
    Calculate_Metrics("Existing TAO", y_pred_tao, y_test)

    # --- 2. Ridge Classifier ---
    ridge_filename = os.path.join(model_folder, "Ridge_weights.pkl")
    if os.path.exists(ridge_filename):
        ridge_model = joblib.load(ridge_filename)
    else:
        ridge_model = RidgeClassifier(alpha=1.0)
        ridge_model.fit(x_train, y_train)
        joblib.dump(ridge_model, ridge_filename)

    y_pred_ridge = ridge_model.predict(x_test)
    Calculate_Metrics("Existing Ridge", y_pred_ridge, y_test)

def proposed_classifier():
    global x_train, y_train, x_test, y_test, mlmodel
    text.delete('1.0', END)

    # Defining Stacking estimators (Missing in original code)
    level0 = [('et', ExtraTreesClassifier(n_estimators=100))]
    level1 = RidgeClassifier()

    model_path = os.path.join(model_folder, "Stacking_weights.pkl")
    if os.path.exists(model_path):
        mlmodel = joblib.load(model_path)
    else:
        mlmodel = StackingClassifier(
        estimators=level0_models, 
        final_estimator=level1_meta_model,
        cv=5,
        passthrough=False 
    )
    
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_path)

    y_pred = mlmodel.predict(x_test)
    Calculate_Metrics("Proposed Stacking", y_pred, y_test)

def Predictions():
    global test_data, filename, pred
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n')
    test_data = pd.read_csv(filename)
    text.insert(END, str(test_data.head()) + "\n\n----------")

    LE=LabelEncoder()
    for i in test_data.columns:
        test_data[i]=LE.fit_transform(test_data[i])

    pred=mlmodel.predict(test_data)
    test_data['predictions'] = pred
    text.insert(END, str(test_data) + "\n\n")


def graph():
    if len(accuracy) < 2:
        messagebox.showwarning("Warning", "Run both classifiers first to compare!")
        return
        
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
        'Value': [accuracy[0], precision[0], recall[0], fscore[0], accuracy[1], precision[1], recall[1], fscore[1]],
        'Algorithm': ['Existing']*4 + ['Proposed']*4
    }
    df = pd.DataFrame(data)
    sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df)
    plt.title("Performance Comparison")
    plt.show()

def close():
    main.destroy()

# --- GUI Setup ---
main = tk.Tk()
main.title("INSIDER THREAT DETECTION SYSTEM")
main.state('zoomed') # Open full screen

title = tk.Label(main, text='INSIDER THREAT DETECTION SYSTEM', bg='gold2', fg='black', font=('times', 18, 'bold'), height=3)
title.pack(fill=tk.X)

# Button Frame
btn_frame = tk.Frame(main, bg='DarkSlateGray1')
btn_frame.place(x=20, y=100)

buttons = [
    ("Dataset", uploadDataset),
    ("Preprocessing", Preprocess_Dataset),
    ("Train Test Splitting", Train_Test_Splitting),
    ("Existing Classifier", existing_classifier),
    ("Proposed Classifier", proposed_classifier),
    ("Prediction", Predictions),
    ("Comparison Graph", graph),
    ("Exit", close)
]

for txt, cmd in buttons:
    tk.Button(btn_frame, text=txt, command=cmd, font=('times', 12, 'bold'), width=20).pack(pady=5)

# Text Area
text = tk.Text(main, height=30, width=100, font=('times', 12))
text.place(x=330, y=100)

main.config(bg='DarkSlateGray1')
main.mainloop()