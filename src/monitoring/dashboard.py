import mlflow.pyfunc
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

_ = load_dotenv()

FULL_PATH = os.environ.get('FULL_PATH')
MODEL_PATH = "models/mlartifacts/358795740332304726/397750d2b5594f73810739dd50706918/artifacts/model"
DATASET_PATH = "datasets/churn_model_filtered.csv"

MODEL = os.path.join(FULL_PATH, MODEL_PATH)
DATASET = os.path.join(FULL_PATH, DATASET_PATH)

@st.cache_resource 
def load_model():
    return mlflow.pyfunc.load_model(MODEL)

def model_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1score = f1_score(y, y_pred, average="weighted")
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred)

    return accuracy, f1score, precision, recall, cm

def bar_metrics_plot(df_metrics: pd.DataFrame):
    st.subheader('Metrics')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_metrics['Metrics'],
        y=df_metrics['Score'],
        name='Metrics',
        marker_color='green'
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def conf_matrix_test_plot(cm_test):
    st.subheader('Confusion Matrix - Test Side')
    fig = px.imshow(cm_test)
    
    st.plotly_chart(fig)


def conf_matrix_train_plot(cm_train):
    st.subheader('Confusion Matrix - Train Side')
    fig = px.imshow(cm_train)
    
    st.plotly_chart(fig)


def trp_fpr(cm_test):
    tp,  fn , fp, tn = cm_test[0, 0], cm_test[0, 1], cm_test[1, 0], cm_test[1, 1]

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    df_tpr_fpr = pd.DataFrame({'Type': ['TPR', 'FPR'],
                               'Rate': [tpr, fpr]})

    st.subheader('True Positive Rate x False Positive Rate')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_tpr_fpr['Type'],
        y=df_tpr_fpr['Rate'],
        name='TPR x FPR',
        marker_color='blue'
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def class_report_test(y_test, y_pred_test):
    st.subheader('Classification Report - Test Side')
    classif_report = classification_report(y_test ,y_pred_test, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)


def class_report_train(y_train, y_pred_train):
    st.subheader('Classification Report - Train Side')
    classif_report = classification_report(y_train ,y_pred_train, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)

def main():
    df = pd.read_csv(DATASET)

    X = df.drop(columns=['Customer ID', 'Customer Status'])
    y = df['Customer Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = load_model()

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracy_test, f1score_test, precision_test, recall_test, cm_test = model_metrics(y_test, y_pred_test)

    accuracy_train, f1score_train, precision_train, recall_train, cm_train = model_metrics(y_train, y_pred_train)

    df_metrics = pd.DataFrame({'Metrics': ['Accuracy Score Test', 'F1 Score Test',
                                        'Precision Score Test', 'Recall Score Test',
                                        'Accuracy Score Train', 'F1 Score Train',
                                        'Precision Score Train', 'Recall Score Train'],
                            'Score':   [accuracy_test, f1score_test, precision_test, 
                                        recall_test, accuracy_train, f1score_train, 
                                        precision_train, recall_train]})

    st.title('Metrics Monitor')

    st.sidebar.title('Choose here your visualization:')
    option = st.sidebar.selectbox(
        'Choose Visualization:',
        ['Metrics', 'Confusion Matrix - Test', 'Confusion Matrix - Train',
        'True Positive Rate x False Positive Rate', 'Classification Report - Test',
        'Classification Report - Train']
    )

    match option:
        case 'Metrics':
            bar_metrics_plot(df_metrics)
        case 'Confusion Matrix - Test':
            conf_matrix_test_plot(cm_test)
        case 'Confusion Matrix - Train':
            conf_matrix_train_plot(cm_train)
        case 'True Positive Rate x False Positive Rate':
            trp_fpr(cm_test)
        case 'Classification Report - Test':
            class_report_test(y_test, y_pred_test)
        case 'Classification Report - Train':
            class_report_train(y_train, y_pred_train)
        case _:
            st.error("Opção inválida!")

if __name__ == "__main__":
    main()