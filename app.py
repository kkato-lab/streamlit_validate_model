import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


# Streamlit
st.title("validate model")

st.header("upload File")


input_manual_raw = st.file_uploader("手入力データ", type='csv')
input_ai_raw = st.file_uploader("推論データ", type='csv')

if input_manual_raw is not None:
    input_manual = pd.read_csv(input_manual_raw).fillna(0)
if input_ai_raw is not None:
    input_ai = pd.read_csv(input_ai_raw)


def calculateDf(input_manual, input_ai):
    output_list = []
    output_col_list = []
    filterd = []

    for col in input_manual:
        df = pd.DataFrame({'threshold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []})

        for i in range(1, 101):
            threshold=0.01*i
            df_predict_filterd = input_ai.where(input_ai>threshold, 0).where(input_ai<threshold,1)

            filterd.append(df_predict_filterd)
            accuracy = accuracy_score(input_manual[col], df_predict_filterd[col])
            precision = precision_score(input_manual[col], df_predict_filterd[col])
            recall = recall_score(input_manual[col], df_predict_filterd[col])
            f1 = f1_score(input_manual[col], df_predict_filterd[col])

            df = pd.concat([df, pd.DataFrame([{'threshold': threshold, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1':f1 }])], ignore_index=True)

        output_list.append(df)
        output_col_list.append(col)
    return output_list, output_col_list

def printResultDf(output_list, input_manual, input_ai):
    cols = ["accuracy", "precision", "recall", "f1", "train_count", "AUC"]

    result_df = pd.DataFrame(columns=cols)

    cols = ["accuracy", "precision", "recall", "f1", "train_count", "AUC"]
    for i in range(len(output_list)):
        fpr, tpr, thresholds = roc_curve(input_manual.iloc[:, i], input_ai.iloc[:, i])
        auc_value = auc(fpr, tpr)

        row = pd.Series([output_list[i].max()[1], output_list[i].max()[2],output_list[i].max()[3],output_list[i].max()[4],int(input_manual.iloc[:, i].sum()), auc_value], index=cols)
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    return result_df

if st.button(label='Start!'):
    if input_manual_raw is None:
        st.warning('手入力データをアップロードしてください。')
    if input_ai_raw is None:
        st.warning('推論データをアップロードしてください。')
    if (input_manual_raw is not None and input_ai_raw is not None):
        output_list, output_col_list = calculateDf(input_manual, input_ai)
        st.dataframe(printResultDf(output_list, input_manual, input_ai))
        

