#streamlit run auto_ml_new.py
#conda create --name auto_ml_38 python=3.8
#conda activate auto_ml_38
#conda install scikit-learn pandas numpy matplotlib tqdm joblib xgboost lightgbm streamlit numba
#conda install pycaret=3.2.0
#conda install -n automl ipykernel
#pip install ydata_profiling
#pip install joblib==1.3.2
#pip install streamlit-pandas-profiling
#pip install pydantic-settings
#pip install --upgrade pandas

import streamlit as st
import pandas as pd
#import pandas_profiling

import pycaret.classification as cls 
import pycaret.regression as reg
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image('https://qtxasset.com/cdn-cgi/image/w=850,h=478,f=auto,fit=crop,g=0.5x0.5/https://qtxasset.com/quartz/qcloud4/media/image/fiercevideo/1554925532/googlecloud.jpg?VersionId=hJC0G.4VGlXbcc35EzyI9RhCJI.mslxN')
    st.title("AutoML_with_pycaret")
    choice = st.radio("Navigation", 
                      ["Upload","Profiling","Data_Preprocessing", "Modelling", 
                       "Evaluation","Download"])
    st.info("This project application helps you build and explore your Machine Learning Model.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df, title="Data Report", explorative=True)
    st_profile_report(profile_df)
    
    #profile_df = df.profile_report()
    #st_profile_report(profile_df)

if choice == "Data_Preprocessing": 
    st.title("Data_Preprocessing")

    st.subheader("Drop & Duplicate")
    
    drop_options = df.columns
    selection = st.pills("삭제할 열을 선택하세요.(중복선택가능)", 
                         drop_options, selection_mode="multi")
    st.markdown(f"삭제하는 열: {selection}.")
    df = df.drop(selection, axis=1)

    duplicate_options = st.radio("중복되는 행 삭제 유무 결정", ("delete", "keep"))
    if duplicate_options == "delete": 
        df = df.drop_duplicates()
    else: 
        pass

    st.subheader("Select the Target Column")

    chosen_target = st.selectbox('Choose the Target Column : ', df.columns)

    y = df[chosen_target]
    X = df.drop(chosen_target, axis=1)

    test_size = st.slider("Select test size(%)?", 0.00, 1.00, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)

    st.subheader("Train and Test Data")
    st.info("Train Data Length: " + str(len(X_train)))
    
    st.write("X_train")
    X_train = X_train.reset_index(drop=True)
    st.dataframe(X_train.head())

    st.write("y_train")
    y_train = y_train.reset_index(drop=True)
    st.dataframe(y_train.head())

    st.info("Test Data Length: " + str(len(X_test)))

    st.write("X_test")
    X_test = X_test.reset_index(drop=True)
    st.dataframe(X_test.head())

    st.write("y_test")
    y_test = y_test.reset_index(drop=True)
    st.dataframe(y_test.head())
    
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test 
    st.session_state.y_test = y_test
    
if choice == "Modelling": 
    mdl = st.selectbox('Chose Modelling Type : ',['Classification','Regression'])
    
    if st.button('Run Modelling'):
        
        if mdl == 'Classification':
            st.session_state.Model = cls
            #from pycaret.classification import setup, compare_models, pull, save_model 

        
        elif mdl == 'Regression':
            st.session_state.Model = reg
            #from pycaret.regression import setup, compare_models, pull, save_model

        st.info("Your model of choice is " + mdl) 

        st.session_state.Model.setup(st.session_state.X_train, 
                                     target=st.session_state.y_train)
        setup_df = st.session_state.Model.pull()
        st.dataframe(setup_df)
        best_model = st.session_state.Model.compare_models()
        compare_df = st.session_state.Model.pull()
        st.dataframe(compare_df)
        st.session_state.best_model = best_model
        st.session_state.Model.save_model(best_model, 'best_model')

if choice == "Evaluation":
    y_pred_df = st.session_state.Model.predict_model(st.session_state.best_model, 
                                                  data=st.session_state.X_test)
    y_pred_df['y_true'] = st.session_state.y_test

    st.subheader("Predictions and True Values")
    st.info("Test Data Length: " + str(len(y_pred_df)))
    st.dataframe(y_pred_df)

    st.session_state.y_pred = y_pred_df["prediction_label"]

    # classification_report를 딕셔너리 형태로 변환 후 데이터프레임 생성
    report_dict = classification_report(st.session_state.y_test, st.session_state.y_pred,
                                        output_dict=True)
    df = pd.DataFrame(report_dict).transpose()

    # Streamlit에서 출력
    st.subheader("Classification Report")
    st.dataframe(df)  # 인터랙티브 테이블 형태로 출력
        
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

