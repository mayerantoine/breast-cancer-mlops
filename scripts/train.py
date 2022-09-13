
import argparse
import os
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run, Dataset
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder",type=str,default='./data')

    args = parser.parse_args()
    folder = args.data_folder

    run = Run.get_context()
    ws = run.experiment.workspace
    ds_tr = ws.get_default_datastore()
    ds = Dataset.Tabular.from_delimited_files(path=ds_tr.path('cancer_data/cancer_data.csv'))


    #df = pd.read_csv(os.path.join(folder,'cancer_data.csv'))
    df = ds.to_pandas_dataframe()
    y = df['diagnosis'].astype('category')
    X = df.drop('diagnosis',axis=1)

    lbl_encoder = LabelEncoder()
    y_encode = lbl_encoder.fit_transform(y)

    print("cols:",X.columns)
    print("X shape", X.shape)
    print("encoder:", lbl_encoder.classes_)
    print("y encode:", y_encode.shape)

    x_train,x_test,y_train,y_test = train_test_split(X,y_encode,train_size=0.75,random_state=42,stratify =y_encode)

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)

    rf = RandomForestClassifier(n_estimators=40,max_depth=100,max_features=None,min_samples_leaf=3)
    rf.fit(x_train,y_train)

    accuracy = accuracy_score(y_test,rf.predict(x_test))
    run.log("accuracy",accuracy)

    f1 = f1_score(y_test,rf.predict(x_test))
    run.log("f1_score",f1)


    # Write the model to file.
    model_path = "./outputs/cancer_model.pkl"
    os.makedirs("outputs", exist_ok=True)
    print('Saving the model to {}'.format(model_path))
    joblib.dump(rf, model_path)

if __name__ == '__main__':
    main()
