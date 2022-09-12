

import argparse
import os
import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    parser = argparse.ArgumentParser("train")
    
    parser.add_argument("--train", type=str, help="train data")
    parser.add_argument("--test", type=str, help="test data")
    parser.add_argument("--model_file", type=str, help="model file")
    parser.add_argument("--model_name",type=str,help="model name",default='cancer_model.pkl')
    
    args = parser.parse_args()
    
    run = Run.get_context()
    ws = run.experiment.workspace
    ds_tr = ws.get_default_datastore()

    print(args.train)
    print(args.test)
    print(args.model_name)

    train = pd.read_csv(args.train+"/train.csv")
    test = pd.read_csv(args.test+"/test.csv")

    y_train = train.iloc[:,-1]
    train.drop(columns = train.columns[-1],axis=1,inplace=True)
    x_train = train

    y_test = test.iloc[:,-1]
    test.drop(columns = test.columns[-1],axis=1,inplace=True)
    x_test = test

    lbl_encoder = LabelEncoder()
    y_encode = lbl_encoder.fit_transform(y_train)

    print("cols:",x_train.columns)
    print("X shape", x_train.shape)
    print("encoder:", lbl_encoder.classes_)
    print("y encode:", y_encode.shape)


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
    # model_path = "./outputs/cancer_model.pkl"
    os.makedirs(args.model_file, exist_ok=True)
    joblib.dump(rf, args.model_file+f"/{args.model_name}")

    print('Saving the model to {}'.format(args.model_file+f"/{args.model_name}"))

    run.complete()
    

if __name__ == '__main__':
    main()
