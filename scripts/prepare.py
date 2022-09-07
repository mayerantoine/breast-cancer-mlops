
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():

    parser = argparse.ArgumentParser("prepare")

    parser.add_argument("--input_data",type=str)
    parser.add_argument("--train",type=str)
    parser.add_argument("--test",type=str)

    args = parser.parse_args()

    print("train args:",args.train)
    
    run = Run.get_context()
    ws = run.experiment.workspace
    ds_tr = ws.get_default_datastore()


    df = run.input_datasets['raw_data'].to_pandas_dataframe()

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

    train = np.column_stack([x_train,y_train])
    test = np.column_stack([x_test,y_test])
   
    # Write the model to file.
    train_path = "./data/train/"
    test_path = "./data/test/"

    os.makedirs(args.train, exist_ok=True)
    os.makedirs(args.test, exist_ok=True)
    print("Saving the split")

    np.savetxt(os.path.join(args.train,"train.csv"), train, delimiter=",")
    np.savetxt(os.path.join(args.test,"test.csv"), train, delimiter=",")
  

 
if __name__ =='__main__':
    main()


