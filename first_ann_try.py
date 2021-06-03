# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

def run_ann():
    # Load Dataset - Which gives data of a person with class label whether the customer will leave the bank or not
    dataset = pd.read_csv('Bank_Customer_Prediction.csv')

    # Differentiate class label y from entire dataset
    x = dataset.iloc[:,3:13].values
    y = dataset.iloc[:,13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    # Label Encoding for the dataset's categorical data (here geography column)
    labelEncoder_X_1 = LabelEncoder()
    x[:,1] = labelEncoder_X_1.fit_transform(x[:,1])

    # Label Encoding for the dataset's categorical data (here gender column)
    labelEncoder_X_2 = LabelEncoder()
    x[:,2] = labelEncoder_X_2.fit_transform(x[:,2])

    #One Hot Encoding for Geography column
    from sklearn.compose import ColumnTransformer
    col = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1])],remainder='passthrough')
    x = np.array(col.fit_transform(x))

    # Splitting data into training and testing set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

    # Normalizing data / scaling data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train  = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # building classifier blueprint.
    classifier = Sequential()

    # Input and First Hidden layer
    classifier.add(Dense(units = 6, activation = 'relu'))

    # Second Hidden layer
    classifier.add(Dense(units = 6, activation = 'relu'))

    # Third Hidden layer
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training ANN
    classifier.fit(x_train, y_train, batch_size = 32, epochs = 100)

    # testing using test data
    y_pred = classifier.predict(x_test)

    # Setting a threshold of 0.5
    y_pred = (y_pred>0.5)

    # Final Comparison
    final_comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

    # print(final_comparison)

    # building Confusion Matrix
    conf_mtx = confusion_matrix(y_test, y_pred)
    # print(conf_mtx)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Testing using a user provided new pattern
    # print(classifier.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

    return test_accuracy

