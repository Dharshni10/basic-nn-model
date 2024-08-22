# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Create a neural network regression model to accurately predict a continuous target variable based on input features from the provided dataset. The model should be designed to capture complex relationships within the data, leading to precise predictions. It will be trained, validated, and tested to ensure strong generalization to unseen data, with a focus on optimizing performance metrics like mean squared error or mean absolute error. The goal is to derive meaningful insights from the data, enhancing decision-making and understanding of the target variable's behavior.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Dharshni V M
### Register Number: 212223240029
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl_ex1').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
    ])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y=y_train,epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[4]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![Dataset](https://github.com/user-attachments/assets/9f2d3c71-22de-45bd-ac7c-7d3d49502608)

## OUTPUT

### Training Loss Vs Iteration Plot

![output](https://github.com/user-attachments/assets/8c849e1e-0d65-429d-b03b-f7e25a7b01f7)

### Test Data Root Mean Squared Error

![Output](https://github.com/user-attachments/assets/380f0193-f663-4502-b3b0-739d44ce98f4)

### New Sample Data Prediction

![Output](https://github.com/user-attachments/assets/70e7a8c5-3557-443d-8f8a-62ae1a6440cc)

## RESULT

Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
