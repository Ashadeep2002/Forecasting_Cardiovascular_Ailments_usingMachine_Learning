# Forecasting_Cardiovascular_Ailments_usingMachine_Learning
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale # scale and center data

from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.decomposition import PCA # to perform PCA to plot the data

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
from google.colab import drive
drive.mount('/content/MyDrive/', force_remount=True)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

d_df = pd.read_csv("./MyDrive/MyDrive/Heart_Disease_Prediction/heart_disease.csv")
# OR - pd.read_csv(data_path+"deliveries.csv"), where data_path = "../input/"
# reading deliveries dataset
''''score_df'''
m_df = pd.read_csv("MyDrive/MyDrive/Heart_Disease_Prediction/heart_disease.csv")
# OR - pd.read_csv(data_path+"matches.csv"), where data_path = "../input/"
# reading matches dataset
# csv- Comma seperated values
d_df.head()

m_df.head()

d_df.info()

m_df.info()

d_df.describe()

m_df.describe()
# Load the heart disease dataset
data = pd.read_csv("./MyDrive/MyDrive/Heart_Disease_Prediction/heart_disease.csv")

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame
data = pd.DataFrame({'sex': ['M', 'F', 'M', 'F', 'M'],
                   'target': [1, 0, 1, 0, 1]})

# Group the data by sex and target and count the occurrences
grouped = data.groupby(['sex', 'target']).size().reset_index(name='count')

# Create a crosstab of sex and target
crosstab = pd.crosstab(grouped['sex'], grouped['target'])

# Create a heatmap
sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Target')
plt.ylabel('Sex')
plt.title('Heart Disease Cases by Sex')
plt.show()

d_df['target'].count()

sns.countplot(x='target',data=m_df)

import matplotlib.pyplot as plt

# Group the data by sex and target
grouped = data.groupby(['sex', 'target']).size().reset_index(name='count')

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(grouped['sex'].astype(str) + '-' + grouped['target'].astype(str), grouped['count'])
ax.set_xlabel('Sex-Target')
ax.set_ylabel('Count')
ax.set_title('Heart Disease Cases by Sex')
plt.show()

m_df.hist(figsize=(14,14) , color = 'lightblue')
plt.show()
