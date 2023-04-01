import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("Predicting Well Facies")

# Load data
data=pd.read_csv("data.csv")
X_cols = ['Experience','Languages','BEANS?','Glasses','Github','Language','Age']
y_col = 'label'

X=[]
y=[]
def paramChange():
  for i in 
  X = data[selected_cols]
  y = data[y_col]
for i in X_cols:
  st.checkbox(i, value=False, key=None, help=None, on_change=paramChange, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
# User selection of input factors



# Define KNN pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(multi_class='auto', solver='liblinear', 
                                            max_iter=1000, random_state=42))])

# Define stratified sampling CV 
cv = StratifiedKFold(5, shuffle=True)
# Cross-validation
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
#st.write("Cross-validation accuracy scores:", scores)
st.write("Average cross-validation accuracy:", np.mean(scores))

# Fit model on full data
pipe.fit(X, y)

# Plot predicted and true facies
y_pred = pipe.predict(X)
facies_list = np.unique(y)
logs = X.columns

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(y, data.DEPTH, color='k', lw=0.5)
F = np.vstack((y_pred,y_pred)).T
ax.imshow(F, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='viridis', alpha=0.4)

ax.set_xlabel('Facies')
ax.set_ylabel('Depth (m)')
ax.set_title('Well Facies Prediction')

sdata['AILP'] = butter_lowpass_filter(sdata.AI.values, 4, 1000/4, order=5) 
sdata['AIRLP'] = butter_lowpass_filter(sdata.AIR.values, 4, 1000/4, order=5) 
sdata['RHOBLP'] = butter_lowpass_filter(sdata.RHOB.values, 4, 1000/4, order=5) 
sdata['GRLP'] = butter_lowpass_filter(sdata.GR.values, 4, 1000/4, order=5) 
sdata['PHIELP'] = butter_lowpass_filter(sdata.PHIE.values, 4, 1000/4, order=5) 

# Load data
data = pd.read_csv("regdata.csv")
X_train = sdata[['RHOBLP', 'AILP', 'GRLP', 'PHIELP', 'AIRLP']]
y_train = data['label']

# Define stratified sampling CV 
cv = StratifiedKFold(5, shuffle=True)

# Define models
models = {'Random Forest': RandomForestClassifier(),
          'SVM': svm.SVC(),
          'Naive Bayes': GaussianNB(),
          'Decision Tree': tree.DecisionTreeClassifier(),
          'KNN': KNeighborsClassifier(n_neighbors=4)}

# Streamlit App
st.write("Select the model to use")

# User selection of model
selected_model = st.selectbox("Select model to use", list(models.keys()))
if(selected_model == 'KNN'):
    st.success('sheesh this model is fire')

# Define pipeline
pipe = make_pipeline(StandardScaler(), models[selected_model])

# Cross-validation
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
mean_cv_scores = np.mean(cv_scores)
st.write(f"Cross-validation accuracy scores for {selected_model}:", cv_scores)
st.write(f"Average cross-validation accuracy for {selected_model}:", mean_cv_scores)

# Fit model to training data
pipe.fit(X_train, y_train)

# Predict facies on training data
y_pred = pipe.predict(X_train)

# Plot predicted and true facies
facies_list = np.unique(y_train)
logs = X_train.columns

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(y_train, data.DEPTH, color='k', lw=0.5)
F_true = np.vstack((y_train,y_train)).T
F_pred = np.vstack((y_pred,y_pred)).T
ax.imshow(F_true, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='viridis', alpha=0.4)
ax.imshow(F_pred, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='plasma', alpha=0.4)

ax.set_xlabel('Facies')
ax.set_ylabel('Depth (m)')
ax.set_title(f'{selected_model} Facies Prediction')

# Display plot
st.pyplot(fig)
