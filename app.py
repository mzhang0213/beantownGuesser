import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_names = ["first hackathon", "not first hackathon"]
#st.title("Predicting Hackathon Experience")
st.title("Have you ever been to a Hackathon before?")
st.write("\n\n\nWe asked fellow beantown bashers eight questions to try and predict whether or not they have been to a Hackathon before based on the information they gave us.\n")
st.title("Here are the results!\n\n\n\n\n\n")
st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache(persist= True)
def load():
    data= pd.read_csv("data.csv")
    data = data.drop("Name", axis = 1)
    return data
df = load()
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show dataset")
    st.write(df)
models = {'Random Forest': RandomForestClassifier(),
      'SVM': svm.SVC(),
      'Naive Bayes': GaussianNB(),
      'Decision Tree': tree.DecisionTreeClassifier(),
      'KNN': KNeighborsClassifier(n_neighbors=1)}

# Streamlit App
st.write("Select the model to use")
selected_model = st.sidebar.selectbox("Select model to use", list(models.keys()))
data= pd.read_csv("data.csv")

y = data["Hackathon"]
X = data[["Age", "Exp", "Langs", "Beans", "Glasses", "Git", "Lang", "Start"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=1
)
clf = KNeighborsClassifier(n_neighbors=1)
pipe = make_pipeline(StandardScaler(), models[selected_model])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
st.pyplot()
st.write(accuracy)
# User selection of model
def knn_comparison(data, k):
    y = data["Hackathon"]
    X = data[["Age", "Exp", "Langs", "Beans", "Glasses", "Git", "Lang", "Start"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )
    clf = KNeighborsClassifier(n_neighbors=k)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    st.pyplot()
    return accuracy
    
# Load data

# Create sidebar
st.sidebar.title("KNN Comparison")
k = st.sidebar.slider("Choose value of K", 1, 48)

# Calculate accuracy
accuracy = knn_comparison(data, k)

st.write(accuracy)

st.subheader("Conclusions")
st.write("When selecting our questions, we tried to pick questions that would have correlation with experience. The most influencial question in the predictor was \"How many languagues do you know?\". We expected some correlation between all correlations, but we found that NONE of the questions had visible correlation. This shows that hackathons are events in which people of all experiences and ages participate in. Our biggest takeaway is that more people should participate, regardless of skill or age.") 


