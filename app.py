import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_names = ["first hackathon", "not first hackathon"]
st.set_option('deprecation.showPyplotGlobalUse', False)
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
data = pd.read_csv("data.csv")

# Create sidebar
st.sidebar.title("KNN Comparison")
k = st.sidebar.slider("Choose value of K", 1, 48)

# Calculate accuracy
accuracy = knn_comparison(data, k)

st.write(accuracy)



