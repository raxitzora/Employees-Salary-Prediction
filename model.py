import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df=pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Employee_Salary_Dataset.csv")
df.head()

##### All the Data work#####
 df.isnull().sum()
 df["ID"].unique()
 df["Experience_Years"].unique()
 df["Age"].unique()
 df["Gender"].unique()
 df["Salary"].unique()
 df.describe()
 df.info()

#drop columns
df=df.drop(columns=["Gender"])
df=df.drop(columns=["ID"])

#cleaned data
df.head()

#independent and dependent variable
X=df[["Experience_Years","Age"]]
y=df["Salary"]

#train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE",mse)
print("r2score :",r2)

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salaries")
plt.ylabel("Predicted Salaries")
plt.title("Actual vs Predicted Salaries")
plt.show()  

def predict_salary(age, experience):
    # Create a new DataFrame with the input values
    new_data = pd.DataFrame([[experience, age]], columns=['Experience_Years', 'Age'])
     # Ensure the columns are in the same order as the training data
    new_data = new_data[X.columns]
    predicted_salary = model.predict(new_data)
    return predicted_salary[0]

    # Enter your age and experience to predict salary
new_age=12
new_experience=2

# Predict salary based on user input
predicted_salary = predict_salary(new_age, new_experience)
print(f"Predicted Salary for Age {new_age} and Experience {new_experience} years: {predicted_salary}")



##Streamlit App
st.title("Employee Salary Prediction")

# Display evaluation metrics
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# User input
age = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
experience = st.number_input("Enter Experience in Years", min_value=0, max_value=50, value=5)

# Predict button
if st.button("Predict Salary"):
    predicted_salary = predict_salary(age, experience)
    st.write(f"Predicted Salary for Age {age} and Experience {experience} years: ₹{predicted_salary:.2f}")


# Plot the results
st.subheader("Actual vs Predicted Salaries")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salaries")
plt.ylabel("Predicted Salaries")
plt.title("Actual vs Predicted Salaries")
st.pyplot(plt)
