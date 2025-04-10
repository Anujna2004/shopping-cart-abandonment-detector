from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import pickle

# Load the dataset
data = pd.read_csv("cart_data.csv")

# Define features and target variable
X = data[['time_on_site', 'pages_viewed', 'cart_value']]
y = data['abandoned']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))