from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time_on_site = float(request.form['time'])
    pages_viewed = int(request.form['pages'])
    cart_value = float(request.form['value'])

    features = np.array([[time_on_site, pages_viewed, cart_value]])
    prediction = model.predict(features)[0]

    return render_template('index.html', result="Likely to Abandon" if prediction else "Likely to Checkout")

if __name__ == "__main__":
    app.run(debug=True)