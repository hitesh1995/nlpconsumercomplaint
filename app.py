from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.svm import LinearSVC
import joblib



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    pipeline = joblib.load('consumercomplaint.joblib')
    labels = ['Debt collection', 'Mortgage','Credit reporting', 'Credit card','Bank account or service','Consumer Loan','Student loan','Prepaid card','Payday loan','Money transfers','Other financial service']
    my_file = open("complaintlog.csv", "a")
    if request.method == 'POST':
        message = request.form['message']
        msg = request.form['name']
        data = [message]
        name = [msg]
        prediction = pipeline.predict(data)
        my_file.write("\n{},".format((name)))
        my_file.write("{},".format(str(message)))
        my_file.write("{},".format(str(labels[int(prediction)])))
        my_file.close()

        return render_template('predict.html',prediction = prediction)

    



if __name__ == '__main__':  
    app.run(debug=True)
