from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

NOFISH = -1
BREAM = 0
PARKKI = 1
PERCH = 2
PIKE = 3
ROACH = 4
SMELT = 5
WHITEFISH = 6
prediction = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():    
    if request.method == "POST":
        #get form data
        length1 = request.form.get('length1')
        length2 = request.form.get('length2')
        length3 = request.form.get('length3')
        height = request.form.get('height')
        width = request.form.get('width')
        fishname = request.form.get('fish_name')
        
        try:
            prediction = predictTheWeight(length1, length2, length3, height, width, fishname)
            #pass prediction to template
            return render_template('index.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"

def predictTheWeight(length1, length2, length3, height, width, fishname):
    fish_array = [0, 0, 0, 0, 0, 0, 0, 0]
    if int(fishname) == NOFISH:
        pass
    else:
        fish_array[int(fishname)] = 1

    #keep all inputs in array
    test_data = [length1, length2, length3, height, width, *fish_array]
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    #open file
    file = open("static/model_saved.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    
if __name__ == '__main__':
    app.run(debug=True)