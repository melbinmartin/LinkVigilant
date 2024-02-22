from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()

# Load whitelist and blacklist
with open('whitelist.txt', 'r') as f:
    whitelist = [line.strip() for line in f]

with open('blacklist.txt', 'r') as f:
    blacklist = [line.strip() for line in f]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        
        # Check if URL is in blacklist
        if url in blacklist:
            return render_template('index.html', unsafe=True, url=url)
        
        # Check if URL is in whitelist
        if url in whitelist:
            return render_template('index.html', xx=1, url=url)

        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred = gbc.predict(x)[0]
        # 1 is safe, -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        
        # If URL is predicted as unsafe and not in blacklist, add it to blacklist
        if y_pred == -1 and url not in blacklist:
            with open('blacklist.txt', 'a') as f:
                f.write(url + '\n')
        
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html', xx=round(y_pro_non_phishing,2), url=url)
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
