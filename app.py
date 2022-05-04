from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv('news.csv')
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7)
t_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
#this will stop the english words
#words which occur more than 70 percent of the artices will be discarded

t_train = t_vectorizer.fit_transform(x_train.astype(str))

t_test = t_vectorizer.transform(x_test.astype(str))

loaded_model = pickle.load(open('model.pkl', 'rb'))

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = t_vectorizer.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
