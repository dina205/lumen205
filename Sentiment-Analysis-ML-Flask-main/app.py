from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_opinion_target(text):
    stop_words = set(stopwords.words('english'))
    opinion_words = []
    target_words = []

    # tokenize text
    tokens = word_tokenize(text)
    # get POS tags
    pos_tags = nltk.pos_tag(tokens)

    # extract opinion words and target words
    for word, tag in pos_tags:
        # opinion words are adjectives or adverbs
        if tag.startswith('JJ') or tag.startswith('RB'):
            # ignore stop words and punctuation
            if word.lower() not in stop_words and word.lower() not in punctuation:
                opinion_words.append(word.lower())
        # target words are nouns
        elif tag.startswith('NN'):
            # ignore stop words and punctuation
            if word.lower() not in stop_words and word.lower() not in punctuation:
                target_words.append(word.lower())

    return opinion_words, target_words

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')

    # convert to lowercase
    text1 = request.form['text1'].lower()

    text_final = ''.join(c for c in text1 if not c.isdigit())

    # remove punctuations
    # text3 = ''.join(c for c in text2 if c not in punctuation)

    # remove stopwords
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound']) / 2, 2)

    # extract opinion words and target words
    opinion_words, target_words = extract_opinion_target(processed_doc1)

    return render_template('form.html', final=compound, text1=text_final, text2=dd['pos'], text5=dd['neg'],
                           text4=compound, text3=dd['neu'], opinion_words=opinion_words, target_words=target_words)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
