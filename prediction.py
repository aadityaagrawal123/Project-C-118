import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

train_data = pd.read_csv("./static/assets/datafiles/updated_product_dataset.csv")
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model/sentiment_analysis_model.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


# dictionary where key : emotion , value : list
encode_emotions = {
                    "Neutral": [0,"./static/assets/emoticons/neutral.png"],
                    "Positive": [1,"./static/assets/emoticons/positive.png"],
                    "Negative": [2,"./static/assets/emoticons/negative.png"]
                    }


def predict(text):

    sentiment = ""
    emoji_url = ""
    customer_review = []
    customer_review.append(text)
    sequences = tokenizer.texts_to_sequences(customer_review)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = model.predict(padded)
    label = np.argmax(result , axis=1)
    label = int(label)

    # extracting emotion and url from dictionary
    for emotion in encode_emotions:
        if encode_emotions[emotion][0]  ==  label:
            sentiment = emotion
            emoji_url = encode_emotions[emotion][1]

    return sentiment , emoji_url

def show_review():
    review_list = pd.read_csv("./static/assets/data_files/data_entry.csv")
    
    date1 = (review_list['date'].values[0])
    date2 =(review_list['date'].values[1])
    date3 = (review_list['date'].values[2])

    product1 = review_list['product'].values[0]
    product2 = review_list['product'].values[1]
    product3 = review_list['product'].values[2]

    review1 = review_list['review'].values[0]
    review2 = review_list['review'].values[1]
    review3 = review_list['review'].values[2]

    sentiment1 = review_list["sentiment"].values[0]
    sentiment2 = review_list["sentiment"].values[1]
    sentiment3 = review_list["sentiment"].values[2]

    return [
        {
            "date": date1,
            "product": product1,
            "review": review1,
            "sentiment": sentiment1
        },
        {
            "date": date2,
            "product": product2,
            "review": review2,
            "sentiment": sentiment2
        },
        {
            "date": date3,
            "product": product3,
            "review": review3,
            "sentiment": sentiment3
        }
    ]