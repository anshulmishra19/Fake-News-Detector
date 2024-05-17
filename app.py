import streamlit as st
import pickle
from PIL import Image

# Load the trained model and TF-IDF vectorizer
pac = pickle.load(open('model_fakenews.pickle', 'rb'))
tfidf_vectorizer = pickle.load(open('tfid.pickle', 'rb'))


# Create the Streamlit app
st.title('Fake News Detection')

# Get the news article text from the user
news_article = st.text_area('Enter the news article:')

# Preprocess the news article text
input_data = [news_article.rstrip()]
tfidf_test = tfidf_vectorizer.transform(input_data)

# Predict the authenticity of the news article
y_pred = pac.predict(tfidf_test)
print(y_pred)
# Display the prediction result

gif1 = Image.open("static\success.gif")
gif2 = Image.open("static/fake.jpg")
button = st.button('Predict')
if button:
    if y_pred == 'REAL':
        st.success('The news article is real.')
        st.image(gif1, width=300)
    elif y_pred == 'FAKE':
        st.error('The news article is fake.')
        st.image(gif2, width=250)
    