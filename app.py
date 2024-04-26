import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# Load the phishing model and CountVectorizer
loaded_model = pickle.load(open('phishing_model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Function to preprocess input text
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stemmer = SnowballStemmer("english")
    tokens = tokenizer.tokenize(text)
    stems = [stemmer.stem(token) for token in tokens]
    return ' '.join(stems)

# Header image
header_image = 'https://ideas.ted.com/wp-content/uploads/sites/3/2020/01/final_featured_art_phishing_istock.jpg'
st.image(header_image, use_column_width=True)

# Title
st.title('Phishing Website Detector')

# Text input for user to enter URL
url = st.text_input('Enter the URL to check for phishing:', value='')

# Button to trigger prediction
if st.button('Check for Phishing'):
    if url:
        # Preprocess the input URL
        preprocessed_url = preprocess_text(url)

        # Vectorize the preprocessed URL
        vectorized_url = cv.transform([preprocessed_url])

        # Predict whether the URL is phishing or not
        prediction = loaded_model.predict(vectorized_url)

        # Display the prediction
        if prediction[0] == 'good':
            st.success('The URL is likely safe.')
        else:
            st.error('The URL is potentially phishing.')

        # Display the probability of the prediction
        st.write('Phishing Probability:', loaded_model.predict_proba(vectorized_url)[0][1])
    else:
        st.warning('Please enter a URL before checking for phishing.')

# Examples
st.subheader('Examples to Try:')
example_urls = [
    "http://example.com/verify/account",
    "https://en.wikipedia.org",
    "http://securelogin.netflix.access.com"
]
for example_url in example_urls:
    if st.button(example_url, key=example_url):
        url = example_url
        preprocessed_url = preprocess_text(url)
        vectorized_url = cv.transform([preprocessed_url])
        prediction = loaded_model.predict(vectorized_url)
        if prediction[0] == 'good':
            st.success('The URL is likely safe.')
        else:
            st.error('The URL is potentially phishing.')
        st.write('Phishing Probability:', loaded_model.predict_proba(vectorized_url)[0][1])
