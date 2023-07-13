import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re      # Regex
import nltk    # Natural Language Toolkit
#nltk.data.path = [r'C:\Users\KIIT\anaconda3\nltk_data']
from nltk.corpus import stopwords    # Words like the, a
from nltk.stem.porter import PorterStemmer   # Making all stem words in present tense
nltk.download('stopwords')
# Load the XGBoost model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the tuning_sen function
#tuning_sen = joblib.load('tuning_sen.pkl')

# Load the CountVectorizer
with open('cv.pkl', 'rb') as file:
    cv= pickle.load(file)


# tuning_sen Function
def preprocess_text(texts):
    texts=[texts]
    corpus = []
    for i in range(0, len(texts)):
        mail = re.sub('[^a-zA-Z]', ' ', texts[i])
        mail = mail.lower()
        mail = mail.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        #all_stopwords.remove('OK')   # Otherwise it was removing and this is essential
        mail = [ps.stem(word) for word in mail if not word in set(all_stopwords)]
        mail = ' '.join(mail)
        corpus.append(mail)
    return corpus

def predict_spam(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    vectorized_text = cv.transform(preprocessed_text).toarray()

    # Make the prediction
    prediction = model.predict(vectorized_text)

    return prediction[0]

# Streamlit app
def main():
    st.title("Spam Classification App")

    # Text input for user to enter the email
    email_text = st.text_area("Enter the email text", "",height=400)

    
    if st.button("Classify"):
        if email_text:
            # Perform spam classification
            prediction = predict_spam(email_text)

            # Display the result
            if prediction == 1:
                st.error("This email is classified as spam.")
            else:
                st.success("This email is not classified as spam.")
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
