import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the model
model = load_model('LSTM_RNN_Project/next_word_LSTM.h5')

#load the tokenizer 
with open('LSTM_RNN_Project/tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word ,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None


st.title('Next word prediction using LSTM and Earlystopping')
input_text = st.text_input("Enter the text input  ")
if st.button("predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"next_word: {next_word}")