import streamlit as st
import os
from Visualization.HelperFun import load_lottie_file
from streamlit_lottie   import st_lottie
import openai

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY","1a8f434f-2873-4d5d-bd86-836616720fbb"),
    base_url="https://api.sambanova.ai/v1",
)

def AISphere():
    st.markdown("""
        <style>
        .css-18e3th9 {
            padding-top: 0 !important;
        }
        .css-1d391kg {
            padding-top: 0 !important;
        }
        .st-emotion-cache-13ln4jf{
            padding-top: 0 !important;
        }
        .st-emotion-cache-1jicfl2{
            padding-top: 1rem !important;
        }
        .st-emotion-cache-kgpedg{
            padding: 0 !important;
        }
        .st-emotion-cache-7tauuy{
            padding-top: 1rem !important;
        }
        # header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
        
    with st.sidebar:
        image_path = './Visualization/assets/Images/logo.png'
        
        # Check if the file exists before trying to display it
        if os.path.exists(image_path):
            st.image(image_path, width=200)
        else:
            st.error("Logo image not found.")
        
        lottie_json = load_lottie_file("./Visualization/FilesJson/Chatbot.json")    
        st_lottie(lottie_json, speed=1, width=250, height=250, key="initial")
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "system", "content": "Welcome! I'm your chatbot, ready to assist you with anything you need. How can I help you today? 😊"}
        ]



    # Function to display references

    
    # Function to get a response from Azure OpenAI
    def get_response(user_query):
        
        chat_prompt = st.session_state["chat_history"] + [{"role": "user", "content": str(user_query)}]

        # Generate the completion
        completion = client.chat.completions.create(
                    model='Meta-Llama-3.1-8B-Instruct',
                    messages=chat_prompt,
                    temperature =  0.5,
                    top_p = 0.1
                )

        
        return [completion.choices[0].message.content]



    # Display chat history

    for message in st.session_state["chat_history"]:
        if message["role"] == "system":
            with st.chat_message("System"):
                st.write(message["content"])
        elif message["role"] == "user":
            with st.chat_message("Human"):
                st.write(message["content"])
                
        elif message["role"] == "assistant":
            with st.chat_message("AI"):    
                st.write(message["content"])

    # Display chat history and handle the response as needed

    user_query = st.chat_input("Type your message here...")
    if user_query:
        # Append user's query to chat history (ensure it's a string)
        st.session_state["chat_history"].append({"role": "user", "content": str(user_query)})
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Get AI response and display it
        with st.chat_message("AI"):
            response = get_response(user_query)
            st.write(response[0])  # AI's response
            print(response[0])
            
            

        # Append AI's response to chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": response[0]})

import json 
      
def load_lottie_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lottie_json = json.load(f)
    except UnicodeDecodeError:
        # Handle the error or fallback to a different encoding
        with open(file_path, "r", encoding="latin-1") as f:
            lottie_json = json.load(f)
    return lottie_json


if __name__ == "__main__":
    AISphere()
    
    
    
    