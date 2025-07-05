import streamlit as st
import requests
import sys

TIMEOUT =30
def main():
    st.title("Medical Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"http://127.0.0.1:8001/query",
                json={"question": prompt},
                timeout=60
            )
            data = response.json()
            
            with st.chat_message("assistant"):
                st.markdown(data["answer"])
                st.caption(f"Sources: {', '.join(data['sources'])}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"]
            })
            
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed - is the server running?")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()