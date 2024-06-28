import streamlit as st
import requests
import json
import time

st.title("Text Generation with Typewriter Effect")


def typewriter_text(generated_text):
    for char in generated_text:
        if char == '\n\n':
            char = ' ' 
        yield char
        time.sleep(0.03)  # Typing speed

# Streamlit app
input_text = st.text_area("Enter text to generate:")

if st.button("Generate"):
    if input_text.strip():  
        with st.spinner("Generating..."):
            try:
                payload = {"prompts": [input_text]}  
                response = requests.post("http://localhost:9008/generate_response/", json=payload)
                
                if response.status_code == 200:
                    try:
                        data = response.json()  
                        
                        
                        generations = data.get('response', {}).get('generations', [])
                        if generations and len(generations[0]) > 0:
                            generated_text = generations[0][0].get('text', '')
                            
                            if generated_text:
                                st.success("Text generated successfully!")

                                
                                text_placeholder = st.empty()  
                                text_to_show = ""

                                for char in typewriter_text(generated_text):
                                    text_to_show += char
                                    text_placeholder.text(text_to_show)  
                            else:
                                st.warning("No text found in the generated response.")
                        else:
                            st.warning("Invalid or empty generations format in the response.")
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding JSON: {e}")
                        st.error(f"Response content: {response.text}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.RequestException as e:
                st.error(f"Request Exception: {e}")
    else:
        st.warning("Please enter some text before generating.")

st.write("Note: The generated text will appear with a typewriter effect.")
