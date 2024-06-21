import streamlit as st
import requests
import json
import time

st.title("Text Generation with Typewriter Effect")

# Function to generate text with a typewriter effect
def typewriter_text(generated_text):
    for char in generated_text:
        if char == '\n':
            char = ' '  # Replace newline characters with spaces
        yield char
        time.sleep(0.03)  # Typing speed

# Streamlit app
input_text = st.text_area("Enter text to generate:")

if st.button("Generate"):
    if input_text.strip():  # Check if input is not empty
        # Send POST request to FastAPI server
        with st.spinner("Generating..."):
            try:
                payload = {"text": input_text}  # Prepare the input data as a dictionary
                response = requests.post("http://127.0.0.1:6087/generate", json=payload)
                
                if response.status_code == 200:
                    try:
                        data = response.json()  # Try to parse JSON response
                        generated_text = data.get('generated_text', '')  # Get generated text from JSON response
                        if generated_text:
                            st.success("Text generated successfully!")

                            # Display generated text with typewriter effect
                            text_placeholder = st.empty()  # Placeholder to update generated text
                            text_to_show = ""

                            for char in typewriter_text(generated_text):
                                text_to_show += char
                                text_placeholder.text(text_to_show)  # Update text placeholder

                        else:
                            st.warning("Received empty or invalid data format from server.")
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
