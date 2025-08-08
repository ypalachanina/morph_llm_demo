# MorphLLM README

## Overview
MorphLLM Demo is a Streamlit app that uses LLM (gemini-2.5-flash) to act as AR navigation assistant and answer user questions about the environment.

It can analyse stream from live camera or preloaded videos from Azure Storage. 

It accepts multilingual audio input and provides text and audio responses in English, Dutch and Flemish using Azure AI Voices.

## How to Run the App

1. **Install Requirements**  
   Ensure that all required libraries are installed, either manually or by running:
   ```bash
   pip install -r requirements.txt

2. **Navigate to the App Directory**  
   Change your directory to the location of the app:
   ```bash
   cd path/to/MorphLLM

3. **Run the App**  
   Execute the app with the following command:
   ```bash
   streamlit run main.py

4. **View the App**
   After running the command, a local server will start, and you can access the app in your web browser at the URL provided (usually http://localhost:8501/).
