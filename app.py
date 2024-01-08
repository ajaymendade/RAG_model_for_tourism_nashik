from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import faiss
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

app = Flask(__name__)

# Load the pre-trained model
model_data = joblib.load('nashik_model.pkl')
df = model_data['df']
vectorizer = model_data['vectorizer']
index = model_data['index']

# Initialize OpenAI API key
openai.api_key = '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'

# Initialize Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Placeholder for the context (you may need to implement this based on your needs)
context = {'previous_context': ''}

# Placeholder for the retrieve_information function (replace it with your implementation)
def retrieve_information(user_prompt, vectorizer, index, df, entities=None):
    if entities and 'Nashik' not in entities:
        return "Sorry, I can only provide information about places in Nashik."

    if entities:
        location_matches = df[df['Location'].str.contains('Nashik') & df['Location'].str.contains(entities[0], case=False)]

        if not location_matches.empty:
            description_vectors = vectorizer.transform(location_matches['Description'].values.astype('U')).toarray()
            _, similar_indices = index.search(description_vectors.astype(np.float32), k=5)

            if similar_indices.size > 0 and np.all(similar_indices < len(location_matches)):
                retrieved_info = location_matches.iloc[similar_indices[0]]
                return retrieved_info
            else:
                return None
        else:
            return None

    return None

# Placeholder for the generate_gpt_response function (replace it with your implementation)
def generate_gpt_response(prompt):
    # Add "Nashik" to the prompt if not already present
    if 'Nashik in 190 words' not in prompt:
        prompt += ' Nashik'

    try:
        # Set an initial max_tokens limit
        max_tokens_limit = 200

        # Make the initial API call
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens_limit
        )

        # Check if the response is truncated
        while response['choices'][0]['text'].endswith('...'):
            # If truncated, increase the max_tokens limit
            max_tokens_limit += 50

            # Make another API call with an increased max_tokens limit
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens_limit
            )

        return response['choices'][0]['text'].strip()

    except Exception as e:
        error_message = f"Error in OpenAI API call: {e}"
        print(error_message)
        return f"Sorry, I encountered an issue and cannot provide a response at the moment. {error_message}"


# Function to extract entities using Spacy (replace it with your implementation)
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Flask route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    response = None

    if request.method == 'POST':
        user_prompt = request.form['user_prompt']
        action = request.form['action']

        if action == 'Submit':
            # Retrieve information based on user prompt
            entities = extract_entities(user_prompt)
            retrieved_info = retrieve_information(user_prompt, vectorizer, index, df, entities)

            # Update context if information is retrieved
            if retrieved_info is not None:
                context['previous_context'] = f"{context.get('previous_context', '')} {retrieved_info['Description']}"

            # Generate response using GPT-3
            response = generate_gpt_response(user_prompt)

        elif action == 'Clear':
            # Clear previous context and response
            context['previous_context'] = ''
            response = None

    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
