import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

from flask import Flask, render_template, request
import transformers
import spacy
import torch.nn.functional as F
import torch

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

meaningful_adjectives = [
    "little", "bad", "good", "sad", "other", "more", "sorry", "lonely", 
    "last", "stupid", "many", "guilty", "emotional", "depressed", "much", 
    "low", "useless", "horrible", "embarrassed", "terrible", "worthless", 
    "unhappy", "awful", "hopeless", "empty"
    "good", "little", "happy", "more", "other", "sure", "new", "many", 
    "strong", "able", "excited", "important", "comfortable", "confident", 
    "amazing", "well", "free", "wonderful", "positive", "few", "thankful", 
    "own", "special", "glad", "sweet"
    "hot", "sweet", "little", "sympathetic", "gentle", "lovely", "romantic", 
    "passionate", "nostalgic", "generous", "supportive", "delicate", "loyal", 
    "beloved", "good", "more", "other", "fond", "tender", "faithful", "horny", 
    "blessed", "compassionate", "much", "many"
    "little", "angry", "selfish", "jealous", "frustrated", "irritated", "mad", 
    "bitter", "resentful", "other", "irritable", "dangerous", "cold", "rude", 
    "grumpy", "cranky", "dissatisfied", "greedy", "impatient", "agitated", 
    "mean", "violent", "envious", "good", "annoyed"
    "little", "afraid", "anxious", "scared", "nervous", "uncomfortable", 
    "weird", "strange", "overwhelmed", "vulnerable", "unsure", "reluctant", 
    "shaky", "hesitant", "uncertain", "helpless", "paranoid", "shy", "agitated", 
    "insecure", "fearful", "restless", "frightened", "pressured", "confused"
    "weird", "amazed", "strange", "amazing", "overwhelmed", "curious", "funny", 
    "surprised", "impressed", "little", "shocked", "stunned", "other", "good", 
    "more", "many", "much", "last", "same", "new", "first", "few", "sure", 
    "happy", "own"
]

# Emotion labels
labels = {
    "None": "",
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
}

def predict_emotion_with_highlight(phrase, emotion_pipeline):
    """
    Predict the emotion and highlight tokens contributing to the prediction.

    Args:
        phrase (str): Input text.
        emotion_pipeline: Pretrained model pipeline for emotion classification.

    Returns:
        tuple: Predicted label, confidence score, scores for all labels, and token highlights.
    """
    tokenizer = emotion_pipeline.tokenizer
    model = emotion_pipeline.model

    # Tokenize the phrase
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    # Process logits and calculate softmax scores
    logits = outputs.logits[0].detach().numpy()
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    all_scores = {labels[f"LABEL_{i}"]: probabilities[i] for i in range(len(probabilities))}
    
    # Get the predicted label
    predicted_index = probabilities.argmax()
    predicted_label = f"LABEL_{predicted_index}"
    confidence_score = probabilities[predicted_index]

    # Highlight tokens by mapping scores back
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_highlights = [
        {
            "token": token,
            "highlight": token.lower() in meaningful_adjectives 
        }
        for token in tokens if token not in ["[SEP]", "[CLS]"]
    ]

    return predicted_label, confidence_score, all_scores, token_highlights


def analyze_phrase(phrase):
    """
    Analyze the input phrase using SpaCy and return details about each word.
    
    Args:
        phrase (str): Input text.
        
    Returns:
        list: List of word details (POS, DEP).
    """
    doc = nlp(phrase)
    word_details = [
        {"text": token.text, "pos": token.pos_, "dep": token.dep_, "morph": token.morph}
        for token in doc
    ]
    return word_details

def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion_pipeline_distilbert = transformers.pipeline(
        "text-classification",
        model="./emotion_model/distilbert",
        tokenizer="./emotion_model/distilbert",
        return_all_scores=True
    )

    emotion_pipeline_roberta = transformers.pipeline(
        "text-classification",
        model="./emotion_model/roberta",
        tokenizer="./emotion_model/roberta",
        use_fast=False,  # Force using the slow tokenizer
        return_all_scores=True
    )

    emotion_pipeline_bert = transformers.pipeline(
        "text-classification",
        model="./emotion_model/bert",
        tokenizer="./emotion_model/bert",
        return_all_scores=True
    )

    predicted_label = "None"
    confidence_score = "None"
    phrase = ''
    word_details = []
    all_scores = {}
    action = ''
    all_predictions = {}
    highlights = []
    original_phrase = ''
    
    if request.method == 'POST':

        original_phrase = request.form.get('phrase', '')
        phrase =  clean_text(original_phrase)
        action = request.form.get('action')  # Get the button action

        if action == "bert":
            if phrase:
                # Predict emotion
                predicted_label, confidence_score, all_scores, highlights = predict_emotion_with_highlight(phrase, emotion_pipeline_distilbert)
                # Analyze phrase with SpaCy
                word_details = analyze_phrase(original_phrase)
        
        elif action == "distilbert":
            if phrase:
                # Predict emotion
                predicted_label, confidence_score, all_scores, highlights = predict_emotion_with_highlight(phrase, emotion_pipeline_bert)
                # Analyze phrase with SpaCy
                word_details = analyze_phrase(original_phrase)

        elif action == "roberta":
            if phrase:
                # Predict emotion
                predicted_label, confidence_score, all_scores, highlights = predict_emotion_with_highlight(phrase, emotion_pipeline_roberta)
                # Analyze phrase with SpaCy
                word_details = analyze_phrase(original_phrase)

        elif action == "all":
            if phrase:
                # Get predictions from all models
                all_predictions = {
                    "bert": predict_emotion_with_highlight(phrase, emotion_pipeline_bert),
                    "distilbert": predict_emotion_with_highlight(phrase, emotion_pipeline_distilbert),
                    "roberta": predict_emotion_with_highlight(phrase, emotion_pipeline_roberta),
                }
                word_details = analyze_phrase(original_phrase)

    if original_phrase and phrase:
        print(f"ORIGINAL: {original_phrase}, CLEANED: {phrase}")

    return render_template(
        'index.html',
        phrase=original_phrase,
        action = action,
        labels=labels,
        predicted_label=labels[predicted_label],
        all_predictions = all_predictions if action == "all" else None,
        confidence_score=confidence_score,
        word_details=word_details,
        all_scores=all_scores,
        # token_highlights=highlights
    )

if __name__ == '__main__':
    app.run(debug=True)

