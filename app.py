from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import pandas as pd
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import os
import string
import json 

app = Flask(__name__)
# Enable CORS with more explicit configuration
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize lemmatizer, stopwords and sentiment analyzer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Load the trained model and vectorizer
model_path = 'best_fake_news_model.joblib'
vectorizer_path = 'vectorizer.joblib'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    print(f"Model and vectorizer loaded successfully")
    is_lstm = False
elif os.path.exists('best_fake_news_model') and os.path.exists('tokenizer.joblib'):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    model = load_model('best_fake_news_model')
    tokenizer = load('tokenizer.joblib')
    max_len = 200  # Same as in training
    print(f"LSTM model and tokenizer loaded successfully")
    is_lstm = True
else:
    print(f"Warning: Model or vectorizer file not found. API will have limited functionality.")
    model = None
    vectorizer = None
    is_lstm = False

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(processed_tokens)
    else:
        return ''

# Feature extraction for text analysis with enhanced details
def analyze_features(text):
    # Basic text features
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = max(1, len(text.split('.')))
    avg_word_length = np.mean([len(word) for word in text.split()]) if len(text.split()) > 0 else 0
    
    # Common fake news phrases with categories
    fake_phrases = {
        "sensationalism": [
            "shocking truth", "you won't believe", "shocking revelation", "stunning", 
            "mind-blowing", "jaw-dropping", "astounding", "bombshell"
        ],
        "conspiracy": [
            "conspiracy", "cover-up", "they don't want you to know", 
            "what they're hiding", "government doesn't want you to know", 
            "illuminati", "deep state", "shadow government"
        ],
        "medical_misinformation": [
            "miracle cure", "doctors hate", "breakthrough", "alternative medicine",
            "big pharma doesn't want you to know", "secret remedy", "cures all"
        ],
        "exaggeration": [
            "exclusive", "leaked", "exposed", "controversial", "censored",
            "breaking", "urgent", "explosive", "never before seen"
        ],
        "vague_sourcing": [
            "anonymous sources", "sources say", "insiders reveal", "some say",
            "people are saying", "many believe", "reportedly", "allegedly"
        ]
    }
    
    # Common real news phrases with categories
    real_phrases = {
        "research_based": [
            "according to research", "study finds", "data suggests", 
            "evidence shows", "analysis reveals", "researchers have found", 
            "scientists conclude", "findings suggest", "data shows"
        ],
        "expert_opinions": [
            "experts say", "according to experts", "specialist explains",
            "professor states", "scientist confirms", "economist predicts"
        ],
        "official_statements": [
            "official statement", "spokesperson said", "official report",
            "press release states", "according to the official", "statement reads"
        ],
        "statistical_evidence": [
            "statistics indicate", "survey reveals", "polls indicate",
            "percentage of", "majority of respondents", "statistical analysis"
        ],
        "balanced_reporting": [
            "on the other hand", "however", "alternatively", "in contrast",
            "different perspective", "opposing view", "both sides"
        ]
    }
    
    # Sentiment analysis
    sentiment = sid.polarity_scores(text)
    
    # Check for phrases in the text
    text_lower = text.lower()
    
    # Track fake news phrases by category
    fake_phrase_categories = {}
    total_fake_phrases = 0
    for category, phrases in fake_phrases.items():
        category_count = sum(1 for phrase in phrases if phrase in text_lower)
        if category_count > 0:
            fake_phrase_categories[category] = category_count
            total_fake_phrases += category_count
    
    # Track real news phrases by category
    real_phrase_categories = {}
    total_real_phrases = 0
    for category, phrases in real_phrases.items():
        category_count = sum(1 for phrase in phrases if phrase in text_lower)
        if category_count > 0:
            real_phrase_categories[category] = category_count
            total_real_phrases += category_count
    
    # Advanced text analysis
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # POS tag analysis
        num_nouns = len([word for word, pos in pos_tags if pos.startswith('N')])
        num_verbs = len([word for word, pos in pos_tags if pos.startswith('V')])
        num_adj = len([word for word, pos in pos_tags if pos.startswith('J')])
        num_adv = len([word for word, pos in pos_tags if pos.startswith('R')])
        
        # Calculate ratios
        total = len(tokens) if len(tokens) > 0 else 1
        noun_ratio = num_nouns/total
        verb_ratio = num_verbs/total
        adj_ratio = num_adj/total
        adv_ratio = num_adv/total
        
        # Word complexity
        complex_words = [word for word in tokens if len(word) > 6]
        complex_word_ratio = len(complex_words)/total if total > 0 else 0
        
        # Readability (simplified Flesch Reading Ease)
        words_per_sentence = word_count / sentence_count
        syllables = sum([max(1, len(re.findall(r'[aeiouy]+', word.lower()))) for word in tokens])
        syllables_per_word = syllables / word_count if word_count > 0 else 0
        readability_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        
        # Capitalization patterns (often overused in fake news)
        all_caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        all_caps_ratio = all_caps_words / word_count if word_count > 0 else 0
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0
        
    except Exception as e:
        print(f"Error in advanced text analysis: {e}")
        noun_ratio = verb_ratio = adj_ratio = adv_ratio = 0
        complex_word_ratio = all_caps_ratio = punctuation_ratio = 0
        readability_score = 0
        exclamation_count = question_count = 0
    
    # Find quoted material (potential sources)
    quote_pattern = re.compile(r'"([^"]*)"')
    quotes = quote_pattern.findall(text)
    quoted_text_count = len(quotes)
    
    # Find claim patterns
    claim_phrases = ["claim", "claims", "stated", "says", "according to"]
    claim_count = sum(1 for phrase in claim_phrases if phrase in text_lower)
    
    # Find hedge language (common in uncertain claims)
    hedge_words = ["may", "might", "could", "appears", "seems", "possibly", "perhaps", "allegedly"]
    hedge_count = sum(1 for word in hedge_words if f" {word} " in f" {text_lower} ")
    
    # Analyze URL patterns (high number can indicate SEO manipulation)
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    
    return {
        # Basic metrics
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": float(avg_word_length),
        
        # Fake/Real phrase analysis
        "fake_phrase_count": total_fake_phrases,
        "real_phrase_count": total_real_phrases,
        "fake_phrase_categories": fake_phrase_categories,
        "real_phrase_categories": real_phrase_categories,
        
        # Sentiment analysis
        "sentiment": {
            "positive": float(sentiment["pos"]),
            "negative": float(sentiment["neg"]),
            "neutral": float(sentiment["neu"]),
            "compound": float(sentiment["compound"])
        },
        
        # POS analysis
        "pos_analysis": {
            "noun_ratio": float(noun_ratio),
            "verb_ratio": float(verb_ratio),
            "adj_ratio": float(adj_ratio),
            "adv_ratio": float(adv_ratio)
        },
        
        # Style and complexity
        "style_metrics": {
            "complex_word_ratio": float(complex_word_ratio),
            "readability_score": float(readability_score),
            "all_caps_ratio": float(all_caps_ratio),
            "punctuation_ratio": float(punctuation_ratio),
            "exclamation_count": exclamation_count,
            "question_count": question_count
        },
        
        # Source and credibility indicators
        "credibility_indicators": {
            "quoted_text_count": quoted_text_count,
            "claim_count": claim_count,
            "hedge_count": hedge_count,
            "url_count": url_count
        }
    }

# Enhanced function to predict and analyze news with detailed explanation
def analyze_news(text):
    if model is None:
        return {
            "error": "Model not loaded. Please ensure model files are available."
        }
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Extract features for analysis
    features = analyze_features(text)
    
    # Make prediction
    if is_lstm:
        # Process for LSTM model
        sequences = tokenizer.texts_to_sequences([processed_text])
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        prediction_prob = float(model.predict(padded_sequences)[0][0])
        prediction = 1 if prediction_prob > 0.5 else 0
    else:
        # Process for traditional ML models
        vectorized_text = vectorizer.transform([processed_text])
        prediction = int(model.predict(vectorized_text)[0])
        
        try:
            prediction_prob = float(model.predict_proba(vectorized_text)[0][1])
        except:
            try:
                prediction_prob = float(model.decision_function(vectorized_text)[0])
                # Normalize to [0,1] range
                prediction_prob = 1 / (1 + np.exp(-prediction_prob))
            except:
                prediction_prob = float(prediction)
    
    # Generate explanation and insights
    explanation, red_flags, credibility_marks = generate_explanation(text, features, prediction)
    
    # Get key phrases that influenced the decision
    key_phrases = extract_key_phrases(text, prediction, features)
    
    # Compile results
    result = {
        "text": text,
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": float(prediction_prob if prediction == 1 else 1 - prediction_prob),
        "confidence_percentage": float(round((prediction_prob if prediction == 1 else 1 - prediction_prob) * 100, 1)),
        "explanation": explanation,
        "red_flags": red_flags,
        "credibility_marks": credibility_marks,
        "features": features,
        "key_phrases": key_phrases,
        "reliability_score": calculate_reliability_score(features, prediction)
    }
    
    return result

# Generate detailed explanation based on features
def generate_explanation(text, features, prediction):
    explanation = []
    red_flags = []
    credibility_marks = []
    
    # Analyze content characteristics
    if prediction == 1:  # Fake news
        explanation.append("This content shows characteristics often associated with misleading or false information.")
        
        # Check for sensationalist language
        if features["sentiment"]["compound"] > 0.5 or features["sentiment"]["compound"] < -0.5:
            explanation.append("The text uses highly emotional language, which is often used to provoke strong reactions rather than inform.")
            red_flags.append("Emotionally charged language")
        
        # Check for sensationalist phrases
        if "sensationalism" in features["fake_phrase_categories"]:
            explanation.append("The article contains sensationalist phrases that are often used to grab attention rather than provide accurate information.")
            red_flags.append("Use of sensationalist language")
            
        # Check for conspiracy language
        if "conspiracy" in features["fake_phrase_categories"]:
            explanation.append("The text contains conspiracy-themed language that suggests hidden motives or cover-ups without providing evidence.")
            red_flags.append("Conspiracy narrative")
            
        # Check for medical misinformation patterns
        if "medical_misinformation" in features["fake_phrase_categories"]:
            explanation.append("The content uses patterns common in medical misinformation, such as miracle cures or treatments that established medicine supposedly ignores.")
            red_flags.append("Unverified medical claims")
            
        # Check for stylistic red flags
        if features["style_metrics"]["all_caps_ratio"] > 0.1:
            explanation.append("The text contains an unusual amount of ALL CAPS text, which is often used for emphasis in misleading content.")
            red_flags.append("Excessive use of ALL CAPS")
            
        if features["style_metrics"]["exclamation_count"] > 3:
            explanation.append("Multiple exclamation marks are used, which is more common in emotionally manipulative text than in factual reporting.")
            red_flags.append("Excessive exclamation marks")
            
        # Check for lack of sources
        if features["credibility_indicators"]["quoted_text_count"] < 1 and len(text.split()) > 100:
            explanation.append("The text lacks direct quotes or citations from authoritative sources, which are important for verifying claims.")
            red_flags.append("Lack of verifiable sources")
            
        # Check for vague sourcing
        if "vague_sourcing" in features["fake_phrase_categories"]:
            explanation.append("The article uses vague attributions like 'sources say' or 'people are saying' without specific, verifiable sources.")
            red_flags.append("Vague or anonymous sourcing")
            
        # Check for hedge language combined with strong claims
        if features["credibility_indicators"]["hedge_count"] > 2 and features["sentiment"]["compound"] > 0.4:
            explanation.append("The text combines uncertain language with strong claims, which can be a tactic to spread misinformation while avoiding accountability.")
            red_flags.append("Hedging language with strong claims")
            
    else:  # Real news
        explanation.append("This content demonstrates several characteristics associated with credible information.")
        
        # Check for research-based language
        if "research_based" in features["real_phrase_categories"]:
            explanation.append("The text references research or data, which suggests an evidence-based approach.")
            credibility_marks.append("References to research or data")
            
        # Check for expert citations
        if "expert_opinions" in features["real_phrase_categories"]:
            explanation.append("The content cites expert opinions, which adds credibility to the information presented.")
            credibility_marks.append("Expert citations")
            
        # Check for official sources
        if "official_statements" in features["real_phrase_categories"]:
            explanation.append("The article references official statements or reports, which are typically more reliable sources.")
            credibility_marks.append("Official source citations")
            
        # Check for balanced reporting
        if "balanced_reporting" in features["real_phrase_categories"]:
            explanation.append("The text presents multiple perspectives or balanced viewpoints, which is a hallmark of quality journalism.")
            credibility_marks.append("Balanced perspective")
            
        # Check for appropriate style
        if features["style_metrics"]["readability_score"] > 50 and features["style_metrics"]["readability_score"] < 70:
            explanation.append("The content has an appropriate readability level for informational text, neither oversimplified nor needlessly complex.")
            credibility_marks.append("Appropriate readability")
            
        # Check for source indicators
        if features["credibility_indicators"]["quoted_text_count"] >= 2:
            explanation.append("The article includes multiple direct quotes, which allows readers to verify the information.")
            credibility_marks.append("Multiple quoted sources")
            
        # Check for moderate sentiment
        if -0.3 < features["sentiment"]["compound"] < 0.3:
            explanation.append("The text maintains a relatively neutral tone, focusing on facts rather than emotional appeals.")
            credibility_marks.append("Neutral, fact-focused tone")
    
    # Add general analysis based on linguistic features
    if features["pos_analysis"]["noun_ratio"] > 0.4:
        explanation.append("The text has a high proportion of nouns, suggesting it is information-dense and focused on specific entities, events, or concepts.")
        
    if features["style_metrics"]["complex_word_ratio"] > 0.2:
        explanation.append("The article uses specialized or complex vocabulary, which may indicate either subject expertise or intentionally complicated language to seem authoritative.")
    
    return explanation, red_flags, credibility_marks

# Extract key phrases that influenced the decision
def extract_key_phrases(text, prediction, features):
    # Common phrases by category based on prediction
    text_lower = text.lower()
    key_phrases = []
    
    if prediction == 1:  # Fake news
        # Add sensationalist phrases
        for category in features["fake_phrase_categories"]:
            if category == "sensationalism":
                phrases = ["shocking truth", "you won't believe", "shocking revelation", "stunning", 
                        "mind-blowing", "jaw-dropping", "astounding", "bombshell"]
            elif category == "conspiracy":
                phrases = ["conspiracy", "cover-up", "they don't want you to know", 
                        "what they're hiding", "government doesn't want you to know"]
            elif category == "medical_misinformation":
                phrases = ["miracle cure", "doctors hate", "breakthrough", "alternative medicine",
                        "big pharma doesn't want you to know"]
            elif category == "exaggeration":
                phrases = ["exclusive", "leaked", "exposed", "controversial", "censored",
                        "breaking", "urgent", "explosive"]
            elif category == "vague_sourcing":
                phrases = ["anonymous sources", "sources say", "insiders reveal", "some say",
                        "people are saying", "many believe"]
            else:
                phrases = []
                
            for phrase in phrases:
                if phrase in text_lower and phrase not in [p["phrase"] for p in key_phrases]:
                    key_phrases.append({
                        "phrase": phrase,
                        "category": category,
                        "context": extract_context(text, phrase)
                    })
    else:  # Real news
        # Add credibility phrases
        for category in features["real_phrase_categories"]:
            if category == "research_based":
                phrases = ["according to research", "study finds", "data suggests", 
                        "evidence shows", "analysis reveals", "researchers have found"]
            elif category == "expert_opinions":
                phrases = ["experts say", "according to experts", "specialist explains",
                        "professor states", "scientist confirms"]
            elif category == "official_statements":
                phrases = ["official statement", "spokesperson said", "official report",
                        "press release states"]
            elif category == "statistical_evidence":
                phrases = ["statistics indicate", "survey reveals", "polls indicate",
                        "percentage of", "majority of respondents"]
            elif category == "balanced_reporting":
                phrases = ["on the other hand", "however", "alternatively", "in contrast",
                        "different perspective"]
            else:
                phrases = []
                
            for phrase in phrases:
                if phrase in text_lower and phrase not in [p["phrase"] for p in key_phrases]:
                    key_phrases.append({
                        "phrase": phrase,
                        "category": category,
                        "context": extract_context(text, phrase)
                    })
    
    # Limit to top phrases (prioritize by length of context or presence)
    key_phrases = sorted(key_phrases, key=lambda x: len(x["context"]), reverse=True)
    return key_phrases[:5]  # Return top 5 key phrases

# Extract context around a key phrase
def extract_context(text, phrase):
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    
    # Find the start index of the phrase
    start_idx = text_lower.find(phrase_lower)
    if start_idx == -1:
        return ""
    
    # Extract a window around the phrase (50 chars before and after)
    context_start = max(0, start_idx - 50)
    context_end = min(len(text), start_idx + len(phrase) + 50)
    
    # Get the context
    context = text[context_start:context_end]
    
    # Add ellipsis if we've truncated
    if context_start > 0:
        context = "..." + context
    if context_end < len(text):
        context = context + "..."
    
    return context

# Calculate a comprehensive reliability score (0-100)
def calculate_reliability_score(features, prediction):
    score = 50  # Start at neutral
    
    # Adjust based on prediction confidence
    if prediction == 0:  # Real news prediction
        score += 20  # Boost score
    else:  # Fake news prediction
        score -= 20  # Lower score
    
    # Adjust based on fake/real phrase counts
    fake_phrases = features["fake_phrase_count"]
    real_phrases = features["real_phrase_count"]
    
    # More real phrases boost score, more fake phrases lower it
    score += min(15, real_phrases * 3)
    score -= min(15, fake_phrases * 3)
    
    # Adjust based on sentiment extremity
    sentiment_extremity = abs(features["sentiment"]["compound"])
    if sentiment_extremity > 0.5:
        score -= min(10, int((sentiment_extremity - 0.5) * 20))  # Penalize extreme sentiment
    
    # Adjust based on source indicators
    quoted_text = features["credibility_indicators"]["quoted_text_count"]
    score += min(10, quoted_text * 2)  # Reward quotes
    
    # Penalize vague sourcing and hedge language in combination
    hedge_count = features["credibility_indicators"]["hedge_count"]
    if "vague_sourcing" in features.get("fake_phrase_categories", {}) and hedge_count > 1:
        score -= min(10, 5 + hedge_count)
    
    # Cap score between 0-100
    return max(0, min(100, score))

# API route to check if service is alive
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "Fake News Detection API is running",
        "model_loaded": model is not None,
        "enhanced_analysis": True
    })

# API endpoint for analyzing news text
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Print request details for debugging
        print(f"Received request: {request.method} {request.path}")
        print(f"Request headers: {request.headers}")
        print(f"Request data: {request.data}")
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            print(f"Parsed JSON: {data}")
        else:
            try:
                data = request.get_json(force=True)
                print(f"Forced JSON parsing: {data}")
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                return jsonify({'error': 'Invalid JSON provided'}), 400
        
        text = data.get('text')
        print(f"Extracted text: {text[:100]}...")
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = analyze_news(text)
        print(f"Analysis result: {result['prediction']} with {result['confidence']:.2f} confidence")
        
        # Add CORS headers explicitly
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Content-Type', 'application/json')
        return Response(json.dumps(result), mimetype= 'application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint for model information
@app.route('/api/model-info', methods=['GET'])
def model_info():
    # Return model metrics and information
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # These values would typically come from your model evaluation
    metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1Score": 0.95,
        "modelType": "LSTM" if is_lstm else "RandomForest",
        "featuresUsed": [
            "TF-IDF", 
            "Word Count", 
            "POS Tags", 
            "Lexical Features",
            "Sentiment Analysis",
            "Stylistic Patterns",
            "Source Credibility Indicators"
        ],
        "analysisCapabilities": [
            "Content Classification",
            "Linguistic Pattern Analysis",
            "Source Credibility Assessment",
            "Stylistic Analysis",
            "Emotional Tone Evaluation",
            "Detailed Explanation Generation"
        ]
    }
    
    return jsonify(metrics)

# New endpoint for getting detailed analysis categories
@app.route('/api/analysis-categories', methods=['GET'])
def analysis_categories():
    # Return the categories of analysis that the system can perform
    categories = {
        "fake_news_indicators": {
            "sensationalism": "Exaggerated or shocking claims designed to provoke emotional reactions",
            "conspiracy": "Suggestions of hidden plots, cover-ups, or malicious intent by powerful groups",
            "medical_misinformation": "Unfounded health claims, miracle cures, or treatments rejected by medical consensus",
            "exaggeration": "Overstatement of facts or importance beyond what evidence supports",
            "vague_sourcing": "Unclear attribution of information to unspecified or anonymous sources"
        },
        "credibility_indicators": {
            "research_based": "References to scientific studies, data, or empirical evidence",
            "expert_opinions": "Citations of qualified experts in the relevant field",
            "official_statements": "References to official sources, documents, or authorized spokespersons",
            "statistical_evidence": "Use of specific statistics, surveys, or quantitative data",
            "balanced_reporting": "Presentation of multiple perspectives or consideration of alternative viewpoints"
        },
        "linguistic_features": {
            "sentiment": "Emotional tone and intensity of the language used",
            "complexity": "Reading level, vocabulary sophistication, and sentence structure",
            "stylistic_patterns": "Capitalization, punctuation, and formatting choices",
            "part_of_speech": "Distribution of nouns, verbs, adjectives, and adverbs"
        },
        "source_indicators": {
            "quoted_material": "Direct quotes attributed to specific individuals or documents",
            "claim_patterns": "How assertions are presented and attributed",
            "hedging_language": "Use of uncertain or qualifying terms to avoid direct statements",
            "url_patterns": "Presence and nature of links to external sources"
        }
    }
    
    return jsonify(categories)

@app.after_request
def after_request(response):
    # Add additional CORS headers to every response
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    # Start the Flask app with specific host binding to make it accessible
    print("Starting Enhanced Fake News Detection API server on http://127.0.0.1:5000")
    # Use 0.0.0.0 to allow external connections, but 127.0.0.1 for local-only
    app.run(host='0.0.0.0', port=5000, debug=True)