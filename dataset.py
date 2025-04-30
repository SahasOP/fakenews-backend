import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import time
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

# Feature engineering function
def extract_features(df):
    # Create new features
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    df['sentence_count'] = df['text'].apply(lambda x: len(str(x).split('.')))
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0)
    
    # Extract POS features (basic version)
    def get_pos_features(text):
        if not isinstance(text, str) or not text.strip():
            return 0, 0, 0
        
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        num_nouns = len([word for word, pos in pos_tags if pos.startswith('N')])
        num_verbs = len([word for word, pos in pos_tags if pos.startswith('V')])
        num_adj = len([word for word, pos in pos_tags if pos.startswith('J')])
        
        # Calculate ratios
        total = len(tokens) if len(tokens) > 0 else 1
        return num_nouns/total, num_verbs/total, num_adj/total
    
    # Apply POS features extraction to a sample for performance reasons
    sample_size = min(5000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    pos_features = sample_df['processed_text'].apply(get_pos_features)
    sample_df['noun_ratio'], sample_df['verb_ratio'], sample_df['adj_ratio'] = zip(*pos_features)
    
    # Join back with original dataframe
    df = df.join(sample_df[['noun_ratio', 'verb_ratio', 'adj_ratio']], how='left')
    df[['noun_ratio', 'verb_ratio', 'adj_ratio']] = df[['noun_ratio', 'verb_ratio', 'adj_ratio']].fillna(0)
    
    return df

# Load dataset
print("Loading dataset...")
dataset_path = './data/news.csv'  # Replace with your Kaggle dataset path
try:
    # Try to load the dataset first
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with {len(df)} rows")
    
    # Display basic info
    print("\nDataset information:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    
    # Check for class balance
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Handle missing values
    df = df.fillna('')
    
    # Set up our target and merge title with text for better context
    df['text'] = df['title'] + ' ' + df['text']
    
    # Clean and preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature engineering
    print("\nExtracting features...")
    df = extract_features(df)
    
    # Split dataset into training and testing sets (80:20)
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
except FileNotFoundError:
    print(f"Dataset file not found: {dataset_path}")
    print("Generating synthetic dataset for demonstration purposes...")
    
    # Create a synthetic dataset if the real one isn't available
    def create_synthetic_dataset(size=10000):
        # Common words/phrases in fake news
        fake_phrases = [
            "shocking truth", "they don't want you to know", "conspiracy", "cover-up",
            "secret", "you won't believe", "doctors hate", "miracle cure", "breakthrough",
            "shocking revelation", "government doesn't want you to know", "what they're hiding"
        ]
        
        # Words/phrases more common in legitimate news
        real_phrases = [
            "according to research", "study finds", "data suggests", "experts say",
            "evidence shows", "analysis reveals", "researchers have found", "scientists conclude",
            "official statement", "investigation shows", "preliminary results", "statistics indicate"
        ]
        
        # Common topics
        topics = [
            "politics", "health", "economy", "technology", "science", "environment",
            "education", "international relations", "entertainment", "sports"
        ]
        
        # Generate fake news content
        fake_articles = []
        for _ in range(size // 2):
            topic = np.random.choice(topics)
            fake_phrase = np.random.choice(fake_phrases)
            
            title = f"{fake_phrase} about {topic}"
            author = f"Author{np.random.randint(1, 100)}"
            
            # Generate a longer fake article
            paragraphs = []
            for _ in range(np.random.randint(3, 8)):
                para = f"{np.random.choice(fake_phrases)} {topic} {np.random.choice(['revealed', 'discovered', 'exposed'])}. "
                para += f"This {np.random.choice(['shocking', 'incredible', 'unbelievable'])} {topic} story will {np.random.choice(['amaze you', 'shock you', 'change everything'])}. "
                para += f"The {np.random.choice(['elites', 'government', 'media'])} doesn't want you to know this truth about {topic}."
                paragraphs.append(para)
            
            text = ' '.join(paragraphs)
            fake_articles.append((title, author, text, 1))  # 1 for fake
        
        # Generate real news content
        real_articles = []
        for _ in range(size // 2):
            topic = np.random.choice(topics)
            real_phrase = np.random.choice(real_phrases)
            
            title = f"{real_phrase} about {topic} trends"
            author = f"Reporter{np.random.randint(1, 100)}"
            
            # Generate a longer real article
            paragraphs = []
            for _ in range(np.random.randint(3, 8)):
                para = f"{np.random.choice(real_phrases)} regarding {topic} {np.random.choice(['today', 'this week', 'this month'])}. "
                para += f"The {np.random.choice(['data', 'research', 'evidence'])} {np.random.choice(['indicates', 'suggests', 'shows'])} that {topic} {np.random.choice(['is improving', 'is changing', 'requires attention'])}. "
                para += f"According to {np.random.choice(['experts', 'researchers', 'officials'])}, the implications for {topic} are significant."
                paragraphs.append(para)
                
            text = ' '.join(paragraphs)
            real_articles.append((title, author, text, 0))  # 0 for real
        
        # Combine and shuffle
        all_articles = fake_articles + real_articles
        np.random.shuffle(all_articles)
        
        # Create dataframe
        titles, authors, texts, labels = zip(*all_articles)
        df = pd.DataFrame({
            'id': range(len(all_articles)),
            'title': titles,
            'author': authors,
            'text': texts,
            'label': labels
        })
        
        return df
    
    # Create synthetic dataset
    df = create_synthetic_dataset(size=10000)
    
    # Preprocess the synthetic data
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature engineering
    print("\nExtracting features...")
    df = extract_features(df)
    
    # Split dataset into training and testing sets (80:20)
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
# Vectorization for traditional ML models
print("\nVectorizing text data...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Count Vectorization for comparison
count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# For neural networks, tokenize and pad sequences
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Function to evaluate and visualize model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, vectorizer_name="TF-IDF"):
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For probability-based ROC curve
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_pred_proba = model.decision_function(X_test)
        except:
            y_pred_proba = y_pred  # Fallback
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    # Store results
    result = {
        'Model': f"{model_name} ({vectorizer_name})",
        'Accuracy': accuracy,
        'Precision': precision, 
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': time_taken
    }
    
    # Print results
    print(f"\n{model_name} ({vectorizer_name}) Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {time_taken:.2f} seconds")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store additional data for visualization
    result['Confusion Matrix'] = cm
    result['y_pred_proba'] = y_pred_proba
    result['model_object'] = model
    
    return result

# For LSTM evaluation
def evaluate_lstm(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # Define LSTM model
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Train model with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    # Store results
    result = {
        'Model': "LSTM",
        'Accuracy': accuracy,
        'Precision': precision, 
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': time_taken,
        'History': history
    }
    
    # Print results
    print("\nLSTM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {time_taken:.2f} seconds")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store additional data for visualization
    result['Confusion Matrix'] = cm
    result['y_pred_proba'] = y_pred_proba
    result['model_object'] = model
    
    return result

# Define models
print("\nTraining and evaluating models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    # 'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Run all models and collect results
results = []

# TF-IDF Vectorizer
for model_name, model in models.items():
    print(f"\nTraining {model_name} with TF-IDF features...")
    result = evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test, model_name)
    results.append(result)

# Try Count Vectorizer for comparison with some models
for model_name in ['Naive Bayes', 'Decision Tree']:
    model = models[model_name]
    print(f"\nTraining {model_name} with Count Vectorizer features...")
    result = evaluate_model(model, X_train_count, X_test_count, y_train, y_test, model_name, "Count")
    results.append(result)

# Train LSTM model (if sufficient resources)
try:
    print("\nTraining LSTM model...")
    lstm_result = evaluate_lstm(X_train_pad, X_test_pad, y_train.values, y_test.values)
    results.append(lstm_result)
    has_lstm = True
except Exception as e:
    print(f"Error training LSTM: {str(e)}")
    has_lstm = False

# Create results dataframe for comparison
results_df = pd.DataFrame([
    {
        'Model': r['Model'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1 Score': r['F1 Score'],
        'Training Time': r['Training Time']
    } for r in results
])

print("\nModel Comparison:")
print(results_df.sort_values('F1 Score', ascending=False))

# Data Visualizations
print("\nGenerating visualizations...")

# 1. Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Class Distribution in Dataset')
plt.xlabel('Class (0: Real, 1: Fake)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# 2. Word Count Distribution by Class
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='word_count', hue='label', bins=30, kde=True)
plt.title('Word Count Distribution by Class')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend(['Real', 'Fake'])
plt.savefig('word_count_distribution.png')
plt.close()

# 3. Model Performance Comparison
plt.figure(figsize=(14, 8))
perf_metrics = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'], 
                           var_name='Metric', value_name='Score')
sns.barplot(x='Model', y='Score', hue='Metric', data=perf_metrics)
plt.title('Model Performance Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# 4. Training Time Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Training Time', data=results_df)
plt.title('Training Time Comparison (seconds)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('training_time.png')
plt.close()

# 5. Confusion Matrices for Top Models
plt.figure(figsize=(16, 12))
top_models = results_df.sort_values('F1 Score', ascending=False).head(4)['Model'].tolist()

for i, model_name in enumerate(top_models):
    model_result = next(r for r in results if r['Model'] == model_name)
    cm = model_result['Confusion Matrix']
    
    plt.subplot(2, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# 6. ROC Curves
plt.figure(figsize=(10, 8))

for model_result in results:
    try:
        fpr, tpr, _ = roc_curve(y_test, model_result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_result['Model']} (AUC = {roc_auc:.3f})")
    except:
        pass

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

# 7. Feature Importance for Random Forest
try:
    rf_result = next(r for r in results if 'Random Forest' in r['Model'])
    rf_model = rf_result['model_object']
    
    # Get feature names and importances
    feature_names = tfidf_vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_
    
    # Sort and get top 20 features
    indices = np.argsort(importances)[-20:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features for Random Forest Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
except Exception as e:
    print(f"Could not generate feature importance plot: {str(e)}")

# 8. Learning Curves for LSTM (if available)
if has_lstm:
    lstm_history = lstm_result['History'].history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history['accuracy'])
    plt.plot(lstm_history['val_accuracy'])
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history['loss'])
    plt.plot(lstm_history['val_loss'])
    plt.title('LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('lstm_learning_curves.png')
    plt.close()

# 9. POS Tag Distributions (if available)
if 'noun_ratio' in df.columns:
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x='label', y='noun_ratio', data=df)
    plt.title('Noun Ratio by Class')
    plt.xlabel('Class (0: Real, 1: Fake)')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='label', y='verb_ratio', data=df)
    plt.title('Verb Ratio by Class')
    plt.xlabel('Class (0: Real, 1: Fake)')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='label', y='adj_ratio', data=df)
    plt.title('Adjective Ratio by Class')
    plt.xlabel('Class (0: Real, 1: Fake)')
    
    plt.tight_layout()
    plt.savefig('pos_distributions.png')
    plt.close()

# 10. Character Count vs Word Count Scatter Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='word_count', y='char_count', hue='label', data=df.sample(min(5000, len(df)), random_state=42))
plt.title('Character Count vs Word Count by Class')
plt.xlabel('Word Count')
plt.ylabel('Character Count')
plt.legend(['Real', 'Fake'])
plt.savefig('char_word_scatter.png')
plt.close()

# Save the best model
best_model_name = results_df.sort_values('F1 Score', ascending=False).iloc[0]['Model']
best_model_result = next(r for r in results if r['Model'] == best_model_name)
best_model = best_model_result['model_object']

print(f"\nSaving the best model: {best_model_name}")
if "LSTM" in best_model_name:
    best_model.save('best_fake_news_model')
    # Also save the tokenizer for LSTM
    dump(tokenizer, 'tokenizer.joblib')
else:
    dump(best_model, 'best_fake_news_model.joblib')
    # Save the vectorizer
    if "Count" in best_model_name:
        dump(count_vectorizer, 'vectorizer.joblib')
    else:
        dump(tfidf_vectorizer, 'vectorizer.joblib')
