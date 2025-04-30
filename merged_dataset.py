import pandas as pd
import os

# Create output directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Function to load and label a dataset
def load_and_label_dataset(file_path, label_value, is_fake):
    """
    Load a dataset from file_path and add a label column.
    
    Parameters:
    - file_path: Path to the CSV file
    - label_value: 0 for real news, 1 for fake news
    - is_fake: Text label for logging purposes
    
    Returns:
    - Pandas DataFrame with added label column
    """
    print(f"Loading {is_fake} news dataset from {file_path}...")
    try:
        # Try to load the dataset
        df = pd.read_csv(file_path)
        
        # Add label column (0 for real news, 1 for fake news)
        df['label'] = label_value
        
        print(f"Successfully loaded {len(df)} {is_fake} news articles")
        return df
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

# Load fake news dataset
fake_news_path = './Fake.csv'  # Modify this path as needed
fake_df = load_and_label_dataset(fake_news_path, 1, "fake")

# Load real news dataset
real_news_path = 'True.csv'  # Modify this path as needed
real_df = load_and_label_dataset(real_news_path, 0, "real")

# Check if both datasets were loaded successfully
if fake_df is not None and real_df is not None:
    # Ensure compatible columns by finding common columns
    common_columns = list(set(fake_df.columns).intersection(set(real_df.columns)))
    
    # Make sure 'label' is in the common columns
    if 'label' not in common_columns:
        common_columns.append('label')
        
    print(f"Common columns: {common_columns}")
    
    # Merge datasets
    merged_df = pd.concat([fake_df[common_columns], real_df[common_columns]], ignore_index=True)
    
    # Shuffle the merged dataset
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save merged dataset
    output_path = 'data/news.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path} with {len(merged_df)} articles")
    
    # Show class distribution
    print("\nClass distribution:")
    print(merged_df['label'].value_counts())
    
    # Show a few samples
    print("\nSample data:")
    print(merged_df.head())

elif fake_df is None and real_df is None:
    print("\nNo datasets were loaded. Creating a synthetic dataset for demonstration...")
    
    # Create a synthetic dataset
    from sklearn.datasets import make_classification
    import numpy as np
    
    # Generate synthetic data
    n_samples = 10000
    
    # Create random titles and text
    def generate_random_text(n_words, is_fake):
        fake_words = ['shocking', 'unbelievable', 'secret', 'conspiracy', 'scandal', 
                     'miracle', 'breakthrough', 'revolution', 'exposed', 'hidden']
        real_words = ['research', 'study', 'analysis', 'report', 'data', 
                     'experts', 'evidence', 'investigation', 'official', 'findings']
        
        word_list = fake_words if is_fake else real_words
        other_words = ['news', 'today', 'world', 'people', 'government', 'science', 
                      'health', 'technology', 'business', 'politics']
        
        text = []
        for _ in range(n_words):
            # 30% chance to use domain-specific words, 70% other words
            if np.random.random() < 0.3:
                text.append(np.random.choice(word_list))
            else:
                text.append(np.random.choice(other_words))
                
        return ' '.join(text)
    
    # Create synthetic dataset
    synthetic_df = pd.DataFrame()
    
    # Generate 50% fake and 50% real news
    labels = np.random.choice([0, 1], size=n_samples)
    
    titles = []
    texts = []
    authors = []
    
    for i in range(n_samples):
        is_fake = (labels[i] == 1)
        titles.append(generate_random_text(5, is_fake))
        texts.append(generate_random_text(50, is_fake))
        authors.append(f"Author{np.random.randint(1, 100)}")
    
    synthetic_df['title'] = titles
    synthetic_df['text'] = texts
    synthetic_df['author'] = authors
    synthetic_df['label'] = labels
    
    # Save synthetic dataset
    output_path = 'data/news.csv'
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path} with {len(synthetic_df)} articles")
    
    # Show class distribution
    print("\nClass distribution:")
    print(synthetic_df['label'].value_counts())
    
    # Show a few samples
    print("\nSample data:")
    print(synthetic_df.head())
else:
    if fake_df is None:
        print("Failed to load fake news dataset.")
    if real_df is None:
        print("Failed to load real news dataset.")