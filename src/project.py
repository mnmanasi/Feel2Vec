import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_nrc_lexicon():
    emotion_lexicon = defaultdict(list)
    lexicon_file = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    
    with open(lexicon_file, 'r') as file:
        for line in file:
            word, emotion, association = line.strip().split('\t')
            if int(association) == 1:
                emotion_lexicon[word].append(emotion)
    return emotion_lexicon

def preprocess_lyrics(lyrics):
    punctuations = '\'"\\,<>./?@#$%^&*_~/!()-[]{};:'
    # Remove punctuation and any content within brackets (e.g., [chorus])
    lyrics = ''.join([char for char in lyrics if char not in punctuations])
    lyrics = lyrics.split('[')[0]  # Remove anything between [ and ]
    
    tokens = nltk.word_tokenize(lyrics.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def normalize_distribution(distribution):
    total = sum(distribution.values())
    if total > 0:
        return {emotion: value / total for emotion, value in distribution.items()}
    else:
        return {emotion: 0 for emotion in distribution.keys()}  # Return zero for all if total is zero

def assign_emotion_to_song(lyrics, emotion_lexicon, tfidf_vector, tfidf_words):
    emotion_count = defaultdict(int)
    sentiment_count = defaultdict(int)
    tokens = preprocess_lyrics(lyrics)
    
    # For each token, check if it's in the emotion lexicon
    for word in tokens:
        if word in emotion_lexicon:
            # Use np.where to get the index of the word in tfidf_words
            word_index_array = np.where(tfidf_words == word)[0]
            if len(word_index_array) > 0:
                word_index = word_index_array[0]
                word_tfidf_score = tfidf_vector[word_index]
                for emotion in emotion_lexicon[word]:
                    if emotion in ['positive', 'negative']:
                        sentiment_count[emotion] += word_tfidf_score
                    else:
                        emotion_count[emotion] += word_tfidf_score
    
    # Normalize distributions
    normalized_emotion_count = normalize_distribution(emotion_count)
    normalized_sentiment_count = normalize_distribution(sentiment_count)

    # Determine dominant emotion and dominant sentiment
    dominant_emotion = max(normalized_emotion_count, key=normalized_emotion_count.get) if normalized_emotion_count else None
    dominant_sentiment = max(normalized_sentiment_count, key=normalized_sentiment_count.get) if normalized_sentiment_count else None
    
    return dominant_emotion, dominant_sentiment, normalized_emotion_count, normalized_sentiment_count

def assign_emotions_to_dataset(df, emotion_lexicon, tfidf_scores, tfidf_words):
    emotion_results = []
    
    # Loop over the songs in the dataset
    for index, row in df.iterrows():
        song_id = row['id']
        artist = row['artist']
        title = row['title']
        lyrics = row['lyrics']
        
        # Get the corresponding TF-IDF vector for the song
        if index < len(tfidf_scores):
            tfidf_vector = tfidf_scores[index]
        else:
            continue  # Skip if the index is out of bounds
        
        dominant_emotion, dominant_sentiment, emotion_count, sentiment_count = assign_emotion_to_song(lyrics, emotion_lexicon, tfidf_vector, tfidf_words)
        
        emotion_results.append({
            'song_id': song_id,
            'artist': artist,
            'title': title,
            'dominant_emotion': dominant_emotion,
            'dominant_sentiment': dominant_sentiment,
            'emotion_distribution': emotion_count,
            'sentiment_distribution': sentiment_count
        })
    
    return pd.DataFrame(emotion_results)

if __name__ == "__main__":
    # df = pd.read_csv('song_lyrics.csv')
    # df = df[df['language'] == 'en']
    # df = df.dropna()
    # sample_df = df.sample(n = 1000)
    # sample_df.to_csv('sample_df.csv', index=False)

    # Load the dataset
    df_songs = pd.read_csv('sample_df.csv')
    df_songs = df_songs.dropna()

    # Load NRC emotion lexicon
    emotion_lexicon = load_nrc_lexicon()

    # Preprocess the lyrics and create a list of all lyrics
    lyrics_list = df_songs['lyrics'].apply(lambda x: ' '.join(preprocess_lyrics(x)))

    # Compute TF-IDF for the lyrics
    vectorizer = TfidfVectorizer()
    tfidf_scores = vectorizer.fit_transform(lyrics_list).toarray()  # Convert sparse matrix to array
    tfidf_words = vectorizer.get_feature_names_out()  # List of words
    
    # Assign emotions to all songs
    emotion_results_df = assign_emotions_to_dataset(df_songs, emotion_lexicon, tfidf_scores, tfidf_words)
    emotion_results_df = emotion_results_df.dropna()
    
    # Save the results to a CSV file
    emotion_results_df.to_csv('emotion_assigned_songs.csv', index=False)
