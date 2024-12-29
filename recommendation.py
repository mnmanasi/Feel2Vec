import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def get_user_song_details(df_songs):
    user_songs = []
    num_songs = int(input("How many songs do you want to input? "))
    
    for i in range(num_songs):
        title = input(f"Enter the title for song {i+1}: ")
        artist = input(f"Enter the artist for song {i+1}: ")

        # Search for the song in the dataset by title and artist
        song_row = df_songs[(df_songs['title'].str.lower() == title.lower()) & 
                            (df_songs['artist'].str.lower() == artist.lower())]
        
        if not song_row.empty:
            user_songs.append(song_row['lyrics'].values[0])
        else:
            print(f"Song '{title}' by {artist} not found in the dataset.")
    
    return user_songs

def preprocess_user_songs(user_songs):
    return [' '.join(preprocess_lyrics(song)) for song in user_songs]

# Transform the user songs into TF-IDF vectors using the same vectorizer
def vectorize_user_songs(user_songs, vectorizer):
    user_tfidf_vectors = vectorizer.transform(user_songs).toarray()
    return user_tfidf_vectors

def find_similar_songs(user_tfidf_vectors, dataset_tfidf_vectors, df_songs, top_n=5):
    recommendations = []
    # Compute cosine similarity between user songs and dataset songs
    for user_vector in user_tfidf_vectors:
        similarity_scores = cosine_similarity([user_vector], dataset_tfidf_vectors)[0]
        # Get indices of top-n similar songs
        similar_song_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        # Append the top-n recommendations
        for idx in similar_song_indices:
            recommendations.append({
                'song_id': df_songs.iloc[idx]['id'],
                'title': df_songs.iloc[idx]['title'],
                'artist': df_songs.iloc[idx]['artist'],
                'similarity_score': similarity_scores[idx]
            })
    
    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    df_songs = pd.read_csv('sample_df.csv')
    df_songs = df_songs.dropna(subset=['lyrics'])  # Ensure we have lyrics
    emotion_lexicon = load_nrc_lexicon()

    # Preprocess and vectorize the lyrics of the existing dataset
    lyrics_list = df_songs['lyrics'].apply(lambda x: ' '.join(preprocess_lyrics(x)))
    vectorizer = TfidfVectorizer()
    dataset_tfidf_vectors = vectorizer.fit_transform(lyrics_list).toarray()
    
    # Accept user input for song title and artist, then fetch lyrics
    user_songs = get_user_song_details(df_songs)
    
    if user_songs:
        processed_user_songs = preprocess_user_songs(user_songs)
        user_tfidf_vectors = vectorize_user_songs(processed_user_songs, vectorizer)
    
        recommendations_df = find_similar_songs(user_tfidf_vectors, dataset_tfidf_vectors, df_songs)
    
        print("Recommended Songs:")
        print(recommendations_df)
    else:
        print("No valid songs were found based on the input.")


