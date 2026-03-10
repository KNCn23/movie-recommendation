import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_recommendations(title, df):
    # Boş açıklamaları temizle
    df['description'] = df['description'].fillna('')
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    # Cosine Similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Film başlıklarını indeksle eşleştir
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in indices:
        return f"Film '{title}' bulunamadı."
        
    idx = indices[title]
    
    # Benzerlik puanlarını al
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Puanlara göre sırala
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # En benzer 10 filmi al (kendisi hariç)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

if __name__ == "__main__":
    try:
        df = load_data('netflix_titles.csv')
        print("🎬 Netflix Film Öneri Sistemi")
        title = "Inception" # Örnek
        if title in df['title'].values:
            print(f"\n'{title}' filmi için öneriler:")
            print(get_recommendations(title, df))
        else:
            print(f"\nÖrnek film '{title}' bulunamadı, ilk filmi deniyoruz:")
            first_movie = df['title'].iloc[0]
            print(f"'{first_movie}' için öneriler:")
            print(get_recommendations(first_movie, df))
    except FileNotFoundError:
        print("Hata: netflix_titles.csv dosyası bulunamadı.")
