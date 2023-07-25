import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Netflix veri setini yükleyin
file_path = "C:/Users/akcol/Desktop/netflix_titles.csv"
data = pd.read_csv(file_path)

# Boş değerleri "Unknown" ile doldurun
data.fillna("Unknown", inplace=True)

# İçerikleri temsil etmek için birleştirilmiş bir özellik oluşturun
def combine_features(row):
    return row["type"] + " " + row["title"] + " " + row["director"] + " " + row["cast"] + " " + row["country"] + " " + row["listed_in"]

data["combined_features"] = data.apply(combine_features, axis=1)

# Özellikler matrisini oluşturun
cv = CountVectorizer()
features_matrix = cv.fit_transform(data["combined_features"])

# Kozinüs benzerliği hesaplayın
cosine_sim = cosine_similarity(features_matrix)

# Öneri fonksiyonunu tanımlayın
def get_recommendations(title):
    title = title.lower()  # Girişteki filmin adını küçük harflere çevirin
    idx = data[data["title"].str.lower() == title].index[0]
    similar_scores = list(enumerate(cosine_sim[idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:11]  # İlk öğe, verilen içeriği kendisi olacak şekilde çıkartın
    similar_indices = [score[0] for score in similar_scores]
    return data["title"].iloc[similar_indices]

# Kullanıcıdan bir film adı alın
user_input = input("Bir film adı girin: ")

# Öneri yapın
recommended_titles = get_recommendations(user_input)

# Kullanıcıya önerileri gösterin
print(f"{user_input} filminin benzer içeriklere önerileri:")
print(recommended_titles)
