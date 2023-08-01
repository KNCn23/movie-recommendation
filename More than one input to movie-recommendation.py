# TensorFlow ve gerekli kütüphaneleri içe aktarın
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["combined_features"])
sequences = tokenizer.texts_to_sequences(data["combined_features"])
features_matrix = pad_sequences(sequences)

# Kozinüs benzerliği hesaplayın
cosine_sim = cosine_similarity(features_matrix)

# Öneri fonksiyonunu tanımlayın
def get_common_recommendations_for_multiple_titles(titles):
    titles = [title.lower() for title in titles]  # Girişteki film adlarını küçük harflere çevirin

    # Girişteki filmlerin indekslerini bulun
    indices = [data[data["title"].str.lower() == title].index[0] for title in titles]

    # Filmlerin içeriklerini birleştirin
    combined_features = [data.iloc[idx]["combined_features"] for idx in indices]

    # Birleştirilmiş özellik matrisini oluşturun
    sequences = tokenizer.texts_to_sequences(combined_features)
    features_matrix = pad_sequences(sequences)

    # Kozinüs benzerliğini hesaplayın ve öneri yapın
    common_recommendation_scores = sum(cosine_sim[indices]) / len(titles)  # Ortalama benzerlikleri hesaplayın
    similar_indices = common_recommendation_scores.argsort()[:-11:-1]
    recommended_titles = data["title"].iloc[similar_indices]

    return recommended_titles

# Kullanıcıdan birden fazla film adı alın
user_inputs = []
while True:
    user_input = input("Bir film adı girin (Çıkmak için 'Öneride Bulun' yazın): ")
    if user_input.lower() == "öneride bulun":
        break
    user_inputs.append(user_input)

# Öneri yapın
recommended_titles = get_common_recommendations_for_multiple_titles(user_inputs)

# Kullanıcıya ortak öneriyi gösterin
print("Girilen filmlere göre ortak öneri:")
print(recommended_titles)
