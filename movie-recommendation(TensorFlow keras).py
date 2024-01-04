# TensorFlow ve gerekli kütüphaneleri içe aktarın
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# Netflix veri setini yükleyin
file_path = "your file location"
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
def get_recommendations(title):
    """
    Verilen bir film adına göre, benzer içeriklere sahip diğer filmleri öneren fonksiyon.

    Parametreler:
        title (str): Öneri yapılacak filmin adı.

    Dönüş:
        pandas.Series: Verilen filmin benzer içeriklere sahip diğer filmlerin adlarını içeren bir Seri.

    Kullanım:
        Kullanıcıdan bir film adı alın:
        user_input = input("Bir film adı girin: ")

        Öneri yapın ve sonuçları gösterin:
        recommended_titles = get_recommendations(user_input)
        print(f"{user_input} filminin benzer içeriklere önerileri:")
        print(recommended_titles)
    """
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
