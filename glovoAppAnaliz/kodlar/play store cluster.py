# google colab ortamında çalıştırılmıştır
#!pip install transformers

import pandas as pd
import re
from google.colab import files

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns


# play store için
veri = pd.read_csv("/content/playstoretemizveri.csv")


# Vectorize the cleaned text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(veri["temiz icerik"])

# Perform K-means clustering
num_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
veri["Cluster"] = kmeans.fit_predict(X)

veri["Cluster"] = kmeans.labels_


veri.groupby("Cluster")["temiz icerik"].apply(lambda x: " ".join(x)).head()
# Download the vader_lexicon resource
import nltk

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
veri["sentiment"] = veri["temiz icerik"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Tokenize ve pad veriler
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(veri["temiz icerik"])
sequences = tokenizer.texts_to_sequences(veri["temiz icerik"])
X_pad = pad_sequences(sequences, maxlen=200)
# Model oluşturma
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128))
model.add(Dense(1, activation="sigmoid"))
# Model derleme
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# Model eğitme
model.fit(X_pad, veri["sentiment"], epochs=5, batch_size=32)

print("model eğitme tamamlandı")

# Display the reviews for each cluster to identify common themes
for i in range(num_clusters):
    print(f"Cluster {i}:")
    print(veri[veri["Cluster"] == i]["temiz icerik"].tolist())
    print("\n")

# Define UI and UX keywords
ui_keywords = [
    "tasarım",
    "tasarim",
    "arayuz",
    "arayüz",
    "duzen",
    "düzen",
    "dugme",
    "düğme",
    "renk",
    "simge",
    "ikon",
    "buton",
    "butonlar",
    "çerçeve",
    "çizim",
    "çizgi",
    "görsel",
    "dizayn",
    "tema",
    "stil",
    "font",
    "yazıtipi",
    "görünüm",
    "arayüz",
    "arayüzler",
    "ikonlar",
    "buton tasarımı",
    "menü",
    "menüler",
    "şablon",
    "düzenleme",
    "pencere",
    "anahat",
    "başlık",
    "logo",
    "banner",
    "sembol",
    "görsel tasarım",
    "arayüz tasarımı",
    "ekran",
    "ekran düzeni",
    "düzenleme araçları",
    "seçim kutusu",
    "seçim kutuları",
    "radyolar",
    "metin kutusu",
    "metin alanı",
    "simgeler",
    "temalar",
    "görseller",
    "çizimler",
    "çizim araçları",
    "simge setleri",
    "renk paleti",
]

ux_keywords = [
    "gezinme",
    "deneyim",
    "kullanilabilirlik",
    "kullanılabilirlik",
    "cokme",
    "çökme",
    "yavas",
    "yavaş",
    "hata",
    "geri bildirim",
    "kullanıcı",
    "kullanıcı deneyimi",
    "kullanici deneyimi",
    "dostu",
    "hız",
    "performans",
    "etkileşim",
    "geri butonu",
    "ileri butonu",
    "kullanıcı dostu",
    "erişilebilirlik",
    "kullanılabilir",
    "kullanılabilirliği",
    "kullanıcı memnuniyeti",
    "işlevsellik",
    "anlaşılır",
    "yönlendirme",
    "kullanım",
    "hızlı",
    "hizli",
    "yavaşlık",
    "çöküyor",
    "hata mesajı",
    "bug",
    "donma",
    "takılma",
    "iyileştirme",
    "düzeltme",
    "geri dönüş",
    "kullanım kolaylığı",
    "hızlanma",
    "gecikme",
    "akış",
    "kullanıcı geri bildirimi",
    "deneyim iyileştirme",
    "çökme raporu",
    "verimlilik",
    "ergonomi",
    "test",
    "analiz",
    "test etme",
    "yeniden düzenleme",
    "yeniden tasarlama",
    "navigasyon",
    "açılır menü",
    "acilir menu",
    "kullanıcı davranışı",
    "kullanıcı ihtiyaçları",
    "kullanıcı testleri",
    "kullanıcı geri dönüşleri",
]


def classify_complaint(text):
    if any(word in text for word in ui_keywords):
        return "UI"
    elif any(word in text for word in ux_keywords):
        return "UX"
    else:
        return "Other"


veri["category"] = veri["temiz icerik"].apply(classify_complaint)

ui_veri = veri[veri["category"] == "UI"]
ux_veri = veri[veri["category"] == "UX"]
# For UI complaints
X_ui = vectorizer.fit_transform(ui_veri["temiz icerik"])
kmeans_ui = KMeans(n_clusters=3, random_state=42)
kmeans_ui.fit(X_ui)
ui_veri["ui_cluster"] = kmeans_ui.labels_
# For UX complaints
X_ux = vectorizer.fit_transform(ux_veri["temiz icerik"])
kmeans_ux = KMeans(n_clusters=3, random_state=42)
kmeans_ux.fit(X_ux)
ux_veri["ux_cluster"] = kmeans_ux.labels_

# UI Complaints Clustering Visualization
sns.countplot(x="ui_cluster", data=ui_veri)
plt.title("UI Complaints Clustering")
plt.show()
# UX Complaints Clustering Visualization
sns.countplot(x="ux_cluster", data=ux_veri)
plt.title("UX Complaints Clustering")
plt.show()
# Display some example complaints from each UI cluster
for i in range(3):
    print(f"UI Cluster {i}:")
    print(ui_veri[ui_veri["ui_cluster"] == i]["temiz icerik"].tolist())
    print("\n")
# Display some example complaints from each UX cluster
for i in range(3):
    print(f"UX Cluster {i}:")
    print(ux_veri[ux_veri["ux_cluster"] == i]["temiz icerik"].tolist())
    print("\n")


# Reduce dimensionality using PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

# Add PCA results to the DataFrame
veri["PCA1"] = X_pca[:, 0]
veri["PCA2"] = X_pca[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(veri["PCA1"], veri["PCA2"], c=veri["Cluster"], cmap="viridis")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("K-means Clustering of Reviews (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

veri.to_csv("playstorecluster.csv")
files.download("playstorecluster.csv")
