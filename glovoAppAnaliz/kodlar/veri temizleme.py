# google colab ortamında çalıştırılmıştır

import pandas as pd
import re
from google.colab import files

# play store için
veri = pd.read_csv("/content/Playstore_glovo.csv")


def temizle(icerik):
    icerik = re.sub("#[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+", " ", icerik)  # Hashtag'leri temizle
    icerik = re.sub("\\n", " ", icerik)  # Yeni satır karakterlerini boşlukla değiştir
    icerik = re.sub("@[\S]*", " ", icerik)  # Kullanıcı etiketlerini temizle
    icerik = re.sub("https?:\/\/\S+", " ", icerik)  # URL'leri temizle
    icerik = icerik.lower()  # Tüm harfleri küçük harfe çevir
    icerik = re.sub(
        "[^a-zA-ZÀ-ÖØ-öø-ÿ0-9]", " ", icerik
    )  # Harfler ve sayılar dışındaki karakterleri boşlukla değiştir
    icerik = re.sub("^[\s]+|[\s]+$", " ", icerik)  # Baş ve sondaki boşlukları temizle
    return icerik


veri["temiz icerik"] = veri["content"].apply(temizle)
veri.to_csv("playstoretemizveri.csv")
files.download("playstoretemizveri.csv")

# app store için

df = pd.read_csv("/content/appstore_glovo.csv")

df["temiz icerik"] = df["review"].apply(temizle)
df.to_csv("appstoretemizveri.csv")
files.download("appstoretemizveri.csv")
