# play store için
import pandas as pd
from google_play_scraper import Sort, reviews

app_package_name = "com.glovo"  # Uygulamanın paket adını buraya yazın


all_reviews = []

for _ in range(60):
    result, _ = reviews(
        app_package_name,
        lang="it",
        sort=Sort.NEWEST,
        count=20000,  # Bir seferde çekilecek yorum sayısı
    )
    all_reviews.extend(result)

# Alınan yorumları bir DataFrame'e dönüştür ve CSV dosyasına kaydet

reviews_df = pd.DataFrame(all_reviews)
reviews_df.to_csv("Playstore_glovo.csv", index=False)

# app store için

from app_store_scraper import AppStore

# glovo yorum çekme

glovo = AppStore(
    country="it", app_name="glovo-food-delivery-and-more", app_id="951812684"
)
glovo.review(how_many=179000)

# Yorumları DataFrame'e dönüştür ve kaydet

df = pd.DataFrame(glovo.reviews)
df.to_csv("appstore_glovo.csv", index=False)
