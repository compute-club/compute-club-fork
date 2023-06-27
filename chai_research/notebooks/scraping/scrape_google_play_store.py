from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd

result = reviews_all(
    'com.Beauchamp.Messenger.external',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
    filter_score_with=5 # defaults to None(means all score)
)

# Convert to DataFrame
df = pd.DataFrame(result)

# Write to CSV
df.to_csv('chai-google-play-store-reviews.csv', index=False)