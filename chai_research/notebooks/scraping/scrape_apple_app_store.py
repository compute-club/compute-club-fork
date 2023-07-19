import pandas as pd
import numpy as np

from app_store_scraper import AppStore
chai = AppStore(country='us', app_name='chai-chat-with-ai-bots', app_id = '1544750895')

chai.review(how_many=50000)

df = pd.DataFrame(np.array(chai.reviews),columns=['review'])
df2 = df.join(pd.DataFrame(df.pop('review').tolist()))
print(df2.head())

df2.to_csv('chai-apple-app-store-reviews.csv')