MOVIE RATING PREDICTION WITH PYTHON/Codesoft

🎬 IMDb Movies Analysis Adventure 🎥

Lights, camera, action! Let's dive into the captivating world of IMDb movies in India! 🍿

Required Libraries 📚

First things first, make sure you've got your libraries onboard:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
Dataset Exploration 🕵️‍♂️

Our journey begins with exploring the dataset:

# Loading the dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Checking the first five records
df.head()

# Discovering the dataset's dimensions
df.shape

# Unveiling dataset information
df.info()

# Scanning for missing treasures
df.isna().sum()
Data Cleaning Expedition 🧹

We embark on a quest to clean our data:


# Dropping treasures with missing gems
df.dropna(subset=['Name', 'Year', 'Duration', 'Votes', 'Rating'], inplace=True)

# Banishing duplicate treasures
df.drop_duplicates(subset=['Name', 'Year', 'Director'], keep='first', inplace=True)

# Transforming treasures for better readability
df['Year'] = df['Year'].str.strip('()').astype(int)
df['Duration'] = df['Duration'].str.replace(r' min', '').astype(int)
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)
Data Exploration Odyssey 🌌

Our journey continues with exploration:

# Unveiling the cleaned dataset
df.info()

# Plotting the number of movies released each year
# A visual delight!
plt.figure(figsize=(18, 9))
# ... and so on
Model Training Quest 🏹

We train our models to uncover hidden insights:

# Gathering training data
X = df[['Year', 'Duration', 'Votes']]
y = df['Rating']

# Splitting our treasures for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=231)

# Initiating the linear regression quest
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, pred_lr)
print(f'R-squared score (Linear Regression): {r2_lr}')

# And so on, the adventure continues with KNN, SGDRegressor, RandomForestRegressor, and GradientBoostingRegressor!
Conclusion 🏆

Our expedition through the realm of IMDb movies in India has been nothing short of exhilarating! We've cleaned, explored, and ventured into the world of predictive modeling. As we conclude this adventure, we've unlocked valuable insights into the realm of movie ratings. But fear not, for our data exploration journey has merely begun! 🚀

