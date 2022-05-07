# This code analyzes The Cure Discography for pure exploration reasons.
# The data and features description can be found at https://www.kaggle.com/xvivancos/the-cure-discography

# TODO Set the environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# tell matplotlib how to plot pandas types
pd.plotting.register_matplotlib_converters()

# configure displaying arrays on the screen
desired_width = 420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 24)

# load the data
path_to_data = "D:\ML files\The_cure_discography.csv"
df = pd.read_csv(path_to_data)

# TODO Exploratory data analysis
# TODO 1. Look at data
# look at data in various ways

print(df.shape, df.head(), df.describe(), df.describe(include=object), sep='\n\n')
df.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)
print('\n\n', df.info())
# see top 5 most frequent key modes for The Cure
print('Key modes: \n', df["key_mode"].value_counts()[:4])

# What is the most danceable album?
albums_av_danceability = {}
for album in df["album_name"].unique():
    albums_av_danceability.update( { album : round(df.loc[df["album_name"] == album, "danceability"].mean(),2) } )
print('\nWhat is the average danceability of The Cure albums? \n')
for albumname, album_danceability in sorted(albums_av_danceability.items(), key=lambda item: item[1], reverse=True):
    print("{}'s danceability is {}".format(albumname.rjust(45), album_danceability))

most_danceable_album = [key for (key, value) in albums_av_danceability.items()
                        if value == max(albums_av_danceability.values())][0]
print("\nThe most danceable album is",most_danceable_album ,
      "with danceability = ", max(albums_av_danceability.values()))

print('\nIts tracks are:\n', df[(df["album_name"] == most_danceable_album)][["track_name", "danceability"]])
# print('\n', df.loc[df["album_name"] == "The Head On The Door", ["track_name","danceability"]]) - other way of doing it

print('\nAverage duration of a track in minutes:', round(df["duration_ms"].mean() / 60000, 2))

# TODO 2. Make univariate plots

# Plotting average danceability by album
plt.figure(figsize=(8,5))
plt.title("Average Danceability, by Album")

# here the color hue changes according to x-axis, which is not what we want:
# sns.barplot(x=df["album_name"].unique(), y=list(albums_av_danceability.values()),palette="Blues_d")

# building a function that uses the values as indices in the color palette to set the hue to y-axis:
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

x=list(albums_av_danceability.values())
y=df["album_name"].unique()
sns.barplot(x,y,palette=colors_from_values(x, "Blues_d"))
plt.xlabel("Danceability (1.0 is most danceable)")
plt.show()

# Plotting popularity by album

plt.figure(figsize=(10,6))
plt.title("Popularity of tracks, by Album")
# Show each observation with a scatterplot
sns.stripplot(x="track_popularity", y="album_name",
              data=df, dodge=True, alpha=.8, zorder=1)
# Show the conditional means
sns.pointplot(x="track_popularity", y="album_name",
              data=df, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)
plt.xlabel("Popularity")
plt.ylabel("")
plt.show()

# TODO 3. Consider correlations

# Compute the correlation matrix
corr_matrix = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 7))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.title("Correlation of The Cure discography parameters")
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


sns.lmplot(x="valence", y="danceability", data=df)
plt.xticks(rotation=60)
plt.title("Correlation of Positiveness and Danceability of a track")
plt.xlabel("Positiveness (1.0 is most positive)")
plt.ylabel("Danceability (1.0 is most danceable)")
plt.show()

# TODO 4. Check for missing values
# TODO 5. Check for outliers

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Distribution of parameters values')
sns.boxplot(ax=axes[0, 0], data=df, x='album_popularity')
sns.boxplot(ax=axes[0, 1], data=df, x='danceability')
sns.boxplot(ax=axes[0, 2], data=df, x='energy')
sns.boxplot(ax=axes[0, 3], data=df, x='loudness')
sns.boxplot(ax=axes[0, 4], data=df, x='speechiness')
sns.boxplot(ax=axes[1, 0], data=df, x='acousticness')
sns.boxplot(ax=axes[1, 1], data=df, x='liveness')
sns.boxplot(ax=axes[1, 2], data=df, x='valence')
sns.boxplot(ax=axes[1, 3], data=df, x='tempo')
sns.boxplot(ax=axes[1, 4], data=df, x='duration_ms')

plt.show()

# TODO Prediction

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error

# Obtain target and predictors
y = df.track_popularity
features = ['album_popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness','liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
# Select numeric columns only
# numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = df[features].copy()
# or like that: train_data.drop(['track_popularity'], axis=1, inplace=True)


# Break off test set from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])


# Use cross-validation to select parameters for a machine learning model
# TODO Can also change criterion='mae', min_samples_split=20, max_depth=7
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    pipe = Pipeline(steps=[('model', RandomForestRegressor(n_estimators, random_state=0))])
    several_scores = -1 * cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error') # here X and y or train?
    return several_scores.mean()

score_results = {}
for i in range(1,3):
    score_results[30 * i] = get_score(30 * i)
# print("MAE scores for n_estimators",score_results)

print("{:<15} {:<10}".format('n_estimators', 'Mean MAE score'))
for k, v in score_results.items():
    print("{:<15} {:.3}".format(k, v))

plt.plot(score_results.keys(), score_results.values())
plt.title("Which RandomForestRegressor is the best?")
plt.xlabel("Number of estimators")
plt.ylabel("MAE score")
plt.show()

n_estimators_best = [key for (key, value) in score_results.items()
                     if value == min(score_results.values())][0]
# Define a model
my_model = RandomForestRegressor(n_estimators=n_estimators_best, max_depth=8, random_state=0)

# Fit the model to the training data
my_model.fit(X_train, y_train)

# Generate test predictions
preds_test = my_model.predict(X_test)
# Save predictions in format used for competition scoring
# print("Try", [popularity for (id,popularity) in df["track_popularity"] if id in X_test.index])
output = pd.DataFrame({'ID': X_test.index,
                       'popularity_predicted': preds_test,})
# output.join(pd.DataFrame({df["ID"]:df["track_popularity"]}))

print("\nPredicted results\n", output)

output.to_csv('Track popularity predictions.csv', index=False)

# TODO 1. Categorical data transformation
# TODO 2. Create pipeline
# TODO 3. Split the data into training and validation / use cross-validation
# TODO 4. Create features
# TODO 5. Report error rate
