# https://www.kaggle.com/xvivancos/the-cure-discography

# Popularity and audio features for every song and album:

# track_popularity. The value will be between 0 and 100, with 100 being the most popular.
# duration_ms. The duration of the track in milliseconds.
# valence. A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
# danceability. Danceability describes how suitable a track is for dancing based on a combination of musical elements
# including tempo, rhythm stability, beat strength, and overall regularity.A value of 0.0 is least danceable and 1.0 is
# most danceable.
# energy. Represents a perceptual measure of intensity and activity (from 0.0 to 1.0).
# acousticness. A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
# loudness. The overall loudness of a track in decibels (typical range between -60 and 0 db).
# speechiness. Speechiness detects the presence of spoken words in a track. The more exclusively speech-like
# the recording,the closer to 1.0 the attribute value.
# instrumentalness. Predicts whether a track contains no vocals.The closer the instrumentalness value is to 1.0,
# the greater likelihood the track contains no vocal content.
# liveness. Detects the presence of an audience in the recording. A value above 0.8 provides strong likelihood
# that the track is live.
# key_mode. The key the track is in.

import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

desired_width = 420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 24)

path_to_data = "D:\ML files\The_cure_discography.csv"
cure_data = pd.read_csv(path_to_data)

print('\n', cure_data.shape)
print('\n', cure_data.columns)
print('\n', cure_data.head(7))
print('\n', cure_data.tail(3))
print('\n', cure_data.describe())
print('\n', cure_data.info())
print('\n', cure_data.describe(include=object))

print('\n', cure_data["key_mode"].value_counts())

# What is the most danceable album?

print('\n', cure_data["album_name"].unique())
albums_av_danceability = []
for album in cure_data["album_name"].unique():
    albums_av_danceability.append(cure_data.loc[cure_data["album_name"] == album, "danceability"].mean())
    print(album, cure_data.loc[cure_data["album_name"] == album, "danceability"].mean())

print("\nThe most danceable album is",
      cure_data["album_name"].unique()[albums_av_danceability.index(max(albums_av_danceability))],
      "with danceability = ", max(albums_av_danceability))

# print('\n', cure_data.loc[cure_data["album_name"] == "The Head On The Door", ["track_name","danceability"]])
print('\n', cure_data[(cure_data["album_name"] == "The Head On The Door")][["track_name","danceability"]])

print('\n Average duration of a track in minutes:',round(cure_data["duration_ms"].mean()/60000,2))

# TODO visualize

plt.figure(figsize=(12,6))
plt.title("Average Danceability, by Album")
sns.barplot(x=cure_data["album_name"].unique(), y=albums_av_danceability)
plt.xticks(rotation=90)
plt.ylabel("Danceability (1.0 is most danceable)")
plt.show()

# Popularity by album

plt.title("Distibution of tracks' popularity, by album")
# Show each observation with a scatterplot
sns.stripplot(x="track_popularity", y="album_name",
              data=cure_data, dodge=True, alpha=.5, zorder=1)
# Show the conditional means
sns.pointplot(x="track_popularity", y="album_name",
              data=cure_data, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)
plt.show()

# Correlation matrix

# Compute the correlation matrix
corr_matrix = cure_data.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(9, 6))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# cmap = sns.color_palette("mako", as_cmap=True)
plt.title("Correlation of The Cure tracks' parameters")
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

sns.lmplot(data=cure_data, x="valence", y="danceability")
plt.xticks(rotation=60)
plt.show()