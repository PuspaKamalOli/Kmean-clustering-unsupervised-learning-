import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('college_data', index_col=0)
df.head()

sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', fit_reg=False)

kmean = KMeans(n_clusters=2)
kmean.fit(df.drop('Private', axis=1))
print(kmean.cluster_centers_)


def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0


df['cluster'] = df['Private'].apply(converter)
df.head()
print(confusion_matrix(df['cluster'], kmean.labels_))
print('/n')
print(classification_report(df['cluster'], kmean.labels_))
