from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

bootdataclean = pd.read_csv(
    'D:/ВУЗ/Магистратура/1 семестр/ММТС/Books_Data_Clean.csv')
bootdataclean.head(10)

bootdataclean.shape

plt.plot(bootdataclean['index'], bootdataclean['Book_average_rating'])
bootdataclean.columns
bootdataclean.rename(
    columns={"Publishing Year": "Publishing_Year"}, inplace=True)
bootdataclean.rename(columns={"Book Name": "Book_Name"}, inplace=True)
bootdataclean.rename(columns={"gross sales": "gross_sales"}, inplace=True)
bootdataclean.rename(
    columns={"publisher revenue": "publisher_revenue"}, inplace=True)
bootdataclean.rename(columns={"sale price": "sale_price"}, inplace=True)
bootdataclean.rename(columns={"sales rank": "sales_rank"}, inplace=True)
bootdataclean.rename(columns={"units sold": "units_sold"}, inplace=True)
bootdataclean.sample(5)

bootdataclean.duplicated().any()

bootdataclean[bootdataclean.duplicated()]

bootdataclean.info()

bootdataclean.isna().sum()

(bootdataclean.isna().sum() * 100) / bootdataclean.shape[0]

bootdataclean.Book_Name = bootdataclean.Book_Name.fillna("other")
bootdataclean.language_code = bootdataclean.language_code.fillna("other")
bootdataclean.isna().sum()

bootdataclean = bootdataclean.dropna(subset=['Publishing_Year'], how='any')
bootdataclean.isna().sum()


bootdataclean.reset_index(drop=True, inplace=True)
bootdataclean.tail(5)

bootdataclean.info()

bootdataclean.Publishing_Year = bootdataclean.Publishing_Year.astype('int16')
bootdataclean.Author_Rating = bootdataclean.Author_Rating .astype('category')
bootdataclean.genre = bootdataclean.genre.astype('category')
bootdataclean.language_code = bootdataclean.language_code.astype('category')

bootdataclean.info()
bootdataclean.head(5)

bootdataclean['gross_sales'].hist(bins=100)

bootdataclean.boxplot(column=['gross_sales'])

bootdataclean['gross_sales'].describe()

bootdataclean.sort_values(by='gross_sales', ascending=False).head(20)

len(bootdataclean[bootdataclean['gross_sales'] == 0.0])

bootdataclean['Author_Rating'].value_counts().plot.bar()

bootdataclean['genre'].value_counts().plot.bar()

bootdataclean['language_code'].value_counts().plot.bar()

bootdataclean_10 = bootdataclean.sort_values(
    by='publisher_revenue', ascending=False).head(10)
sns.set_style('darkgrid')
plt.figure()
sns.barplot(x='publisher_revenue', y='Book_Name', data=bootdataclean_10)
plt.title('Самый большой доход издателя', fontsize=24)
plt.xlabel('Доход', fontsize=16)
plt.ylabel('Название книги', fontsize=16)
plt.tick_params(axis='x', labelsize=12, rotation=90)
plt.tick_params(axis='y', labelsize=12)

plt.figure(figsize=(15, 10))
sns.countplot(x='genre', data=bootdataclean,
              order=bootdataclean['genre'].value_counts().index)
plt.xticks(rotation=90)

plt.figure(figsize=(15, 10))
sns.countplot(x='Publishing_Year', data=bootdataclean, order=bootdataclean.groupby(
    by=['Publishing_Year'])['Book_Name'].count().sort_values(ascending=False).index)
plt.xticks(rotation=90)

df = pd.read_csv('D:/ВУЗ/Магистратура/1 семестр/ММТС/Books_Data_Clean.csv')
selected_columns = ['Book_average_rating', 'Book_ratings_count',
                    'gross sales', 'publisher revenue', 'units sold']
data = df[selected_columns]
correlation_matrix = data.corr()
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))
plt.title("Тепловая карта корреляции данных")
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", fmt=".2f", square=True)
plt.show()

df = pd.read_csv('D:/ВУЗ/Магистратура/1 семестр/ММТС/Books_Data_Clean.csv')
selected_columns = ['Author_Rating', 'Book_average_rating',
                    'gross sales', 'publisher revenue', 'units sold']
data = df[selected_columns]
sns.pairplot(data)
plt.show()


bootdataclean['Success'] = np.nan
conditions = [(bootdataclean['publisher_revenue'] >= 20000), (bootdataclean['publisher_revenue'] < 20000) & (
    bootdataclean['publisher_revenue'] >= 5000), (bootdataclean['publisher_revenue'] < 5000)]
values = ['High', 'Medium', 'Low']
bootdataclean['Success'] = np.select(conditions, values)
bootdataclean.Success = bootdataclean.Success.astype('category')

X = bootdataclean.iloc[:, 6:8]
y = bootdataclean['Success']

X.sample(3)
y.sample(3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=33)
print(X_train)
print(X_test)
SVC_model = SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(classification_report(SVC_prediction, y_test, zero_division=1))
print(classification_report(KNN_prediction, y_test, zero_division=1))

X = bootdataclean.iloc[:, [6, 7, 11]]
X.sample(3)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count, X.sale_price)
ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
           :, 1], kmeans.cluster_centers_[:, 2], color='black')
ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
           :, 1], kmeans.cluster_centers_[:, 2], color='black')
ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
print(kmeans.cluster_centers_)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
           :, 1], kmeans.cluster_centers_[:, 2], color='black')
ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')
dbscan = DBSCAN(eps=0.4, min_samples=10)
dbscan.fit(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=dbscan.labels_, cmap='rainbow')
clusters = np.unique(dbscan.labels_)
for cluster in clusters:
    cluster_points = X[dbscan.labels_ == cluster]
    cluster_center = np.mean(cluster_points, axis=0)
    ax.scatter(cluster_center[0], cluster_center[1],
               cluster_center[2], color='black')

ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')

dbscan = DBSCAN(eps=0.2, min_samples=15)
dbscan.fit(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=dbscan.labels_, cmap='rainbow')
clusters = np.unique(dbscan.labels_)
for cluster in clusters:
    cluster_points = X[dbscan.labels_ == cluster]
    cluster_center = np.mean(cluster_points, axis=0)
    ax.scatter(cluster_center[0], cluster_center[1],
               cluster_center[2], color='black')

ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')

ac = AgglomerativeClustering()
ac.fit(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=ac.labels_, cmap='rainbow')
clusters = np.unique(ac.labels_)
for cluster in clusters:
    cluster_points = X[ac.labels_ == cluster]
    cluster_center = np.mean(cluster_points, axis=0)
    ax.scatter(cluster_center[0], cluster_center[1],
               cluster_center[2], color='black')

ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')

ac = AgglomerativeClustering(n_clusters=3)
ac.fit(X)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X.Book_average_rating, X.Book_ratings_count,
           X.sale_price, c=ac.labels_, cmap='rainbow')
clusters = np.unique(ac.labels_)
for cluster in clusters:
    cluster_points = X[ac.labels_ == cluster]
    cluster_center = np.mean(cluster_points, axis=0)
    ax.scatter(cluster_center[0], cluster_center[1],
               cluster_center[2], color='black')

ax.set_xlabel('Book_average_rating')
ax.set_ylabel('Book_ratings_count')
ax.set_zlabel('sale_price')

bootdataclean_pair = bootdataclean.loc[:, [
    "Publishing_Year", "Book_average_rating", "Book_ratings_count", "gross_sales", "sale_price", "units_sold"]]
sns.pairplot(bootdataclean_pair)

plt.scatter(bootdataclean.Book_average_rating, bootdataclean.units_sold)
plt.xlabel('Book_average_rating')
plt.ylabel('units_sold')

y = bootdataclean['units_sold']
x = bootdataclean['Book_average_rating']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

fig = plt.figure()
fig = sm.graphics.plot_regress_exog(model, 'Book_average_rating', fig=fig)

y = bootdataclean['units_sold']
x = bootdataclean[['Book_average_rating', 'Book_ratings_count', 'gross_sales']]
regr = linear_model.LinearRegression()
regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('\n')
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

fig = plt.figure()
fig = sm.graphics.plot_regress_exog(model, 'Book_average_rating', fig=fig)

bootdataclean['Success_2'] = np.nan
conditions = [(bootdataclean['units_sold'] >= 3000),
              (bootdataclean['units_sold'] < 3000)]
values = [1, 0]
bootdataclean['Success_2'] = np.select(conditions, values)
bootdataclean.Success_2 = bootdataclean.Success_2.astype('int16')
bootdataclean.head(10)

X = bootdataclean.iloc[:, 9:13]
Y = bootdataclean['Success_2']
X.sample(3)
Y.sample(3)

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape,
      Y_train.shape, Y_val.shape, Y_test.shape)


model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32,
                 epochs=100, validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)[1]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

predictions = model.predict(X_test[:3])
X_test[:3]
predictions
