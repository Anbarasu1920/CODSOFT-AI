import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


data = pd.read_csv('ratings.csv')


reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)


trainset, testset = train_test_split(dataset, test_size=0.25)


model = SVD()
model.fit(trainset)
predictions = model.test(testset)


accuracy.rmse(predictions)


def get_recommendations(user_id, model, dataset, num_recommendations=10):
    all_movie_ids = dataset.df['movieId'].unique()
    user_rated_movie_ids = dataset.df[dataset.df['userId'] == user_id]['movieId']
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in user_rated_movie_ids]

    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [pred.iid for pred in predictions[:num_recommendations]]


user_id = 1
recommendations = get_recommendations(user_id, model, dataset)
print(recommendations)

