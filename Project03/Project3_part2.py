import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import collections
import surprise
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise import KNNBasic

from collections import Counter
from sklearn.metrics import roc_curve , auc
from collections import defaultdict

data_ratings = pd.read_csv('ratings.csv', delimiter=',', names=['user_id', 'item_id', 'rating', 'timestamp'], header=0)
user_ids = list(data_ratings.user_id)
movie_ids = list(data_ratings.item_id)
rating_ids = list(data_ratings.rating)
user_vs_movie_ratings = data_ratings.pivot_table(index=['user_id'], columns=['item_id'], values='rating',
                                                 fill_value=0).values
read = surprise.reader.Reader(line_format=u'user item rating timestamp', sep=',', skip_lines=1, rating_scale=(0.5, 5))
data_load = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
count_movie = Counter(movie_ids)
count_usr_rating = Counter(user_ids)


# Question 1: Compute sparsity of movie rating dataset

def sparsity():
    total_ratings_count = user_vs_movie_ratings.shape[0] * user_vs_movie_ratings.shape[
        1]  # total number of users * total number of movies    #R.shape[0] * R.shape[1]
    available_ratings_count = data_ratings.shape[0]  # total number of ratings available    #df.shape[0]
    sparsity = float(available_ratings_count) / float(total_ratings_count)
    #print('Sparsity of the dataset: ', sparsity)


sparsity()


# Question 2:  Histogram of rating counts

def ratingsHistogram():
    ratings = data_ratings.iloc[0:data_ratings.shape[0], 2]
    plt.hist(ratings, ec='black', bins=9)  # , bins=10)#, bins=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plt.title("Rating value frequencies")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 5.5, 0.5))
    #plt.show()


ratingsHistogram()


# Question 3:

def ratingDistribution():
    movie_ids_2 = list()
    ratings_count = list()
    for i in sorted(count_movie, key=count_movie.get, reverse=True):
        movie_ids_2.append(i)
        ratings_count.append(count_movie[i])

    plt.plot(ratings_count)
    plt.title("Ratings Distribution among movies")
    plt.xlabel("Movie ID")
    plt.ylabel("Ratings Count")
    plt.xticks(range(10000), " ")
    #plt.show()
    return movie_ids_2, ratings_count


movie_ids_2, ratings_count = ratingDistribution()


# Question 4:

def userRatingCounter():
    user_ids_2 = list()
    ratings_count_2 = list()
    for i in sorted(count_usr_rating, key=count_usr_rating.get, reverse=True):
        user_ids_2.append(i)
        ratings_count_2.append(count_usr_rating[i])

    plt.plot(ratings_count_2)
    plt.title(" User Rating Distribution ")
    plt.xlabel("User IDs")
    plt.ylabel("Ratings Counter")
    plt.xticks(range(900), " ")
    #plt.show()


userRatingCounter()


# Question 6

def varianceHistogram():
    data = zip(movie_ids, rating_ids)
    data2 = collections.defaultdict(list)
    for movie, rating in data:
        data2[movie].append(rating)

    movie_list = []
    variance_list = []
    for i in data2.keys():
        movie_list.append(i)
        variance_list.append(np.var(data2[i]))

    plt.hist(variance_list, bins=10, ec='black')
    plt.title("Variance of rating values received by each movie")
    plt.xlabel("Variance")
    plt.ylabel("Movie count")
    plt.xticks(np.arange(0, 5.5, 0.5))
    #plt.show()
    return variance_list


vars = varianceHistogram()

# Question 10

def KPlots(matrix):
    test_rmse = []
    test_mae = []
    k = range(2, 102, 2)
    for i in k:
        print(i)
        knn = surprise.prediction_algorithms.knns.KNNWithMeans(k=i, sim_options={'name': 'pearson', 'user_based': True})
        knn_output = surprise.model_selection.cross_validate(knn, matrix, measures=['RMSE', 'MAE'], cv=10, verbose=False)
        rmse = knn_output['test_rmse']
        mae = knn_output['test_mae']
        test_rmse.append(sum(rmse) / 10.0)
        test_mae.append(sum(mae) / 10.0)
    print("\n\nK values: ", k)
    print("\n\nAverage RMSE Values: ", test_rmse)
    print("\n\nAverage MAE Values", test_mae)
    plt.plot(k, test_rmse)
    plt.xlabel('K')
    plt.ylabel('Average RMSE')
    plt.show()
    plt.plot(k, test_mae)
    plt.xlabel('K')
    plt.ylabel('Average MAE')
    plt.show()
    return test_rmse, test_mae

def NNMFPlots(matrix):
    test_rmse = []
    test_mae = []
    k = range(2, 52, 2)
    for i in k:
        print(i)
        nnmf = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=i)
        nnmf_output = surprise.model_selection.cross_validate(nnmf, matrix, measures=['RMSE', 'MAE'], cv=10, verbose=False)
        rmse = nnmf_output['test_rmse']
        mae = nnmf_output['test_mae']
        test_rmse.append(sum(rmse) / 10.0)
        test_mae.append(sum(mae) / 10.0)
    print("\n\nK values: ", k)
    print("\n\nAverage RMSE Values: ", test_rmse)
    print("\n\nAverage MAE Values", test_mae)
    plt.plot(k, test_rmse)
    plt.xlabel('K')
    plt.ylabel('Average RMSE')
    plt.show()
    plt.plot(k, test_mae)
    plt.xlabel('K')
    plt.ylabel('Average MAE')
    plt.show()
    return test_rmse, test_mae


#test_rmse, test_mae = KPlots(data_load)


# Question 11:

#def stableRMSEandMAE():
 #   for i in range(1, len(test_rmse)):
  #      if (test_rmse[i - 1] - test_rmse[i] <= 0.001) and (test_rmse[i - 1] - test_rmse[
   #         i] > 0):  # and (test_mae[i-1]-test_mae[i] >0) and (test_mae[i-1]-test_mae[i] <= 0.001):
    #        print(2 * i + 1, test_rmse[i - 1] - test_rmse[i], test_rmse[i - 1] - test_rmse[i])


#stableRMSEandMAE()

# Question 12, 13 and 14

def trimmer(typ, matrix, vars , movie_ids_2, ratings_count ):

    if typ == 'popular':
        b = np.less_equal(ratings_count, 2) + 0

    elif typ == 'unpopular':
        b = np.greater(ratings_count, 2) + 0

    elif typ == 'variance':
        c = np.less(ratings_count, 5) + 0
        d = np.less(vars, 2) + 0
        b = np.logical_or(c,d) + 0

    aux = np.multiply(b, movie_ids_2)
    delete_ids = filter(lambda a: a != 0, aux)
    delete_ids = map(str, delete_ids)

    trimatrix = [tup for tup in matrix if not tup[1] in delete_ids]
    return trimatrix

data_knn_popular = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
data_knn_unpopular = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
data_knn_variance = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)

#data_knn_popular.raw_ratings = trimmer('popular',data_load.raw_ratings, vars , movie_ids_2, ratings_count )
#data_knn_unpopular.raw_ratings = trimmer('unpopular',data_load.raw_ratings, vars , movie_ids_2, ratings_count )
#data_knn_variance.raw_ratings = trimmer('variance',data_load.raw_ratings, vars , movie_ids_2, ratings_count )

print('KNN for popular')
#KPlots(data_knn_popular)
print('KNN for unpopular')
#KPlots(data_knn_unpopular)
print('KNN for high variance')
#KPlots(data_knn_variance)

# Question 15

#knn_output = sp.model_selection.cross_validate(knn, matrix, measures=['RMSE', 'MAE'], cv=10, verbose=False)

data_load = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
trainset, testset = train_test_split(data_load, test_size=.10)
model = KNNBasic(k=30, sim_options={'name': 'pearson', 'user_based': True})
knn = surprise.prediction_algorithms.knns.KNNWithMeans(k=30, sim_options={'name': 'pearson', 'user_based': True})
model.fit(trainset)
predictions = model.test(testset)

def plotROC(testset,predictions):
    thresholds = [2.5, 3, 3.5, 4]
    for threshold in thresholds:
        true = np.greater_equal([tup[2] for tup in testset],threshold)+0
        pred = [tup[3] for tup in predictions]

        fpr, tpr, thresholds = roc_curve(true,pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for t = ' + str(threshold))
        plt.legend(loc="lower right")
        plt.show()

        print ('Area under the curve for Threshold = %d is %f' %(threshold,roc_auc))

plotROC(testset,predictions)

# Quastion 17 and 18

#data_load = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
#test_rmse, test_mae = NNMFPlots(data_load)

# Question 19, 20 and 21

data_nnmf_popular = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
data_nnmf_unpopular = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
data_nnmf_variance = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)

#data_nnmf_popular.raw_ratings = trimmer('popular',data_load.raw_ratings, vars , movie_ids_2, ratings_count )
#data_nnmf_unpopular.raw_ratings = trimmer('unpopular',data_load.raw_ratings, vars , movie_ids_2, ratings_count )
#data_nnmf_variance.raw_ratings = trimmer('variance',data_load.raw_ratings, vars , movie_ids_2, ratings_count )

print('NNMF for popular')
#NNMFPlots(data_knn_popular)
print('NNMF for unpopular')
#NNMFPlots(data_knn_unpopular)
print('NNMF for high variance')
#NNMFPlots(data_knn_variance)

# Question 22

#data_load = surprise.Dataset.load_from_file(file_path='ratings.csv', reader=read)
#trainset, testset = train_test_split(data_load, test_size=.10)
#model = KNNBasic(k=30, sim_options={'name': 'pearson', 'user_based': True})
nnmf = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=20)
nnmf.fit(trainset)
predictions = model.test(testset)

plotROC(testset,predictions)
