import heapq
import numpy as np

from MovieLens import MovieLens
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
from scipy.spatial import distance
        
testSubject = '100'
k = 10

# Load our data set and compute the user similarity matrix
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()
genres = ml.getGenres()
usergenre = ml.getUserGenres(100)

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )
kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
print("\nOriginal Recommendations:")
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        if (ratingSum > 5):
            ratingSum = 5
        print(ml.getMovieName(int(movieID)), int(ratingSum))
        pos += 1
        if (pos > 9):
            break

print("\nNew Recommendations:")
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        dst = distance.euclidean(usergenre, genres[int(movieID)])
        if (0 < dst < 2.2):
            ratingSum += ratingSum * 1/int(dst)
            if (ratingSum > 5):
                ratingSum = 5
            print(ml.getMovieName(int(movieID)), int(ratingSum))
        else:
            pos -= 1

        pos += 1
        if (pos > 9):
            break



