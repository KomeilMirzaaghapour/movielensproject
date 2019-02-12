# This R script is for the Harvard CapStone Movielens project.
# If your computer has sufficient memory (>64GB), then you should be able to run the script and get the predicted RMSE. 
# This script starts with the 9 helper functions described in the pdf report (helper functon 8 is inside helper function 9).
# the code for generating edx and validation set is copied from the course material website.
# the data visualization code can all be found in the movielensreport.Rmd file. 
# If you can't run funkSVD from recommenderlab package, you should still be able to get the preSVD RMSE using function postprocess1.

# load libraries (except for recommenderlab).
library(tidyverse)
library(caret)
library(lubridate)

# generate edx and validation data sets from the course material. 

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# helper functions: 

preprocess <- function(tab){
  # helper function 1, remove genres, titles columns, and convert timestamp column into date.
  # round the date into monthly for b_ut(date_m) and quarterly for b_it (date_q).
  # remove timestamp and date column. 
  #
  # Args:
  #   tab: either edx data frame or validation data frame.
  #
  # Returns:
  #   Transformed edx data frame or validation data frame. 
  tab <- tab%>%select(-c('title','genres'))%>%
    mutate(date=as_datetime(timestamp))%>%
    mutate(date_q=round_date(date, unit='quarter'), date_m=round_date(date,unit='month'))%>%
    select(-c('date','timestamp'))
  return (tab)
}

partitionf <- function(tab){
  # helper function 2, randomly divide edx into train (~ 90% edx) and test (~ 10% edx).
  # make sure train contains all userId and movieId in test set.
  #
  # Args:
  #   tab: transformed edx data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #
  # Returns:
  #   a list with 2 data frames, test data frame and train data frame.
  test_index <- createDataPartition(y = tab$rating, times = 1, p = 0.1, list = FALSE)
  train <- tab[-test_index,]
  temp <- tab[test_index,]
  # Make sure userId and movieId in test set are also in train set
  test <- temp %>% 
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")
  # Add rows removed from test set back into train set
  removed <- anti_join(temp, test)
  train <- rbind(train, removed)
  return (list(train=train,test=test))
}

fit_bu_bi<- function(L1, tab){
  # helper function 3, function to help find best l1 for b_u and b_i.
  # cross validates 5 times (sampling using helper function 2 ,partitionf() for 5 times).
  # average RMSEs across 5 randomly generated test sets for each l1 value will be calculated. 
  #
  # Args:
  #   L1: a numeric vector of regularization terms for b_i and b_u.
  #   tab: transformed edx data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #
  # Returns:
  #   a numeric vector of average RMSEs for each l1 value.
  rmses<- replicate(5,{
    t <- partitionf(tab = tab)  # generate a random test and train set 
    mu <- mean(t$train$rating)  # average rating for each new train set from partitionf()  
    sapply(L1,function(l1){
      # calculate b_i and b_u on the train set 
      b_i <- t$train%>%
        group_by(movieId)%>%
        summarize(b_i=sum(rating-mu)/(n()+l1))
      b_u <- t$train%>%
        left_join(b_i, by='movieId')%>%
        group_by(userId)%>%
        summarize(b_u=sum(rating-mu-b_i)/(n()+l1))
      # add b_i and b_u to the test set to calculate predicted rating 
      p <- t$test%>%
        left_join(b_i,by='movieId')%>%
        left_join(b_u,by='userId')%>%
        mutate(p = mu+b_i+b_u)%>%.$p
      # caret package already has an RMSE function ( sqrt(mean(truevalue-predvalue)^2)))
      return(RMSE(t$test$rating,p)) 
    }
    )
  })
  # RMSEs are returned as a matrix (each row is for l1 value and each column is for each test sample set)
  # average RMSEs will be the average row value across all test samples 
  avg_rmses <- rowMeans(rmses) 
  return(avg_rmses)
}

fit_it <- function(L2, tab, l1){
  # helper function 4, function to help find best l2 for b_it.
  # cross validates 5 times (sampling using helper function 2 ,partitionf() for 5 times).
  # average RMSEs across 5 randomly generated test sets for each l2 value will be calculated. 
  #
  # Args:
  #   L2: a numeric vector of regularization terms for b_it.
  #   tab: transformed edx data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #   l1: a numeric value of regularization term for b_i and b_u.
  #
  # Returns:
  #   a numeric vector of average RMSEs for each l2 value.
  rmses <- replicate(5,{
    t <- partitionf(tab = tab)  #generate a random test and train set 
    mu <- mean(t$train$rating)  #average for each new train set using partitionf()  
    # use l1 to fit b_i and b_u on each new train set 
    b_i <- t$train%>%
      group_by(movieId)%>%
      summarize(b_i=sum(rating-mu)/(n()+l1)) 
    b_u <- t$train%>%left_join(b_i, by='movieId')%>%
      group_by(userId)%>%summarize(b_u=sum(rating-mu-b_i)/(n()+l1)) 
    # add b_i and b_u to each train set before sapply to speed up the process
    train<- t$train%>%
      left_join(b_i, by='movieId')%>%left_join(b_u, by='userId')
    # add b_i and b_u to each test before sapply to speed up the process
    test <- t$test%>%
      left_join(b_i,by='movieId')%>%left_join(b_u,by='userId')
    sapply(L2,function(l2){
      # calculate b_it for the train set 
      b_it <- train%>%
        group_by(movieId, date_q)%>%
        summarize(b_it=sum(rating-mu-b_i-b_u)/(n()+l2))
      # add b_it to test set to calculate predicted rating 
      # for some movies that didn't get rated during that quarter will get a zero as no information during that period 
      p <- test%>%left_join(b_it,by=c('movieId','date_q'))%>%
        mutate(b_it=ifelse(is.na(b_it),0,b_it))%>%
        mutate(p = mu+b_i+b_u+b_it)%>%.$p
      # using RMSE in caret package 
      return(RMSE(test$rating,p))
    }
    )
  }
  )
  # RMSEs are returned as a matrix (each row is for l2 value and each column is for each test sample set)
  # average RMSEs will be the average row value across all test samples 
  avg_rmses <- rowMeans(rmses) 
  return(avg_rmses)
}

fit_ut <- function(L3, tab, l1, l2){
  # helper function 5, function to help find best l3 for b_ut.
  # cross validates 5 times (sampling using helper function 2 ,partitionf() for 5 times).
  # average RMSEs across 5 randomly generated test sets for each l3 value will be calculated. 
  #
  # Args:
  #   L3: a numeric vector of regularization terms for b_ut.
  #   tab: transformed edx data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #   l1: a numeric value of regularization term for b_i and b_u.
  #   l2: a numeric value of regularization term for b_it.
  #
  # Returns:
  #   a numeric vector of average RMSEs for each l3 value.
  rmses <- replicate(5,{ 
    t <- partitionf(tab = tab)  #generate a random test and train set 
    mu <- mean(t$train$rating)  #average for each new train set using partitionf()  
    # use l1 to fit b_i and b_u on each new train set 
    b_i <- t$train%>%
      group_by(movieId)%>%summarize(b_i=sum(rating-mu)/(n()+l1))
    b_u <- t$train%>%left_join(b_i, by='movieId')%>%
      group_by(userId)%>%summarize(b_u=sum(rating-mu-b_i)/(n()+l1))
    # add b_i and b_u to each train set before sapply to speed up the proces
    train <- t$train%>%left_join(b_i, by='movieId')%>%left_join(b_u, by='userId')
    # use l2 to calculate b_it for each new train set
    b_it <- train%>%group_by(movieId,date_q)%>%
      summarize(b_it=sum(rating-mu-b_i-b_u)/(n()+l2))
    # add b_it to each train set before sapply to speed up the process 
    train<- train%>%
      left_join(b_it, by=c('movieId','date_q'))
    # add b_i, b_u, and b_it to each test set before sapply to speed up the process
    # for some movies that didn't get rated during that quarter will get a zero as no information during that period 
    test <- t$test%>%
      left_join(b_i,by='movieId')%>%left_join(b_u,by='userId')%>%
      left_join(b_it, by=c('movieId','date_q'))%>%
      mutate(b_it=ifelse(is.na(b_it),0,b_it))
    sapply(L3,function(l3){
      # calculate b_ut on the train set
      b_ut <- train%>%
        group_by(userId, date_m)%>%
        summarize(b_ut=sum(rating-mu-b_i-b_u-b_it)/(n()+l3))
      # add b_ut to the test set to calculate predicted rating 
      # for some users that didn't rate during that month will get a zero as no information during that period 
      p <- test%>%
        left_join(b_ut,by=c('userId','date_m'))%>%
        mutate(b_ut=ifelse(is.na(b_ut),0,b_ut))%>%
        mutate(p = mu+b_i+b_u+b_it+b_ut)%>%.$p
      # using RMSE in caret package 
      return(RMSE(test$rating,p)) 
    }
    )
  }
  )
  # RMSEs are returned as a matrix (each row is for l3 value and each column is for each test sample set)
  # average RMSEs will be the average row value across all test samples 
  avg_rmses <- rowMeans(rmses) 
  return(avg_rmses) 
}

postprocess1<-function(tab1,tab2,l1,l2,l3){
  # helper function 6, use the best overall l1,l2,and l3 to calculate b_i, b_u, b_it, and b_ut. 
  # calculates preSVD predicted rating and RMSE for validation set. 
  #
  # Args:
  #   tab1: transformed edx data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #   tab2: transformed validation data frame from function preprocess() with 5 columns(userId, movieId, rating, date_q, date_m).
  #   l1: a numeric value of regularization term for b_i and b_u.
  #   l2: a numeric value of regularization term for b_it.
  #   l3: a numeric value of regularization term for b_ut.
  #
  # Returns:
  #   a list of 3 items, train data frame, test data frame, and rmse
  #   train, transformed tab1 with 3 columns(userId, movieId, and calculated residual (true rating - predicted rating)).
  #   test, transformed tab2 with 4 columns(userId, movieId, actual rating, and preSVD predicted rating).
  #   rmse, a numeric value of RMSE for the validation set without SVD
  mu <- mean(tab1$rating)  # calculate the mean using tab1 
  # calculate b_i and b_u using tab1 and l1
  b_i <- tab1%>%
    group_by(movieId)%>%summarize(b_i=sum(rating-mu)/(n()+l1))
  b_u <- tab1%>%
    left_join(b_i, by='movieId')%>%
    group_by(userId)%>%summarize(b_u=sum(rating-mu-b_i)/(n()+l1))
  # add b_i and b_u to tab1 and save as train
  train <- tab1%>%
    left_join(b_i, by='movieId')%>%
    left_join(b_u, by='userId')
  # calculate b_it using train(modified tab1) and l2
  b_it <- train%>%
    group_by(movieId,date_q)%>%
    summarize(b_it=sum(rating-mu-b_i-b_u)/(n()+l2))
  # add b_it to train
  train <- train%>%
    left_join(b_it, by=c('movieId','date_q'))
  # calculate b_ut using train and l3
  b_ut <- train%>%
    group_by(userId, date_m)%>%
    summarize(b_ut=sum(rating-mu-b_i-b_u-b_it)/(n()+l3))
  # add b_ut to train 
  # calculate residual (true rating - predicted rating)
  # select userId, movieId, and residual to make matrix later for SVD using recommenderlab 
  train <- train%>%
    left_join(b_ut, by=c('userId','date_m'))%>%
    mutate(p = mu+b_i+b_u+b_it+b_ut)%>%
    mutate(resid=rating-p)%>%
    select(userId,movieId,resid)
  # add b_i,b_u,b_it,b_ut to tab2 and save as test 
  # for b_it and b_ut, 0 is given when no information during that period 
  # calulate predicted rating using just base predictor biases
  # select userId, movieId, predicted rating and actual rating for final prediction
  test<- tab2%>%
    left_join(b_i, by='movieId')%>%
    left_join(b_u, by='userId')%>%
    left_join(b_it, by=c('movieId','date_q'))%>%
    mutate(b_it=ifelse(is.na(b_it),0,b_it))%>%
    left_join(b_ut,by=c('userId','date_m'))%>%
    mutate(b_ut=ifelse(is.na(b_ut),0,b_ut))%>%
    mutate(p = mu+b_i+b_u+b_it+b_ut)%>%select(userId,movieId,p,rating) 
  # calculate preSVD RMSE for validation set using caret package 
  return (list(train=train,test=test,rmse=RMSE(test$rating,test$p))) 
}

SVDf <- function(tab){
  # helper function 7, perform funkSVD to calculate U and V feature factors.
  # calculates 10 features for each user and each movie.
  #
  # Args:
  #   tab: transformed edx data frame, train, using function postprocess1() with 3 columns (userId, movieId, and calculated residual).
  #
  # Returns:
  #   a list of 2 items, movie feature matrix, movief, and user feature matrix, userf.
  library(recommenderlab)  #load the recommenderlab package
  # convert tab into a matrix (row as userId, column as movieId, each matrix value as residual)
  resid_matrix <- tab%>%as('realRatingMatrix')%>%as('matrix')
  # set max iteration to 300 and set verbose to true to see real time progress 
  # caution: don't run this, it will take one day and it may crash your computer! 
  fsvd <- funkSVD(resid_matrix,max_epochs = 300,verbose = TRUE)
  # get movie matrix from fsvd (each row is for movieId and 10 columns for 10 feature factors)
  movief <- fsvd$V
  # set the correct moiveId for movie feature matrix
  rownames(movief)<-colnames(resid_matrix)
  # get user matrix from fsvd (each row is for userId and 10 columns for 10 feature factors)
  userf <- fsvd$U
  # set the correct userId for user feature matrix 
  rownames(userf)<-rownames(resid_matrix) 
  return(list(movief=movief,userf=userf))
}

finalprocess <- function(tab, movief, userf){
  # helper function 9, calculates the final predicted rating and final RMSE for the validation set
  #
  # Args:
  #   tab: transformed validation data frame, test, using function postprocess1() with 4 columns(userId, movieId, actual rating, and preSVD predicted rating).
  #   movief: movie feature matrix using function SVDf().
  #   userf: user feature matrix using function SVDf().
  # Returns:
  #   a list of 2 items, predicted rating, pred, and final RMSE, rmse.
  #   pred is a large numeric vector of final predicted rating for the validation set.
  #   rmse is the final RMSE for the validation set.
  # Create helperfunction 8 to calculate specific user and movie residual
  resid_f<- function(u,i){
    # helperfunction 8, calculates predicted residual for user u and movie i using vector dot product.
    #
    # Args:
    #   u: a numeric value of userId.
    #   i: a numeric value of movieId.
    #
    # returns:
    #   a numeric value of residual (user factor vector dot product movie factor vector). 
    u <-userf[toString(u),]  # create user factor vector by matching userId with correct rowname
    i <- movief[toString(i),]  # create movie factor vector by matching movieId with correct rowname
    # calculates residual and returns the numeric value
    return(sum(u*i))
  }
  # Use mapply to use function resid_f() to calculate predicted residual and final predicted rating.
  test <-tab%>%
    mutate(resid = mapply(resid_f,userId,movieId))%>%
    mutate(p=p+resid)
  # calculates final RMSE using caret package
  return(list(pred=test$p,rmse=RMSE(test$rating,test$p)))
}

# Use function preprocess() to create modified edx and validation set, rename as train and test 
train <- preprocess(edx)
test <- preprocess(validation)
# rm edx and validation set to save space 
rm(edx, validation) 
gc()

# Find best l1 for b_u and b_i using helper function fit_bu_bi() and helper function partitionf().
L1 <- seq(4,6,0.25)  # L1 values are carefully selected after some trial and error.
fit1<- fit_bu_bi(L1 = L1, tab = train)  # fit1 returns a vector of average RMSEs for each l1 value. 
l1 <- L1[which.min(fit1)]  # pick the l1 with the lowest average RMSE.

# Calculate b_i and b_u using l1 and find best l2 for b_it using function fit_it() and helper function partitionf().
L2 <- seq(48,55,0.5)  # L2 values are carefully selected after some trial and error.
fit2 <- fit_it(L2 = L2, tab = train,l1 = l1)  # fit2 returns a vector of average RMSEs for each l2 value. 
l2 <- L2[which.min(fit2)]  # pick the l2 with the lowest average RMSE.

# Calculate b_i and b_u using l1.
# Calculate b_it using l2.
# Find best l3 for b_ut using function fit_ut() and helper function partitionf().
L3 <- seq(11,14,0.25)  # L3 values are carefully selected after some trial and error.
fit3 <- fit_ut(L3 = L3, tab = train,l1 = l1,l2 = l2)  # fit3 returns a vector of average RMSEs for each l3 value.
l3 <- L3[which.min(fit3)]  # pick the l3 with the lowest average RMSE.

# Use helper function postprocess1() to calculate baseline biases using edx (train) and predict for validation (test).
preSVD <- postprocess1(tab1 = train,tab2 = test,l1 = l1,l2 = l2,l3 = l3)  # contains modified test,train data frames and RMSE as a list
# remove test, and train set to save space.
rm(test,train) 
gc()

# use helper function SVDf() to get movie and user factor matrices. 
# recommenderlab is loaded inside helper function SVDf()
movie_user_feature<- SVDf(tab = preSVD$train)  # contains movie factor matrix and user factor matrix as a list

# Use helper function finalprocess() to calculate predicted residual and rating for the validation set and the final RMSE. 
finalresult <- finalprocess(tab = preSVD$test, movief = movie_user_feature$movief,userf = movie_user_feature$userf)  # a list containing predicted rating and final RMSE
# Remove movie_user_feature and preSVD to save space
rm(preSVD,movie_user_feature)
gc()

# Final predicted rating for validation set
pred <- finalresult$pred

# Final rmse for validation set
RMSE <- finalresult$rmse
# remove finalresult to save space
rm(finalresult)
gc()

# show final RMSE
RMSE
