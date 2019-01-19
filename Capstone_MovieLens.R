############################################
### MovieLens Capstone Prediction Script ###
############################################
# Jon Wayland
# 1/17/2019

#############################################################
# Create edx set, validation set, and submission file
#############################################################

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

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

#validation <- validation %>% select(-rating) # This part is commented out so that we can obtain the RMSE

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = NA),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#########################
### END DATA CREATION ###
#########################


###################
### Model Build ###
###################
# This script contains 2 methods, both of which will yield a RMSE below the cut-off for full credit.
# The method used was the second method, i.e. matrix factorization from the recosystem package.
# If this is tested, the 'Feature Engineering' section does NOT need to be ran.


##################################
### Loading Necessary Packages ###
##################################

library(tidyverse) # Loading the tidyverse suite of packages, including dplyr and ggplot2
library(caret) # Loading caret for data splitting and potential training
library(data.table) # Loading data.table to leverage certain structured data.table functions
library(stringr) # Loading stringr to access string manipulation capabilities

###########################
### Feature Engineering ###
###########################
# Note: None of these features are dependent on the rating itself or the distribution of predictors
## Thus can be applied on both the training and validation sets

## *** Flags for Each Genre ***
genres <- edx %>% select(movieId, genres) %>% unique()

# Splitting the genres out based on pipe deliminator
s <- strsplit(genres$genres, split = "\\|")

# Restructuring Data Frame to Include Each Genre
genres<-data.frame(genre = unlist(s), 
                   movieId = rep(genres$movieId, sapply(s, length)))

# Getting the unique list of genres
genres %>% select(genre) %>% unique

# Creating genre flags for each movie
genres$value<-1 # 1 for presence of genre
genres <- spread(genres, genre, value) # Going from long to wide format
genres[is.na(genres)]<-0 # 0 for absence of genre
genres$genreCount <- rowSums(genres[,2:ncol(genres)]) # Summing the presence of genres to get a count

# Renaming genre columns with dashes for ease of use later
genres <- genres %>% rename(No_Genre_Listed = `(no genres listed)`,
                            Film_Noir = `Film-Noir`,
                            Sci_Fi = `Sci-Fi`)

# Joining back to edx
edx <- edx %>% inner_join(genres, by = c("movieId" = "movieId"))
validation <- validation %>% inner_join(genres, by = c("movieId" = "movieId"))
# Save space
rm(genres)
rm(s)

## *** Year Movie was Released ***

# Helper function to use substr from the right-hand side
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}
edx$year <- as.integer(gsub(")","",substrRight(edx$title, 5)))
validation$year <- as.integer(gsub(")","",substrRight(validation$title, 5)))

####################################################
### Splitting the data into testing and training ###
####################################################
# Note: This section is commented out as it was used to develop the models, however, the predictions are built using
### all of the edx data. The report outlines the use of movieTrain and movieTest, but for purposes of the predictions,
### movieTrain has been replaced with edx and movieTest has been replaced with validation.

# Data splits
#set.seed(1)
#test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
#movieTrain <- edx[-test_index,]
#temp <- edx[test_index,]

# Make sure userId and movieId in movieTest set are also in movieTrain
#movieTest <- temp %>% 
#  semi_join(movieTrain, by = "movieId") %>%
#  semi_join(movieTrain, by = "userId")

#rm(temp)
#rm(edx)
#rm(validation)
#rm(test_index)
#rm(s)

############################
### The Predictive Model ###
############################
# Method 1: Movie, User, and Genre Effects
# Method 2: Matrix Factorization

### ***NOTE*** The method used for predictions was the SECOND method.

#########################################################################
### First Method - Introducing Bias Variables: Movie, User, and Genre ###
#########################################################################

# User-Genre combinations
genres <- edx %>% select(userId, movieId, rating, genres) %>% unique()

# Splitting the genres out based on pipe deliminator
s <- strsplit(genres$genres, split = "\\|")

# Restructuring Data Frame to Include Each Genre
genres<-data.frame(genre = unlist(s), 
                   userId = rep(genres$userId, sapply(s, length)),
                   movieId = rep(genres$movieId, sapply(s, length)),
                   rating = rep(genres$rating, sapply(s, length)))

# Save space
rm(s)

# Defining the mean and lambda
mu <- mean(edx$rating)
l<-4.25 # Found using optimal RMSE

# Creating the userPref dataframe
userPref <- genres %>% 
  group_by(userId, genre) %>%
  summarize(b_g = sum(rating-mu)/(n()+l))

# Save space
rm(genres)

# Renaming genres to match with previously built columns
userPref <- userPref %>% mutate(genre = case_when(genre == '(no genres listed)' ~ 'No_Genre_Listed',
                                                  genre == 'Film-Noir' ~ 'Film_Noir',
                                                  genre == 'Sci-Fi' ~ 'Sci_Fi',
                                                  TRUE ~ as.character(genre)))

# Spreading the user preferences across by genre
userPref <- spread(userPref, genre, value = b_g)

# If the user never rated any then there is no effect
userPref[is.na(userPref)]<-0

# Adding a suffix to the column names
colnames(userPref) <- paste0(colnames(userPref), "_Effect")

# Ignoring the (no genres listed) instance
userPref$No_Genre_Listed_Effect<-NULL

# Creating the overall genre effect b_g based on present combinations and removing the individual effects
## Note: This is done by multiplying the effect by the flag (i.e. establishing genre presence), 
#### summing all of the effects and then dividing by the count of genres to get the average genre effect by user, genre combination

# Joining to the training set
edx <- edx %>%
  inner_join(userPref, by = c('userId' = 'userId_Effect')) %>%
  mutate(
    Action_Effect = Action_Effect*Action,
    Adventure_Effect = Adventure_Effect*Adventure,
    Animation_Effect = Animation_Effect*Animation,
    Children_Effect = Children_Effect*Children,
    Comedy_Effect = Comedy_Effect*Comedy,
    Crime_Effect = Crime_Effect*Crime,
    Documentary_Effect = Documentary_Effect*Documentary,
    Drama_Effect = Drama_Effect*Drama,
    Fantasy_Effect = Fantasy_Effect*Fantasy,
    Film_Noir_Effect = Film_Noir_Effect*Film_Noir,
    Horror_Effect = Horror_Effect*Horror,
    IMAX_Effect = IMAX_Effect*IMAX,
    Musical_Effect = Musical_Effect*Musical,
    Mystery_Effect = Mystery_Effect*Mystery,
    Romance_Effect = Romance_Effect*Romance,
    Sci_Fi_Effect = Sci_Fi_Effect*Sci_Fi,
    Thriller_Effect = Thriller_Effect*Thriller,
    War_Effect = War_Effect*War,
    Western_Effect = Western_Effect*Western) %>%
  mutate(b_g = Action_Effect + Adventure_Effect + Animation_Effect + Children_Effect +
           Comedy_Effect + Crime_Effect + Documentary_Effect + Drama_Effect + Fantasy_Effect +
           Film_Noir_Effect + Horror_Effect + IMAX_Effect + Musical_Effect + Mystery_Effect +
           Romance_Effect + Sci_Fi_Effect + Thriller_Effect + War_Effect + Western_Effect) %>%
  select(-c(Action_Effect,Adventure_Effect,Animation_Effect,Children_Effect,
            Comedy_Effect,Crime_Effect,Documentary_Effect,Drama_Effect,Fantasy_Effect,
            Film_Noir_Effect,Horror_Effect,IMAX_Effect,Musical_Effect,Mystery_Effect,
            Romance_Effect,Sci_Fi_Effect,Thriller_Effect,War_Effect,Western_Effect)) %>%
  mutate(b_g = b_g/genreCount)

# Joining to the testing set
validation <- validation %>%
  inner_join(userPref, by = c('userId' = 'userId_Effect')) %>%
  mutate(
    Action_Effect = Action_Effect*Action,
    Adventure_Effect = Adventure_Effect*Adventure,
    Animation_Effect = Animation_Effect*Animation,
    Children_Effect = Children_Effect*Children,
    Comedy_Effect = Comedy_Effect*Comedy,
    Crime_Effect = Crime_Effect*Crime,
    Documentary_Effect = Documentary_Effect*Documentary,
    Drama_Effect = Drama_Effect*Drama,
    Fantasy_Effect = Fantasy_Effect*Fantasy,
    Film_Noir_Effect = Film_Noir_Effect*Film_Noir,
    Horror_Effect = Horror_Effect*Horror,
    IMAX_Effect = IMAX_Effect*IMAX,
    Musical_Effect = Musical_Effect*Musical,
    Mystery_Effect = Mystery_Effect*Mystery,
    Romance_Effect = Romance_Effect*Romance,
    Sci_Fi_Effect = Sci_Fi_Effect*Sci_Fi,
    Thriller_Effect = Thriller_Effect*Thriller,
    War_Effect = War_Effect*War,
    Western_Effect = Western_Effect*Western) %>%
  mutate(b_g = Action_Effect + Adventure_Effect + Animation_Effect + Children_Effect +
           Comedy_Effect + Crime_Effect + Documentary_Effect + Drama_Effect + Fantasy_Effect +
           Film_Noir_Effect + Horror_Effect + IMAX_Effect + Musical_Effect + Mystery_Effect +
           Romance_Effect + Sci_Fi_Effect + Thriller_Effect + War_Effect + Western_Effect) %>%
  select(-c(Action_Effect,Adventure_Effect,Animation_Effect,Children_Effect,
            Comedy_Effect,Crime_Effect,Documentary_Effect,Drama_Effect,Fantasy_Effect,
            Film_Noir_Effect,Horror_Effect,IMAX_Effect,Musical_Effect,Mystery_Effect,
            Romance_Effect,Sci_Fi_Effect,Thriller_Effect,War_Effect,Western_Effect)) %>%
  mutate(b_g = b_g/genreCount)

# movie effect
b_i<-edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating-mu)/(n()+l))

# user effect (as b_g is dependent on user, it is included in b_u calculation)
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating-b_g-b_i-mu)/(n()+l))

# Joining b_u and b_i back to training and testing and using the new prediction
edx <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu+b_i+b_u+b_g)

validation <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu+b_i+b_u+b_g)

#############################################################
### Second Method - Matrix Factorization Using Recosystem ###
#############################################################

# Install/Load recosystem
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")


# Using recosystem 
train_data <- data_memory(user_index = edx$userId, item_index = edx$movieId, 
                          rating = edx$rating, index1 = T)

test_data <- data_memory(user_index = validation$userId, item_index = validation$movieId, index1 = T)


# Choices for parameters were optimized using RMSE as a loss function. The code is in the report but not here as it takes over 24 hours to optimize.

recommender <- Reco()
recommender$train(train_data, opts = c(dim = 30, 
                                       costp_l1 = 0.0, costp_l2 = 0.01, 
                                       costq_l1 = 0.0, ostq_l2 = 0.1,
                                       lrate = 0.1, niter = 500, nthread = 6, verbose = F))  

prediction <- recommender$predict(test_data, out_memory())

# Adding bounds to the rating scale
prediction[prediction < 0.5] <-0.5
prediction[prediction > 5.0] <-5.0

# Applying the prediction
validation$pred <- prediction

# The final model's RMSE:
RMSE(validation$rating,validation$pred)

# 0.785109



#############################################################
### Writing Validation Predictions to the Submission File ###
#############################################################
# As this portion is no longer needed from the script, it is commented out

#submission <- read.csv("C:\\Users\\jwayland\\Desktop\\personal\\submission.csv")
#validation <- validation %>% select(userId, movieId, prediction) %>% rename(rating = prediction)
#submission <- submission %>% select(userId, movieId)
#submission <- submission %>%
#  inner_join(validation, by = c('userId', 'movieId'))

#write.csv(submission, file = "C:\\Users\\jwayland\\Desktop\\personal\\R\\submission files\\submission.csv")
