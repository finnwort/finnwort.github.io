#run sims for Ariely experiment

library(ggplot2)
library(tidyverse)

# Function to take two values and a tolerance level and return true if 
# mainval within tolerance level of checkagainstval 
#(ie 5 checked against 7 with tolerance level of 2 would return true)

idcheck <- function(mainval, checkagainstval, tolerance){
  valmax <- checkagainstval+tolerance
  valmin <- checkagainstval-tolerance
  result <- ifelse(mainval < valmax & mainval > valmin,
                   TRUE, FALSE)
  return(result)
}

#convert previous function to check one value against every value in a set

idcheckset <- function(mainval, set, tolerance){
  idvec <- c()
  for (i in 1:length(set)) {
    idvec[i] <- idcheck(mainval, set[i], tolerance)
  }
  return(idvec)
}

# "uniform" kernel function specified in Bhui paper

kernel <- function(X){
  X[X >= -1 & X <= 1] <- (X[X >= -1 & X <= 1]+1)/2
  X[X <= -1] <- 0
  X[X >= 1] <- 1
  return(X)
}

#Main function: gives position of x within inferred distribution based on set X given bandwidth h
#so a CDF is built using X and h then function returns position of x in this CDF
#note that subsetsize is the size of a set sampled from the set then used when inferring distribution
# this will be useful in future but normally we make it redundant by setting it as the n of X

biaskde <- function(x, X, h, subsetsize){
  X1 <- sample(X, subsetsize)
  xdiff <- x - X1
  kx <- xdiff/h
  kX <- kernel(kx)
  CDF <- sum(kX*1/length(kX))
  return(CDF)
}

#Simple extension to previous function but recomputes 
# CDF and returns values x for every x in set x given set X 
#(supposed to represent a set of items (x) being judged serially etc against presented set of X)
#note. this is not important in current code

biaskdevec <- function(x, X, h, subsetsize){
  Xpred <- x
  for (i in 1:length(X)){
    Xpred[i] <- biaskde(x[i], X, h, subsetsize)
  }
  return(Xpred)
}

#function to take set, sample initialsetn from set, compute CDF from sample with bandwidth h
# then sample test values until the value is within decisionthreshold distance from 0.5
# function then returns value that was "accepted" as mean

meandiscrim <- function(set, initialsetn, decisionthreshold, h){
  setnew <- sample(set, initialsetn)
  rankval <- 0
  while (idcheck(rankval, 0.5, decisionthreshold) != 1){
    testval <- runif(1, min(setnew), max(setnew))
    rankval <- biaskde(testval, setnew, h, length(setnew))
  }
  return(testval)
}

#pretty useful, simple function to add noise to each value in set, 
# noise is generated from a normal distribution with mean 0 and SD of noisesd

perceptualnoise <- function(set, noisesd){
  newset <- c()
  for (i in 1:length(set)) {
    newset[i] <- set[i]+(rnorm(1, 0, noisesd))
  }
  return(newset)
}

#Ignore next few things I have only left them in because I think they might be useful
lset <- c()
sset <- c()
for (i in 1:10){
  lset[i] <- meandiscrim(rep(c(0.1, 0.5, 0.6, 0.7, 0.8, 0.9),2), 12, 0.04, 5)
  sset[i] <- meandiscrim(c(0.1, 0.5, 0.6, 0.7, 0.8, 0.9), 6, 0.04, 5)
}
lset
sset
#potential proof that we need subsample < n

#function that takes a set, adds noisy retrieval, estimates mean of set using parameters from before
#then returns 1 if testvalue is above mean and 0 if testvalue is below mean
# this should represent people being given a set then a comparison and asked if the comparison is above the mean of the set

meandiscrimcompare <- function(set, testvalue, noisesd, initialsetn, decisionthreshold, h){
  newtestvalue <- perceptualnoise(testvalue, noisesd)
  newset <- perceptualnoise(set, noisesd)
  mean1 <- meandiscrim(newset, initialsetn, decisionthreshold, h)
  diff <- mean1 - newtestvalue
  choice <- ifelse(diff >= 0, 0, 1)
  return(choice)
}

#Next lines just follow the rule Ariely used to produce dot sizes - 
#parts of this were changed between conditions
set1 <- c(-1.5, -0.5, 0.5, 1.5)
set1 <- 0.25*(1.40^set1)
data <- c()
theme_set(theme_bw())

#Function takes in a set and other parameters and is then asked a number of times (330 in this case)
#whether a randomly produced value between testbounds[1] and testbounds[2] is lower or higher than estimated set mean
#output is graph of data and probit regression fit
#second value outputted is discrimination threshold (difference in order to be 1 SD away from 50/50 choice)
#PS let me know if this conceptualisation of discrimination threshold is wrong 

discrimthresholdsim <- function(set, testbounds, noisesd, initialsetn, decisionthreshold, h){
  set1 <- set
  difference <- c()
  choice <- c()
  testval <- c()
  for (i in 1:330) {
    testval[i] <- runif(1, testbounds[1], testbounds[2])
    difference[i] <- testval[i] - mean(set1)
    choice[i] <- meandiscrimcompare(set1, testval[i], noisesd = noisesd, initialsetn = initialsetn, 
                                    decisionthreshold = decisionthreshold, h = h)
  }
  data <- data.frame(c(1:330), difference, choice)
  colnames(data) <- c("Round", "Difference", "Choice")
  probreg <- glm(Choice ~ Difference, family = binomial(link = "probit"),
                 data = data)
  absdifference <- (1-probreg$coefficients[1])/probreg$coefficients[2]
  discrimthreshold <- (absdifference/mean(set1))*100
  data$predict <- pnorm(probreg$coefficients[1]+
                          (data$Difference*probreg$coefficients[2]))
  plot <- ggplot(data, aes(x = Difference, y = Choice)) +
    geom_point() +
    geom_line(aes(y = predict)) +
    xlab("Set 1 - Set 2") +
    ylab("Chose Set 1")
  print(plot)
  return(c(absdifference, discrimthreshold))
}

#set, testbounds, noisesd, initialsetn, decisionthreshold, h
#Example run of previous function with noisesd, h, and decision threshold that seems to work 

discrimthresholdsim(set1, c(0.1, 0.58), 0.03, 4, 0.04, 0.1)

#Next set of code is similar to before but it isn't set up as a single function
#this code takes into account set, testvals, internal noise sd and threshold
#then produces recognition data for each of the test vals to be part of set1
#important thing is the graph it produces

idtestvals <- c(-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5)
set1 <- c(-1.5, -0.5, 0.5, 1.5)
testvals <- 0.25*(1.4^idtestvals)
set1 <- 0.25*(1.4^set1)
noisesd <- 0.07
threshold <- 0.06

value <- c()
yeschoice <- c()
for (i in 1:500) {
  value[i] <- sample(testvals, 1)
  id <- idcheckset(value[i], perceptualnoise(set1, noisesd), threshold)
  yeschoice[i] <- ifelse(length(id[id == T]) == 0, 0, 1)
}
df <- data.frame(value, yeschoice)
head(df)
plot(df$value, df$yeschoice)
plot <- ggplot(df, aes(x = as.factor(value), y = yeschoice))+
  geom_point()+
  geom_line(stat = "summary")
sumdf <- df %>% group_by(value) %>% 
  summarise(percentyes = sum(yeschoice == 1)/sum(yeschoice == 1|yeschoice == 0))
plot <- ggplot(sumdf, aes(x = value, y = percentyes)) +
  geom_line()
plot


# Notes: no perceptual noise but identification threshold noise -
# need very low threshold noise for right disc. threshold
# very low threshold noise leads to perfect performance in id experiment
# high perceptual noise and low threshold noise leads to very large discrimination threshold




