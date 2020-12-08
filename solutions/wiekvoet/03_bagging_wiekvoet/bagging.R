# Bagging Example

# https://wiekvoet.blogspot.com/2015/08/predicting-titanic-deaths-on-kaggle-iii.html

#install.packages("ipred")
library(ipred)
library(rpart)
library(lattice)

# read and combine
train <- read.csv('../input/train.csv', stringsAsFactors = TRUE)
train$status <- 'train'
test  <- read.csv('../input/test.csv', stringsAsFactors = TRUE)
test$status <- 'test'
test$Survived <- NA
tt <- rbind(test,train)

# generate variables
tt$Pclass <- factor(tt$Pclass)
tt$Survived <- factor(tt$Survived)
tt$age <- tt$Age
tt$age[is.na(tt$age)] <- 35
tt$age <- cut(tt$age,c(0,2,5,9,12,15,21,55,65,100))
tt$Title <- sapply(tt$Name,function(x) strsplit(as.character(x),'[.,]')[[1]][2])
tt$Title <- gsub(' ','',tt$Title)
tt$Title[tt$Title %in% c('Capt','Col','Don','Sir','Jonkheer','Major')] <- 'Mr'
tt$Title[tt$Title %in% c('Lady','Ms','theCountess','Mlle','Mme','Ms','Dona')] <- 'Miss'
tt$Title <- factor(tt$Title)
tt$A <- factor(grepl('A',tt$Cabin))
tt$B <- factor(grepl('B',tt$Cabin))
tt$C <- factor(grepl('C',tt$Cabin))
tt$D <- factor(grepl('D',tt$Cabin))
tt$E <- factor(grepl('E',tt$Cabin))
tt$F <- factor(grepl('F',tt$Cabin))
tt$ncabin <- nchar(as.character(tt$Cabin))
tt$PC <- factor(grepl('PC',tt$Ticket))
tt$STON <- factor(grepl('STON',tt$Ticket))
tt$cn <- as.numeric(gsub('[[:space:][:alpha:]]','',tt$Cabin))
tt$oe <- factor(ifelse(!is.na(tt$cn),tt$cn%%2,-1))
tt$Fare[is.na(tt$Fare)]<- median(tt$Fare,na.rm=TRUE)

## 

forage <- tt[!is.na(tt$Age) & tt$status=='train',names(tt) %in% 
               c('Age','Sex','Pclass','SibSP',
                 'Parch','Fare','Title','Embarked','A','B','C','D','E','F',
                 'ncabin','PC','STON','oe')]

ipbag1 <- bagging(Age ~.,data=forage)
ipbag1

plot(tt$Age~predict(ipbag1,tt))

tt$AGE <- tt$Age
tt$AGE[is.na(tt$AGE)] <- predict(ipbag1,tt[is.na(tt$AGE),])

titanic <- tt[tt$status=="train",]

di1 <- subset(titanic,select=c(
  age,SibSp,Parch,Fare,Sex,Pclass,
  Title,Embarked,A,B,C,D,E,F,ncabin,PC,STON,oe,AGE,Survived))
dso <- expand.grid(ns=seq(100,300,25),nbagg=c(500),minsplit=1:6)
la <- lapply(1:nrow(dso),function(ii) {
  ee <-    errorest(Survived ~ .,
                    ns=dso$ns[ii],
                    control=rpart.control(minsplit=dso$minsplit[ii], cp=0, 
                                          xval=0,maxsurrogate=0),
                    nbagg=dso$nbagg[ii],
                    model=bagging,
                    data=di1,
                    est.para=control.errorest(k=20)
  )
  cc <- c(ns=dso$ns[ii],minsplit=dso$minsplit[ii],nbagg=dso$nbagg[ii],error=ee$error)
  print(cc)
  cc
})
las <- do.call(rbind,la) 
las <- as.data.frame(las)
xyplot(error ~ ns, groups= minsplit, data=las,auto.key=TRUE,type='l')

bagmod <- bagging(Survived ~.,ns=275,nbagg=500,
                  control=rpart.control(minsplit=5, cp=0, xval=0,maxsurrogate=0),
                  data=di1)

pp <- predict(bagmod,test)

out <- data.frame(
  PassengerId=test$PassengerId,
  Survived=pp,row.names=NULL)
write.csv(x=out,
          file='bag8aug.csv',
          row.names=FALSE,
          quote=FALSE)

