# https://wiekvoet.blogspot.com/2015/07/predicting-titanic-deaths-on-kaggle-ii.html

library(dplyr)
library(gbm)
# install.packages("gbm")

set.seed(4321)

## Training Data ####

titanic <- read.csv('../input/train.csv', stringsAsFactors = TRUE) %>%
  mutate(.,Pclass=factor(Pclass),
         Survived=factor(Survived),
         age=ifelse(is.na(Age),35,Age),
         age = cut(age,c(0,2,5,9,12,15,21,55,65,100)),
         Title=sapply(Name,function(x) strsplit(as.character(x),'[.,]')[[1]][2]),
         Title=gsub(' ','',Title),
         Title =ifelse(Title %in% c('Capt','Col','Don','Sir','Jonkheer','Major'),'Mr',Title),
         Title =ifelse(Title %in% c('Lady','Ms','theCountess','Mlle','Mme','Ms','Dona'),'Miss',Title),
         Title = factor(Title),
         A=factor(grepl('A',Cabin)),
         B=factor(grepl('B',Cabin)),
         C=factor(grepl('C',Cabin)),
         D=factor(grepl('D',Cabin)),
         E=factor(grepl('E',Cabin)),
         F=factor(grepl('F',Cabin)),
         ncabin=nchar(as.character(Cabin)),
         PC=factor(grepl('PC',Ticket)),
         STON=factor(grepl('STON',Ticket)),
         cn = as.numeric(gsub('[[:space:][:alpha:]]','',Cabin)),
         oe=factor(ifelse(!is.na(cn),cn%%2,-1)),
         train = sample(c(TRUE,FALSE),
                        size=891,
                        replace=TRUE, 
                        prob=c(.9,.1)   ) )

## Test Data ####

test <- read.csv('../input/test.csv', stringsAsFactors = TRUE) %>%
  mutate(.,
         Embarked=factor(Embarked,levels=levels(titanic$Embarked)),
         Pclass=factor(Pclass),
         #      Survived=factor(Survived),
         age=ifelse(is.na(Age),35,Age),
         age = cut(age,c(0,2,5,9,12,15,21,55,65,100)),
         Title=sapply(Name,function(x) strsplit(as.character(x),'[.,]')[[1]][2]),
         Title=gsub(' ','',Title),
         Title =ifelse(Title %in% c('Capt','Col','Don','Sir','Jonkheer','Major'),'Mr',Title),
         Title =ifelse(Title %in% c('Lady','Ms','theCountess','Mlle','Mme','Ms','Dona'),'Miss',Title),
         Title = factor(Title),
         A=factor(grepl('A',Cabin)),
         B=factor(grepl('B',Cabin)),
         C=factor(grepl('C',Cabin)),
         D=factor(grepl('D',Cabin)),
         E=factor(grepl('E',Cabin)),
         F=factor(grepl('F',Cabin)),
         ncabin=nchar(as.character(Cabin)),
         PC=factor(grepl('PC',Ticket)),
         STON=factor(grepl('STON',Ticket)),
         cn = as.numeric(gsub('[[:space:][:alpha:]]','',Cabin)),
         oe=factor(ifelse(!is.na(cn),cn%%2,-1))
  )
test$Fare[is.na(test$Fare)]<- median(titanic$Fare)


forage <- filter(titanic,!is.na(titanic$Age)) %>%
  select(.,Age,SibSp,Parch,Fare,Sex,Pclass,Title,Embarked,A,B,C,D,E,F,ncabin,PC,STON,oe)

## gbm ####

rfa1 <- gbm(Age ~ ., 
            data=forage,
            interaction.depth=4,
            cv.folds=10,
            n.trees=8000,
            shrinkage=0.0005,
            n.cores=2)

gbm.perf(rfa1)


titanic$AGE<- titanic$Age
titanic$AGE[is.na(titanic$AGE)] <- predict(rfa1,titanic,n.trees=7118)[is.na(titanic$Age)]
test$AGE<- test$Age
test$AGE[is.na(test$AGE)] <- predict(rfa1,test,n.trees=7118)[is.na(test$Age)]


gb1 <- filter(titanic,train) %>%
  select(.,age,SibSp,Parch,Fare,Sex,Pclass,
         Title,Embarked,A,B,C,D,E,F,ncabin,PC,STON,oe,AGE,Survived)%>%
  mutate(Survived=c(0,1)[Survived]) # not integer or factor but float

#table(gb1$Survived)
gb1m <-      gbm(Survived ~ .,
                 cv.folds=11,
                 n.cores=2,
                 interaction.depth=5,
                 shrinkage = 0.0005,
                 distribution='adaboost',
                 data=gb1,
                 n.trees=10000)
gbm.perf(gb1m)


preds <- predict(gb1m,titanic,
                 n.trees=6000, type='response')
density(preds) %>% plot


## Prediction 2 ####

preds2<- preds[!titanic$train]
target <- c(0,1)[titanic$Survived[!titanic$train]]
sapply(seq(.3,.7,.01),function(step)
  c(step,sum(ifelse(preds2<step,0,1)!=target)))

pp <- predict(gb1m,test,n.trees=6000,type='response')
pp <- ifelse(pp<0.48,0,1)
out <- data.frame(
  PassengerId=test$PassengerId,
  Survived=pp,row.names=NULL)

## Generate Submission ####

write.csv(x=out,
          file='gbm.csv',
          row.names=FALSE,
          quote=FALSE)
