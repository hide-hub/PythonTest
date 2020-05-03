#練習問題 2.1
planet<-read.table("planet.data",header=T)
head(planet)
#
公転周期 <-planet[,"公転周期"]
軌道長半径<-planet[,"軌道長半径"]
軌道長半径2乗<-軌道長半径^2
planet.lm1<-lm(公転周期~軌道長半径)
summary(planet.lm1)
planet.lm2<-lm(公転周期~軌道長半径2乗)
summary(planet.lm2)
op<-par()
#グラフィックを2行2列の4画面に分割
par(mfrow=c(2,2))
plot(軌道長半径,公転周期)
plot(公転周期,planet.lm1$residuals,xlab="公転周期",ylab="残差",main="説明変数は軌道長半径")
plot(公転周期,planet.lm2$residuals,xlab="公転周期",ylab="残差",main="説明変数は軌道長半径の2乗")
plot(planet.lm1$residuals,planet.lm2$residuals,xlab="説明変数は軌道長半径",ylab="説明変数は軌道長半径の2乗",main="残差の比較")
par(mfrow=op$mfrow)
#冥王星を除くと
id<-rownames(planet$data)=="冥王星"
planet2.data<-planet[!id,]
公転周期2 <-planet2.data[,"公転周期"]
軌道長半径2<-planet2.data[,"軌道長半径"]
軌道長半径2乗2<-軌道長半径2^2
planet2.lm1<-lm(公転周期2~軌道長半径2)
summary(planet2.lm1)
planet2.lm2<-lm(公転周期2~軌道長半径2乗2)
summary(planet2.lm2)
op<-par()
#グラフィックを2行2列の4画面に分割
par(mfrow=c(2,2))
plot(軌道長半径2,公転周期2)
plot(公転周期2,planet2.lm1$residuals,xlab="公転周期",ylab="残差",main="説明変数は軌道長半径")
plot(公転周期2,planet2.lm2$residuals,xlab="公転周期",ylab="残差",main="説明変数は軌道長半径の2乗")
plot(planet2.lm1$residuals,planet2.lm2$residuals,xlab="説明変数は軌道長半径",ylab="説明変数は軌道長半径の2乗",main="残差の比較")
par(mfrow=op$mfrow)

#練習問題2.2
#room1
#賃貸マンションデータ1
room1<-read.table("Mansion1.csv",header=T,sep=",")
head(room1)
家賃1<-room1[,"家賃"]
大きさ1<-room1[,"大きさ"]
room1.reg1<-lm(家賃1~大きさ1)
summary(room1.reg1)
#賃貸マンションデータ2
room2<-read.table("Mansion2.data",header=T,sep="\t")
家賃2<-room2[,"家賃"]
大きさ2<-room2[,"大きさ"]
room2.reg1<-lm(家賃2~大きさ2)
summary(room2.reg1)

#練習問題2.3
room2<-read.table("Mansion2.data",header=T,sep"\t")
head(room2)
家賃2<-room2[,"家賃"]
大きさ2<-room2[,"大きさ"]
近さ2<-room2[,"近さ"]
築年数2<-room2[,"築年数"]
room2.reg1<-lm(家賃2~大きさ2)
room2.reg2<-lm(家賃2~近さ2)
room2.reg3<-lm(家賃2~築年数2)
summary(room2.reg1)
summary(room2.reg2)
summary(room2.reg3)

#決定係数から
cat("説明変数が大きさの場合　R^2=",cor(家賃2,room2.reg1$fitted.values)^2,"\n")
cat("説明変数が近さの場合　R^2=",cor(家賃2,room2.reg2$fitted.values)^2,"\n")
cat("説明変数が築年数の場合　R^2=",cor(家賃2,room2.reg3$fitted.values)^2,"\n")

#練習問題2.4
test<-read.table("test.data",header=T,sep="\t",row.names=1)
head(test)

wa<-test[,1]+test[,2]
sa<-test[,2]-test[,1]
plot(wa,sa,xlab="試験結果の和",ylab="試験結果の差")
２回目<-test[,1]
１回目<-test[,2]
２回目<-(２回目-mean(２回目))/sd(２回目)
１回目<-(１回目-mean(１回目))/sd(１回目)
test.reg1<-lm(２回目~１回目)
summary(test.reg1)

#練習問題2.5
baseball<-read.table("baseball.data",header=T,sep="\t") #球団名も変数として扱う
head(baseball)
League<-baseball[,2]
Salary<-baseball[,3]
Winning<-baseball[,4]
par(mfcol=c(2,2))
boxplot(Salary~League)
boxplot(Winning~League)
plot(Salary,Winning)
plot(Winning,Salary)
par(mfcol=c(1,1))
summary(baseball)
sapply(baseball[,3:4],sd)

cor(baseball[,3:4])

#仮説　利益への選手の貢献＝＞年棒が高い＝＞結果は勝率
baseball.reg1<-lm(Winning~Salary)
summary(baseball.reg1)
plot(Winning,baseball.reg1$residuals)