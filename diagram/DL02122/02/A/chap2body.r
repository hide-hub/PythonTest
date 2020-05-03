dir()
Salary<-read.table("Salary.data",header=T,sep="\t")
head(Salary)
mean2000<-apply(Salary[1:12,2:3],2,mean)
cat("平均",mean2000,"\n")
SalaryIndex.ts<-ts(Salary[,2:3]/mean2000,start=c(2000,1),frequency=12)
#図2.1
plot(SalaryIndex.ts,xlab="図2.1")

#図2.2
SalaryofAll.ts<-ts(Salary[,2]/10000,start=c(2000,1),frequency=12)
SalaryofTokyo.ts<-ts(Salary[,3]/10000,start=c(2000,1),frequency=12)
#図2.2
plot(decompose(SalaryofAll.ts),xlab="図2.2  年月")

#図2.3
plot(acf(SalaryofTokyo.ts),xlab="図2.3  月差")

#2章回帰　家賃
room2<-read.table("Mansion2.data",header=T,sep="\t")
head(room2)
Yachin<-room2[,"家賃"]*10  #10円を1円へ
Okisa<-room2[,"大きさ"]
#readline("\nType  <Return>\t to 図2.4 : ")
#図2.4
hist(Yachin,xlim=c(60000,200000),xlab="図2.4 家賃")
#readline("\n Next 図2.5 ")
#図2.5
plot(qqnorm(Yachin),xlab="図2.5 Q-Qプロット")
#readline("\n Next 図2.6 ")
#図2.6
plot(Okisa,Yachin,xlab="図2.6 大きさ",ylab="家賃")
#直線の当てはめ
Yachin.reg1<-lm(Yachin~Okisa)
#回帰の結果
summary(Yachin.reg1)
#図2.8

plot(Yachin.reg1$fitted.value,Yachin.reg1$residuals,xlab="図2.8 予測値",ylab="残差",main="家賃　予測値 vs 残差")
