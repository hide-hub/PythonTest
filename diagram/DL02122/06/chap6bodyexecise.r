#‚UÍ
room2<-read.table("Mansion2.data",header=T,sep="\t")
head(room2)

Yachin.reg2<-lm(room2[,"‰Æ’À"]~room2[,"‘å‚«‚³"]+room2[,"“k•à"])
summary(Yachin.reg2)
plot(room2[,"‰Æ’À"],Yachin.reg2$residuals,xlab="‰Æ’À",ylab="c·")

#6Í—ûK–â‘è6.1
room2<-read.table("Mansion2.data",header=T,sep="\t")
head(room2)
#‘ŠŠÖŒW”
Rxx<-cor(room2[,c("“k•à","‘å‚«‚³","’z”N”")])
print(Rxx)
abs(det(Rxx))
Yachin.reg3<-lm(room2[,"‰Æ’À"]~room2[,"“k•à"]+room2[,"‘å‚«‚³"]+room2[,"’z”N”"])
summary(Yachin.reg3)
#}6.1
plot(room2[,"‰Æ’À"],Yachin.reg3$residuals,xlab="‰Æ’À",ylab="c·")

