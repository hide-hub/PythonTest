#�U��
room2<-read.table("Mansion2.data",header=T,sep="\t")
head(room2)

Yachin.reg2<-lm(room2[,"�ƒ�"]~room2[,"�傫��"]+room2[,"�k��"])
summary(Yachin.reg2)
plot(room2[,"�ƒ�"],Yachin.reg2$residuals,xlab="�ƒ�",ylab="�c��")

#6�͗��K���6.1
room2<-read.table("Mansion2.data",header=T,sep="\t")
head(room2)
#���֌W��
Rxx<-cor(room2[,c("�k��","�傫��","�z�N��")])
print(Rxx)
abs(det(Rxx))
Yachin.reg3<-lm(room2[,"�ƒ�"]~room2[,"�k��"]+room2[,"�傫��"]+room2[,"�z�N��"])
summary(Yachin.reg3)
#�}6.1
plot(room2[,"�ƒ�"],Yachin.reg3$residuals,xlab="�ƒ�",ylab="�c��")

