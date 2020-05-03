# ��S�� chap4.R
# �} 4.1 �i�o�����z�֐��j
x <- rnorm(10)
#x <- rchisq(30,df=3)
#qqnorm(x) ; qqline(x)
plot.ecdf(x)

# �} 4.2 Q-Q �v���b�g�̐}��
xx <- seq(-3,5,0.1)
y0 <- pnorm(xx)
y1 <- pnorm(xx,m=2)
plot(c(-3.5,5), c(-0.05,1),xaxt="n",yaxt="n",xlab="",ylab="",type="n",axes=F)
lines(xx,y0) ; lines(xx,y1)
#
lines(c(-3.2,5),c(0,0)) ; lines(c(-3.2,5),c(1,1))
lines(c(-3.2,-3.2), c(0,1)) ; lines(c(5,5), c(0,1))
lines(c(-3.2,qnorm(0.4,m=2)), c(0.4,0.4))
lines(rep(qnorm(0.4,m=0),2),c(0,0.4))
lines(rep(qnorm(0.4,m=2),2),c(0,0.4))
text(-3.4,0,"0") ; text(-3.4,1,"1") ; text(-3.4,0.4,"p")
text(qnorm(0.4,m=0),-0.03,expression(q[x]))
text(qnorm(0.4,m=2),-0.03,expression(q[y]))
text(qnorm(0.4,m=0)-.2,0.45,expression(F[x]))
text(qnorm(0.4,m=2)-.2,0.45,expression(F[y]))

# �} 4.3 Q-Q �v���b�g
# �} 4.3 ���F���K���z
x <- rnorm(10)
qqnorm(x) ; qqline(x)
# �} 4.3 �E�F �J�C��敪�z
x <- rchisq(30,df=3)
qqnorm(x) ; qqline(x)

# �} 4.4 �̏d�̐��K Q-Q �v���b�g
male.weight <- c(53.1,56.0,58.0,59.0,59.5,60.0,61.9,63.9,69.8,76.3,96.5)
mean(male.weight)
sd(male.weight)
stem(male.weight)
qqnorm(male.weight,xlab="")
qqline(male.weight)

# �} 4.5 �Ɛ} 4.6 �̏d�f�[�^�̈ꕔ (n=420) �𗘗p
male420 <- read.table("maleweight420.txt",header=T)
# ��W�c�̈ꕔ (n=420)
hist(male420$weight,br=20,freq=F,main="",xlab="weight")
#mean(male420$weight); sd(male420$weight)
xx <- seq(min(male420$weight),max(male420$weight),0.1)
yy <- dnorm(xx,m=mean(male420$weight),s=sd(male420$weight))
lines(xx,yy)

# x-bar (n=11) �̕��z�i�V�~�����[�V�����j
ww <- numeric(420)
for (i in 1:420) ww[i]<-mean(sample(weight.data,11))
hist(ww,br=20,freq=F,xlab="mean of n=11",main="")
xx <- seq(min(ww),max(ww),0.1)
yy <- dnorm(xx,m=mean(ww),s=sd(ww))
lines(xx,yy)

pp <- c(.05,.1,.5,.9,.95)
quantile(ww,pp)
qnorm(pp,m=mean(ww),s=sd(ww))
mean(weight.data) - 1.96 * sd(weight.data) / sqrt(11)

# p.143�`145 �̎��s��
height <- read.table("height.txt",header=T)
height
names(height)
attach(height)
plot(father,son)
cor(father,son)
mean(height)
var(height)

# ��A���� �F �e�q�̐g���̃f�[�^
reg.h <- lm(son ~ father)
summary(reg.h)
abline(reg.h)

# �C�� �i�Q�̊O��l�����O���čĐ��肷����@�D���K���̖�4.5�Q�Ɓj
reg.h2 <- update(reg.h, subset=(father > 159))
summary(reg.h2)
detach(height)

# p.144 �̎��s�� �i�\�t�g�{�[�������C��A�̌��ہj
soft<-read.table("softball.txt", header=T)
attach(soft) 
mean(soft.x); mean(soft.y)
sd(soft.x); sd(soft.y)
cor(soft.x,soft.y)
# ��A����
soft.reg <- lm(soft.y ~ soft.x)
summary(soft.reg)
plot(soft.x,soft.y)
abline(soft.reg)
# �t�����̉�A (p.145)
soft.reg2 <- lm(soft.x ~ soft.y)
summary(soft.reg2)
plot(soft.y,soft.x)
abline(soft.reg2)
detach(soft) 

# p.145 �̎��s�� �i�t�����̉�A�j
reg.inv <- lm(height$father ~ height$son)
summary(reg.inv)

# �} 4.9 �̎��s��
# ���֌W���̕��z
rho <- 0.8 # �����Œ�`����鑊�֌W���̒l��K���ɕύX�ł���
aa <- (1-sqrt(1-rho^2))/rho
B <- 50000 # �J��Ԃ���
corr.b <- numeric(B)
for (i in 1:B) {
u <- rnorm(20) ; v <- rnorm(20)
xx <- aa*u + v ; yy <- u + aa*v
corr.b[i] <- cor(xx,yy)
}
hist(corr.b, xlim=c(0.2,1),br=50,xlab="r",main="Correlation",freq=F)
# �ϊ����ꂽ z �̕��z
z.b <- (1/2) * log((1+corr.b)/(1-corr.b))
hist(z.b,br=50,freq=F,xlab="",main="z transform")

# p. 147 �M����ԁF�e�q�̐g���̑��֌W��
cor(father,son)
(1/2) * log((1+0.3251)/(1-0.3251)) # == 0.3373
tanh(0.337) # ���Z
0.3373 - 1.96/sqrt(20-3)
zeta <- c(0.3373 - 1.96/sqrt(20-3),0.3373 + 1.96/sqrt(20-3))
tanh(zeta)
