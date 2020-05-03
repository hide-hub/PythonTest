# ��T�� chap5.R

# �} 5.5 �i�a�ƃ��b�g�j �֘A�̕���
wgain <- read.table("rats.txt",header=T)
attach(wgain)
names(wgain)
boxplot(wgain,ylab="weight gain")
#
mean(wgain)
var(wgain)
sd(wgain)
t.test(beef.L,beef.H)
t.test(beef.L,beef.H,var.equal=T)
t.test(beef.L,cereal,var.equal=T,alt="less")
# pooled variance
s.pool <- sqrt((var(beef.L)*9 + var(beef.H)*9)/18)
(mean(beef.L)-mean(beef.H))/s.pool/sqrt(1/10+1/10)
qt(c(0.025,0.05),18)

par(mfrow=c(2,2))
qqnorm(cereal)
qqnorm(beef.L)
qqnorm(beef.H)
par(mfrow=c(1,1))
detach(wgain)

# �} 5.6 �i�_�C�G�b�g�f�[�^�j �֘A�̕���
df.diet <- read.table("dietweight.txt",header=T)
diet.d <- (df.diet$diet.w2 - df.diet$diet.w)
diet1 <- diet.d[1:10] ; diet2 <- diet.d[11:20]
mean(diet1)
sd(diet1)
(10^.5)*mean(diet1)/sd(diet1)  # t-value = -2.542817
pt((10^.5)*mean(diet1)/sd(diet1),df=9) # P-value = 0.0158
pt(-2.5428,df=9)
qt(0.05,df=9)
-1.833113 * sd(diet1)/sqrt(length(diet1))
# 2�W�{�ƌ��Ȃ����������Ȃ����� �i2-sample test)
wt1 <- df.diet$diet.w[1:10]
wt2 <- df.diet$diet.w2[1:10]
mean(wt2) - mean(wt1)
boxplot(data.frame(wt2,wt1))
plot(wt2,wt1)
cor(wt2,wt1)
sd(wt2) ; sd(wt1)
t.test(wt1,wt2,var.equal=T,alternative="less")

# �} 5.6 ���f�[�^�ƊO��l
diet1.err <- diet1 ; diet1.err[3]<- 6.4
# �c�������낦��
qqnorm(diet1, ylim=c(-7,7)) ; qqline(diet1)
qqnorm(diet1.err, ylim=c(-7,7)) ; qqline(diet1.err)
t.test(diet1.err)
#
mean(diet2)
sd(diet2)
t.test(diet1)
t.test(diet2)

# p. 178 �_�C�G�b�g�f�[�^�̉�A���� : diet1 = a + b wt1
diet.y <- diet1 ; diet.x <- wt1 
reg <- lm(diet.y ~ diet.x)
summary(reg)
# �e�i�K�̌v�Z�i�ڍׁj
diet.y
diet.x
sxy <- sum( (diet.x - mean(diet.x))*(diet.y - mean(diet.y)) )
sxx <- sum( (diet.x - mean(diet.x))^2 )
h.beta <- sxy/sxx
hat.y <- mean(diet.y) + h.beta * (diet.x-mean(diet.x))
fitted(reg) # hat.y �Ɉ�v����
residuals(reg) # �c�� (diet.y - hat.y) �Ɉ�v����
s2 <- sum((diet.y - hat.y)^2 / (length(diet.y)-2) )
(t.beta <- h.beta/sqrt(s2/sxx))

# p. 180 �����֐��̌���
diet.r <- cor(diet.x,diet.y) #  0.3544978
sxy <- sum( (diet.x - mean(diet.x))*(diet.y - mean(diet.y)) )
sxx <- sum( (diet.x - mean(diet.x))^2 )
syy <- sum( (diet.y - mean(diet.y))^2 )
sxy/sqrt(sxx*syy) # ���֌W�� diet.r �ɓ�����

cov(diet.x,diet.y)*9 # sxx, sxy, syy �Ɣ�r���� 
var(diet.x)*9
var(diet.y)*9

sqrt(8) * diet.r /sqrt(1-diet.r^2) # t �l
cor.test(diet.x,diet.y)


# p. 172 ��̓ǂݕ� �i�䗦�Ɋւ���Б�����j 
binom.test(2400-1250,2400, 0.5, alt="greater")
p <- 1250/2400 ; P <- 0.5
1-pnorm((p-0.5)/sqrt(P*(1-P)/2400))
1-pbinom(1249,2400,0.5) # ���m�Ȋm��
1-pnorm((1249.5/2400-0.5)/sqrt(P*(1-P)/2400)) # �A���C��

