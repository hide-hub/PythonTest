# 第５章 chap5.R

# 図 5.5 （餌とラット） 関連の分析
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

# 図 5.6 （ダイエットデータ） 関連の分析
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
# 2標本と見なした正しくない分析 （2-sample test)
wt1 <- df.diet$diet.w[1:10]
wt2 <- df.diet$diet.w2[1:10]
mean(wt2) - mean(wt1)
boxplot(data.frame(wt2,wt1))
plot(wt2,wt1)
cor(wt2,wt1)
sd(wt2) ; sd(wt1)
t.test(wt1,wt2,var.equal=T,alternative="less")

# 図 5.6 元データと外れ値
diet1.err <- diet1 ; diet1.err[3]<- 6.4
# 縦軸をそろえる
qqnorm(diet1, ylim=c(-7,7)) ; qqline(diet1)
qqnorm(diet1.err, ylim=c(-7,7)) ; qqline(diet1.err)
t.test(diet1.err)
#
mean(diet2)
sd(diet2)
t.test(diet1)
t.test(diet2)

# p. 178 ダイエットデータの回帰分析 : diet1 = a + b wt1
diet.y <- diet1 ; diet.x <- wt1 
reg <- lm(diet.y ~ diet.x)
summary(reg)
# 各段階の計算（詳細）
diet.y
diet.x
sxy <- sum( (diet.x - mean(diet.x))*(diet.y - mean(diet.y)) )
sxx <- sum( (diet.x - mean(diet.x))^2 )
h.beta <- sxy/sxx
hat.y <- mean(diet.y) + h.beta * (diet.x-mean(diet.x))
fitted(reg) # hat.y に一致する
residuals(reg) # 残差 (diet.y - hat.y) に一致する
s2 <- sum((diet.y - hat.y)^2 / (length(diet.y)-2) )
(t.beta <- h.beta/sqrt(s2/sxx))

# p. 180 無相関性の検定
diet.r <- cor(diet.x,diet.y) #  0.3544978
sxy <- sum( (diet.x - mean(diet.x))*(diet.y - mean(diet.y)) )
sxx <- sum( (diet.x - mean(diet.x))^2 )
syy <- sum( (diet.y - mean(diet.y))^2 )
sxy/sqrt(sxx*syy) # 相関係数 diet.r に等しい

cov(diet.x,diet.y)*9 # sxx, sxy, syy と比較する 
var(diet.x)*9
var(diet.y)*9

sqrt(8) * diet.r /sqrt(1-diet.r^2) # t 値
cor.test(diet.x,diet.y)


# p. 172 例の読み方 （比率に関する片側検定） 
binom.test(2400-1250,2400, 0.5, alt="greater")
p <- 1250/2400 ; P <- 0.5
1-pnorm((p-0.5)/sqrt(P*(1-P)/2400))
1-pbinom(1249,2400,0.5) # 正確な確率
1-pnorm((1249.5/2400-0.5)/sqrt(P*(1-P)/2400)) # 連続修正


