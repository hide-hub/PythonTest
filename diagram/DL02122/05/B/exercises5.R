# ５章練習問題
# 問 5.1
x <- 15 ; n <- 20 ; p0 <- 1/2
abs(x - n*p0)/sqrt(n*p0*(1-p0))
abs(x - n*p0)/sqrt(n*p0*(1-p0)) >= 1.96

# 別解
dbinom(15,n,p0)
cond <- dbinom(0:20,n,p0) <= dbinom(15,n,p0)
sum( dbinom(0:20,n,p0)[cond] )
binom.test(15,20,p=0.5)

# 問 5.2
# Michaelson-Morley : light speed 光速
lightspeed <- read.table("lightspeed.txt",header=T)
names(lightspeed)
attach(lightspeed)
stem(speed)
qqnorm(speed) ; qqline(speed) # 正規性の確認

# mean(speed) : 909.0, sd(speed) : 104.93
s <- 104.93; n <- 20 
(t <- sqrt(n)*(mean(speed)-990)/s)
(t <- sqrt(n)*(mean(speed)-792)/s)
t.test(speed, mu=990)
t.test(speed, mu=792)

# 問 5.3
sum( dbinom(4999:5001, size=10000, p=1/2) )
sum( dbinom(4998:5002, size=10000, p=1/2) )
sum( dbinom(4997:5003, size=10000, p=1/2) )


# 問 5.4
# (1)
500 - 1.645* 100/sqrt(40)
( z <- sqrt(40)*(485-500)/100 )

# (2)
485 + c(-1,1)* 1.96 * 100/sqrt(40)
485 + 1.645 * 100/sqrt(40)

# (3)


# 問 5.5
# (1)
sqrt(16)*(207-200)/10 # = 2.80 
2 * pnorm(2.80, lower=F)

# (2)
207 +c(-1,1)* 1.96 *10/sqrt(16) 

# (3)
207 +c(-1,1)* 2.13 *10/sqrt(16) 


# 問 5.6
# (3)
( z <- (65.0 - 70.0)/6.32/sqrt(1/20+1/20) )

2*pnorm(z,low=T)

# 問 5.7
# (1)
qchisq(c(.025,.975),df=149)
149 * 434.537/400 
149 * 480.10/400 

# (2)
(149 * 434.537 + 149 * 480.10) /(149 + 149)
s <- sqrt((434.537 + 480.10)/2)
t <- (144.75 - 155.57)/ (21.39 * sqrt(1/150+1/150))
qt(.975,df=298)

# (3)
-10.83 \pm 2.78 \times 3.891/\sqrt{5}
(-10.83)/(3.891/sqrt(5)) 
2*pt(-6.22,df=4) #  P-値 (p-value)

# 問 5.8
# 分割表
eij <- 120 * outer(c(63,57)/120 , c(92,28)/120)
oij <- matrix(c(52,11, 40,17), nrow = 2, ncol=2, byrow=TRUE)
sum((oij-eij)^2/eij)
# 3.7^2*sum(1/eij)
pchisq(2.56, df=1,lower=F)

# 二項分布
p1 <- 52/63 ; p2 <- 40/57 ; ps <- 92/120
z <- (p1-p2)/sqrt(ps*(1-ps)*(1/63+1/57))
(0.8254-0.7018)/sqrt(0.7667*(1-0.7667)*(1/63+1/57)) 
2*pnorm(1.60,lower=F) # P-値