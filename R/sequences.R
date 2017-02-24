#!/usr/bin/R


#Generating sequences

print("a")
a<-c(1,2,3)
a

print("b")
b<-1:15
b

print("c")
c<-15:1
c



print.noquote("")


n<-10
d1<-1:n-1
d2<-1:(n-1)

print("d1")
d1
print("d2")
d2




print.noquote("")


e1<-seq(1,5)
e2<-seq(1,5,by=0.4)
e3<-seq(1,5,length=8)
e4<-seq(from=-5, by=1.2, length=8)

print("e1")
e1
print("e2")
e2
print("e3")
e3
print("e4")
e4



print.noquote("")


f<-c(3,4,5)
f1<-rep(f,times=5)
f2<-rep(f,each=5)


print("f1")
f1
print("f2")
f2

