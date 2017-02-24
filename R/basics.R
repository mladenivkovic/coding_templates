#!/usr/bin/R

# The shebang is not really necessary...

# This is a comment.
# To load a script in R via the console, type
# R (to launch R)
# source("thisfile.R")
# to just execute the script, use Rscript thisfile.R from the console



#printing
print("Hello world!")
print.noquote(' ' )







#Assignment

x <- c(11.1, 22.2, 33.3, 44.4)  #assign vector to x
c(55.5, 66.6, 77.7, 88.8) -> y
assign("z",c(0.1, 0.2, 0.3))


print.noquote('')

print("x") 
x

print("1/x") 
1/x

print("y")
y

print("z")
z

print("z<-c(x,0,y,0,0)")
z<-c(x,0,y,0,0)
z


print.noquote('')
print.noquote('')
print.noquote('')







###########################################
print("Vector arithmetic")

print.noquote('')

print("v<- 2*x + y - 1")
v<- 2*x + y - 1
v

print.noquote('')
print("v^2")
v^2

print.noquote('')
print("sqrt(v)")
sqrt(v)

print.noquote('')
print("sqrt(v^2)")
sqrt(v^2)

print.noquote('')
print("log(v)")
log(v)

print.noquote('')
print("exp(v)")
exp(v)

print.noquote('')
print("exp(log(v))")
exp(log(v))

print.noquote('')
print("sin(v)")
sin(v)

print.noquote('')
print("cos(v)")
cos(v)

print.noquote('')
print("tan(v)")
tan(v)


print.noquote('')
print("max(v)")
max(v)

print.noquote('')
print("min(v)")
min(v)

print.noquote('')
print("sum(v)")
sum(v)

print.noquote('')
print.noquote('')
print.noquote('')
print.noquote('')
print.noquote('')
print.noquote('')
print.noquote('')
print.noquote('')


print.noquote('')
print.noquote('')
print.noquote('')

print("Complex numbers")
print("sqrt(-17)")
sqrt(-17)


print.noquote('')
print("sqrt(-17+0i)")
sqrt(-17+0i)
