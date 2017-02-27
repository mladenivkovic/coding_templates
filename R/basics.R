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







print("Assigning Vectors")

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
print("Handling vectors")

print("x")
x<-1:10
x
x[(1:5)]    # from element 1 to 5
x[-(3:7)]   # all but elements 3 to 7
x[8]<- -1
x
z<-c(1:3, NA,NaN) # fourth element saved as not assigned, fifth is not a number
z
is.na(z)      # logical array; true if element of Z is NA or NaN
is.nan(z)     # logical array; true if element of Z is NaN

print.noquote('')
print.noquote('')
print.noquote('')

###########################################
print("Vector arithmetic")

x <- c(11.1, 22.2, 33.3, 44.4)
y <- c(55.5, 66.6, 77.7, 88.8) 

print("x")
x
print("y")
y



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

#print.noquote('')
#print.noquote('')
#print.noquote('')
#print.noquote('')
#print.noquote('')
#print.noquote('')
#print.noquote('')
#print.noquote('')


print.noquote('')
print.noquote('')
print.noquote('')

print("Complex numbers")
print("sqrt(-17)")
sqrt(-17)


print.noquote('')
print("sqrt(-17+0i)")
sqrt(-17+0i)



print.noquote('')
print.noquote('')
print.noquote('')



######################################

print("Logicals")

a <- 2
b <- 3

a > b
a < b
a == b
a != b
a >= b
a <= b

print.noquote('')

t <- a < b
f <- a > b

t & f   # AND
t | f   # OR
!t      # NOT
!f      # NOT

