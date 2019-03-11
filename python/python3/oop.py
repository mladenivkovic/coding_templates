#!/usr/bin/env python3


#=======================================================
# Intro to object oriented programming
# https://www.python-course.eu/python3_inheritance.php
#=======================================================


#================
class Person:
#================

    def __init__(self, first, last):
        self.firstname = first
        self.lastname = last
        return

    def Name(self):
        return self.firstname + " " + self.lastname





#===========================
class Employee(Person):
#===========================


    def __init__(self, first, last, staffnum):
        Person.__init__(self,first, last)
        self.staffnumber = staffnum

        import random as r 
        self.__secret = r.randint(1,1000)
        print("You can use private vars inside the class:", self.__secret)
        return

    def GetEmployee(self):
        return self.Name() + ", " +  self.staffnumber

x = Person("Marge", "Simpson")
y = Employee("Homer", "Simpson", "1007")

print(x.Name())
print(y.GetEmployee())

try:
    print(y.__secret)
except AttributeError:
    print("AttributeError: 'Employee' object has no attribute '__secret'")
