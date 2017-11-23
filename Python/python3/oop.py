#!/usr/bin/python3


class Employee:
    """
    Common base class for all employees.
    """

    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("Total employee", Employee.empCount)

    def displayEmployee(self):
        print("Name: ", self.name, ", salary:", self.salary)



emp = Employee("Bernhard", 2000)
emp.displayCount()
emp.displayEmployee()

