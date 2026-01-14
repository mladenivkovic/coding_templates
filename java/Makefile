default: all


all: HelloWorld.class Cow.class OOPBasics.class


HelloWorld.class: HelloWorld.java
	javac HelloWorld.java

OOPBasics.class: OOPBasics.java
	javac OOPBasics.java


# wildcard
%.class: %.java
	javac $<


.PHONY: clean

clean:
	rm -f *.class
