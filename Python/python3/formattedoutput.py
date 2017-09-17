#!/usr/bin/python3



print( ('{0:8}{1:8}{2:8}'.format("names", "String1", "someotherstring")) )
print( ('{0:8}{1:8}{2:8d}'.format("ints", '1234'.zfill(7), 123)) )
print( ('{0:8}{1:8.4f}{2:8.1f}'.format("flts", 12.9, 18.7)) )
print( ('{0:12}{1:12.4E}{2:12.3E}'.format("flts", 1223420.9, 18.7998234)) )

