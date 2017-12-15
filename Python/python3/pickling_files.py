#!/usr/bin/python3

#=============================================
# Pickling data
# Store non-text (= binary) data in files
# for easy read-in
#=============================================


import pickle

#open file for writing
pickle_file = open('inputfiles/my_picklefile.pkl', 'w')

#create whatever you need
my_list = ["Johnny B. Goode", 21, 1.89234e-7, (7,3), ['2',4,'c']]

#dump the contents
pickle.dump(my_list, pickle_file)
pickle_file.close()



#unpacking files
pickle_backup = open('inputfiles/my_picklefile.pkl', 'rb')
recovered_list = pickle.load(pickle_backup)
print("Recovered list: ", recovered_list)

