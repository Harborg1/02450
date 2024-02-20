# exercise 1.5.2
import importlib_resources
import numpy as np

# You can read data from excel spreadsheets after installing and importing xlrd
# module. In most cases, you will need only few functions to accomplish it:
# open_workbook(), col_values(), row_values()
import xlrd

# If you need more advanced reference, or if you are interested how to write
# data to excel files, see the following tutorial:
# http://www.simplistix.co.uk/presentations/python-excel.pdf}

# Get path to the datafile
filename = importlib_resources.files("dtuimldmtools").joinpath("data/penguins.xls")
print(filename)
# Print the location of the iris.xls file on your computer. 
# You should inspect it manually to understand the format and content
print("\nLocation of the iris.xlsx file: {}".format(filename))


# Load xls sheet with data
# There's only a single sheet in the .xls, so we take out that sheet
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=8)
# Try calling help(doc.row_values). You'll see that the above means
# that we extract columns 0 through 4 from the first row of the document,
# which contains the header of the xls files (where the attributen names are)

# Extract class names to python list, then encode with integers (dict) just as
# we did previously. The class labels are in the 5th column, in the rows 2 to
# and up to 151:
classLabels = doc.col_values(7, 1, 343)  # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])
print(y)

# Preallocate memory, then extract data to matrix X
X = np.empty((336, 7))


print("ASDSADS")
for i in range(7):
    X[:, i] = np.array(doc.col_values(i, 7, 343)).T
    

print(X)

print(X[0,0])

print(X[0,1])
print(X[0,2])
print(X[0,3])
print(X[0,4])
print(X[0,5])
print(X[0,6])


# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print(N)
print(M)
print(C)
