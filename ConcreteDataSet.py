import importlib_resources
import numpy as np
import xlrd

filename = 
# Load xls sheet with data
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=4)
# Try calling help(doc.row_values). You'll see that the above means
# that we extract columns 0 through 4 from the first row of the document,
# which contains the header of the xls files (where the attributen names are)

# Extract class names to python list, then encode with integers (dict) just as
# we did previously. The class labels are in the 5th column, in the rows 2 to
# and up to 151:
classLabels = doc.col_values(4, 1, 151)  # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])