import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

file_path = 'C:\\Users\\Christian\\OneDrive\\Dokumenter\\GitHub\\02450\\penguins.xls'

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Check the number of columns in the DataFrame
num_columns = len(data.columns)

# Now you can proceed with the rest of your code
# Define features and target variable
features = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
target = ['species']

# Extract features (X) and target variable (y)
X = data.loc[:, features].values
y = data.loc[:,target].values
# Standardize features
x = StandardScaler().fit_transform(X)

print(X)
print(y)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['species']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['species'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

plt.show()


