from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame
df = pd.read_csv('/Users/abhijitghosh/Documents/DataScience/iris.csv')
#df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
plt.hist(df['sepal.width'], color='skyblue', edgecolor='black')

# Add titles and labels
plt.title('Histogram Example')
plt.xlabel('Sepal Width Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

median_sepalWidth = np.median(df['sepal.width'])
mean_sepalWidth = np.mean(df['sepal.width'])
print(median_sepalWidth)
print(mean_sepalWidth)

sepal_width_73rd_percentile = df['sepal.width'].quantile(.73)
print(sepal_width_73rd_percentile)

columns_to_plot = ['sepal.length', 'sepal.width', 'petal.length','petal.width']
sns.pairplot(df[columns_to_plot], markers='o')

data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)

plt.hist(PlantGrowth['weight'], bin=.3, color='skyblue', edgecolor='black')

weights=PlantGrowth['weight']
start_point = 3
bin_width = .3
bin_edges = np.arange(start_point,max(weights)+bin_width,bin_width)

plt.hist(weights, bins=bin_edges, edgecolor='black', color='skyblue')

# Add labels and title
plt.title('Histogram of Plant Weights')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x='group', y='weight', data=PlantGrowth, palette='coolwarm')
plt.show()

#tr1 and tr2 weights
trt1_weights = PlantGrowth[PlantGrowth['group'] == 'trt1']['weight']
trt2_weights = PlantGrowth[PlantGrowth['group'] == 'trt2']['weight']

#minimum count
min_trt2_weight = trt2_weights.min()
below_min_count = (trt1_weights < min_trt2_weight).sum()
print(below_min_count)
#percentage w.r.t. tr1
percentage_below = (below_min_count / len(trt1_weights)) * 100
print(percentage_below)

data = PlantGrowth[PlantGrowth['weight'] > 5.5]
frequency_table = data['group'].value_counts() 

labels_int = frequency_table.index.tolist() 
# convert those integers to strings
labels = list(map(str, labels_int)) 
values = frequency_table.values
print(frequency_table)
# Create the bar plot
sns.barplot(x=labels, y=values, color='red')
plt.title("Bar Plot (seaborn)")
plt.xlabel("Group")
plt.ylabel("Value")
plt.show()