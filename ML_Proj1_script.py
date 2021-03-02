import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

# Read relevant columns from data and filter out rows with nan values.
data = pd.read_csv(r'covid-data.csv')
data = data.drop_duplicates(subset='iso_code', keep = 'last')
data = data[['iso_code', 'continent', 'total_cases_per_million', 'total_deaths_per_million', 
             'population', 'population_density', 'median_age', 'gdp_per_capita', 'human_development_index',
             'life_expectancy', 'diabetes_prevalence']]
data = data.dropna()
data_numbers = data[['total_cases_per_million', 'total_deaths_per_million', 
             'population', 'population_density', 'median_age', 'gdp_per_capita', 'human_development_index',
             'life_expectancy', 'diabetes_prevalence']]

#Summary stats
means = data_numbers.mean()
stds = data_numbers.std()
medians = data_numbers.median()

# Plots to check for correlations
figscat, ax = plt.subplots(2, 2, sharey='row')
figscat.subplots_adjust(hspace=0.4, wspace=0.2)
ax[0,0].scatter(data.gdp_per_capita, data.total_deaths_per_million)
ax[0,1].scatter(data.diabetes_prevalence, data.total_deaths_per_million)
ax[1,0].scatter(data.population_density, data.total_deaths_per_million)
ax[1,1].scatter(data.total_cases_per_million, data.total_deaths_per_million)
ax[0,0].set(ylabel='total_deaths_per_million', xlabel='gdp_per_capita')
ax[0,1].set(ylabel='total_deaths_per_million', xlabel='diabetes_prevalence')
ax[1,0].set(ylabel='total_deaths_per_million', xlabel='population_density')
ax[1,1].set(ylabel='total_deaths_per_million', xlabel='total_cases_per_million')
figscat.savefig(r'C:\Users\Andreas\Desktop\ML_proj1plots\correlation.eps', format='eps', dpi=1000)


standardized_data = (data_numbers-data_numbers.mean())/data_numbers.std()

# Boxplot to check for outliers
fig = plt.figure()
standardized_data.boxplot(fontsize = 8)
plt.xticks(rotation=70)
fig.savefig(r'C:\Users\Andreas\Desktop\ML_proj1plots\box.eps', format='eps', dpi=1000)

# Histograms to check for normal distribution
fig0, ax = plt.subplots(3,3)
fig0.subplots_adjust(hspace=1, wspace = 0.5)
ax[0,0].hist(data_numbers.total_cases_per_million)
ax[0,1].hist(data_numbers.total_deaths_per_million)
ax[0,2].hist(data_numbers.population)
ax[1,0].hist(data_numbers.population_density)
ax[1,1].hist(data_numbers.median_age)
ax[1,2].hist(data_numbers.gdp_per_capita)
ax[2,0].hist(data_numbers.human_development_index)
ax[2,1].hist(data_numbers.life_expectancy)
ax[2,2].hist(data_numbers.diabetes_prevalence)
ax[0,0].set(title = 'total_cases_per_milliom')
ax[0,1].set(title = 'total_deaths_per_million')
ax[0,2].set(title = 'population')
ax[1,0].set(title = 'population_density')
ax[1,1].set(title = 'median_age')
ax[1,2].set(title = 'gdp_per_capita')
ax[2,0].set(title = 'human_development_index')
ax[2,1].set(title = 'life_expectancy')
ax[2,2].set(title = 'diabetes_prevalence')
fig0.savefig(r'C:\Users\Andreas\Desktop\ML_proj1plots\hist.eps', format='eps', dpi=1000)

# SVD
u,s,v = np.linalg.svd(standardized_data, full_matrices = True)

# plot to show amount of variation explained
variation = (s*s) / (s*s).sum()
variationFracSum = np.cumsum(variation)
figPCA1 = plt.figure()
plt.plot(variationFracSum)
plt.xlabel('Number of PCA components')
plt.ylabel('Amount of variation explained. Scale = 0-1')
figPCA1.savefig(r'C:\Users\Andreas\Desktop\ML_proj1plots\PCA1.eps', format='eps', dpi=1000)

# The 2 first Principal directions
v_1 = v[0,:]
v_2 = v[1,:]

# Projection into 2-d space
proj1 = data_numbers.dot(v_1)
proj2 = data_numbers.dot(v_2)

figPCA2 = plt.figure()
plt.scatter(proj1,proj2)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
figPCA2.savefig(r'C:\Users\Andreas\Desktop\ML_proj1plots\PCA2.eps', format='eps', dpi=1000)





