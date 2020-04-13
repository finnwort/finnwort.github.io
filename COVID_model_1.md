# Machine Learning and COVID-19
## When we apply multiple regression to death, cases and country data, what do we learn about different countries and their approach to Coronavirus? 
So, the main aim here is to take a very basic approach and use it to look at the differences that countries make to the predicted deaths they have. Beware the things we learn from this are entirely contextual and it only means something if you understand the numbers we are using and the real-world situation. 

What is the plan? Applying machine learning to create regression estimates for death data per data across all countries, we can find out if certain countries expect more deaths even when the cases increases (so, general infection levels) is taken into account. By looking at this we can get an idea of two things, the efficacy of certain country's strategies and the discrepancy in testing strategies across certain countries. The difference between these two is not clear even post-analysis so this is an interesting case study in how we can have different explanations for the same results (higher coefficients for certain countries). 

What is the analysis? I have, in my quarantine-driven boredom, made some progress in something I set out to do a long time ago. That was to learn Python and data analytics using it. I have just learned some very (very very!) basic machine learning techniques. These techniques can be applied to COVID data and tell at least a little. The analysis itself uses by-day data for cases and deaths seperated for each country. 

What is interesting is this data is open to anyone who might like to use it. What is more interesting is how hard it is to understand. Because cases, deaths and country generally show a similar trend (with cases and deaths going in similar directions), we have a basic understanding of what the data is saying, but we don't know what more we might learn from parsing these things apart and looking at how bad and disproportionate deaths are compared to every other country! This is why machine learning can be so effective, when we have huge numbers of categorical variables, such as country IDs, the machine learning can do the looking for us. Exciting!


```python
# For this practical example we will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
```

We've loaded in the data, now it is time to look at the data and start cleaning it up. We will begin by removing all NAs. 


```python
# Load the data from a .csv in the same folder
raw_data = pd.read_csv('COVID.csv')

```


```python

```


```python
data = raw_data
data.isnull().sum()
```




    dateRep                      0
    day                          0
    month                        0
    year                         0
    cases                        0
    deaths                       0
    countriesAndTerritories      0
    geoId                       29
    countryterritoryCode       205
    popData2018                163
    dtype: int64




```python
data_no_mv = data.dropna(axis=0)
```


```python
sns.distplot(data_no_mv['deaths'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c1e51d110>




![png](output_7_1.png)



```python
data_1 = data_no_mv
data_cleaned = data_1.reset_index(drop=True)
```


```python
data_cleaned = data_cleaned.drop(['geoId', 'countryterritoryCode', 'popData2018', 'day', 'month', 'dateRep', 'popData2018', 'year'],axis=1)
```


```python
data_cleaned.columns.values
```




    array(['cases', 'deaths', 'countriesAndTerritories'], dtype=object)



Next, it's time for a lovely features of PANDAS. Get_dummies creates dummy variables for all the categorical variables so we don't have to go through painstakingly changing things to 0s and 1s. This is a lifesaver and the reason why python is so effective at this kind of analysis. 


```python
data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)
```


```python
data_preprocessed = data_with_dummies
```


```python
targets = data_preprocessed['deaths']

# The inputs are everything BUT the dependent variable, so we can simply drop it
inputs = data_preprocessed.drop(['deaths'],axis=1)
```


```python
# Import the scaling module
from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()
# Fit the inputs (calculate the mean and standard deviation feature-wise)
scaler.fit(inputs)
scaler
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
# Scale the features and store them in a new variable (the actual scaling procedure)
inputs_scaled = scaler.transform(inputs)
```

I have, rather arbritrarily, decided on an 80-20 split for training and testing. This seems to be the norm, I am not going to pretend to know any more than that.


```python
from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state
# To have the same split as mine, use random_state = 365
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
```


```python
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
y_hat = reg.predict(x_train)
```


```python
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
```


![png](output_21_0.png)



```python
sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)
```




    Text(0.5, 1.0, 'Residuals PDF')




![png](output_22_1.png)



```python
reg.score(x_train,y_train)
```




    0.7060159339924923




```python
reg.intercept_
```




    10.940808006809693



Judging by the graphs and the residuals, this model is not exceptional. It does perform to some extent, but this isn't amazing. This can be put down to all sorts of factors but ultimately, the model probably deals with data that is too varied, as certain countries have infections spread through them that gets to thousands of people and other countries have not even dealt with it yet. 


```python
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary[reg_summary['Weights'] > 0.2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>cases</td>
      <td>75.38</td>
    </tr>
    <tr>
      <td>2</td>
      <td>countriesAndTerritories_Algeria</td>
      <td>0.21</td>
    </tr>
    <tr>
      <td>17</td>
      <td>countriesAndTerritories_Belgium</td>
      <td>1.55</td>
    </tr>
    <tr>
      <td>63</td>
      <td>countriesAndTerritories_France</td>
      <td>7.75</td>
    </tr>
    <tr>
      <td>92</td>
      <td>countriesAndTerritories_Italy</td>
      <td>10.91</td>
    </tr>
    <tr>
      <td>128</td>
      <td>countriesAndTerritories_Netherlands</td>
      <td>1.02</td>
    </tr>
    <tr>
      <td>170</td>
      <td>countriesAndTerritories_Spain</td>
      <td>7.36</td>
    </tr>
    <tr>
      <td>174</td>
      <td>countriesAndTerritories_Sweden</td>
      <td>0.30</td>
    </tr>
    <tr>
      <td>188</td>
      <td>countriesAndTerritories_United_Kingdom</td>
      <td>4.21</td>
    </tr>
  </tbody>
</table>
</div>



## Country Coefficients

Here is what we really wanted to see. We can see how certain countries such as Iran have a higher coefficient and therefore generally more death, even when cases for the day is considered. A simplistic view of this is that Iran have dealt with the crisis very badly but of course we can't say that just from these coefficients. What we can say, though, is certain countries such as Spain, must have a huge number of tests, leading to a coefficient that needs to tune deaths down after accounting for cases.

In other words, a lower coefficient suggests that a country has fewer deaths on any given day, than they might expect to have given the number of cases being reported. 


```python
y_hat_test = reg.predict(x_test)
```


```python
plt.scatter(y_test, y_hat_test)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
```


![png](output_29_0.png)


Looking at this and the other graph of predictions and data, we can draw some conclusions about the model. Firstly, it performs okay in a general, broad view. Secondly, it performs badly in certain cases. In particularly, down the bottom the model over predicts deaths. A simply explanation is that the training data overwhelmingly included reports from countries that had a low number of cases and very few deaths yet. The model then overpredicts deaths at low numbers of cases because of the numbers of deaths seen at higher numbers of cases. This simply reflects the real-world lag between cases and people dying, which tends to be a couple of weeks. 


```python
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
```

    /Users/finnwort/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: overflow encountered in exp
      """Entry point for launching an IPython kernel.



```python
df_pf['Target'] = np.exp(y_test)
```


```python
y_test = y_test.reset_index(drop=True)
```


```python
df_pf['Target'] = y_test
df_pf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prediction</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3454.02</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.98</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.92</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.49</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.95</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2011</td>
      <td>1.09</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2012</td>
      <td>0.66</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2013</td>
      <td>351.04</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2014</td>
      <td>1.02</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2015</td>
      <td>1.30</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2016 rows × 2 columns</p>
</div>




```python
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
```


```python
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

```

## What did we learn? 

Well, personally I think the most striking thing is the difference between Spain, Italy, France and Germany. Italy perform massively worse even though qualitatively we have seen Spain deal with a similar situation. This, very simply, is probably down to two things. The amount of testing and the fact that Italy do have a rather old population and have seen a worse death % than anyone else. Some, maybe all, of this result could be due to testing though. This is further supported by the fact that Germany perform fantastically and the UK do  very badly. Two countries that seem to be on opposite ends of the testing spectrum. 


```python
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prediction</th>
      <th>Target</th>
      <th>Residual</th>
      <th>Difference%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1470</td>
      <td>0.99</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.80</td>
    </tr>
    <tr>
      <td>62</td>
      <td>0.99</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.80</td>
    </tr>
    <tr>
      <td>1027</td>
      <td>0.99</td>
      <td>1</td>
      <td>0.01</td>
      <td>1.27</td>
    </tr>
    <tr>
      <td>210</td>
      <td>3.04</td>
      <td>3</td>
      <td>-0.04</td>
      <td>1.46</td>
    </tr>
    <tr>
      <td>590</td>
      <td>0.98</td>
      <td>1</td>
      <td>0.02</td>
      <td>2.01</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>868</td>
      <td>0.02</td>
      <td>0</td>
      <td>-0.02</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>867</td>
      <td>12718510.61</td>
      <td>0</td>
      <td>-12718510.61</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>866</td>
      <td>0.89</td>
      <td>0</td>
      <td>-0.89</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>895</td>
      <td>0.99</td>
      <td>0</td>
      <td>-0.99</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>2015</td>
      <td>1.30</td>
      <td>0</td>
      <td>-1.30</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
<p>2016 rows × 4 columns</p>
</div>


