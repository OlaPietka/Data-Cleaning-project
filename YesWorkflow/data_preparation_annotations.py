#!/usr/bin/env python
# coding: utf-8

# # Description
# The purpose of this notebook is to take the "cleaned" csv file as an input, and then further refine/split the data for future SQL tables creation.
#
# ---

# # About the data
# This dataset (Ask A Manager Salary Survey 2021 dataset) contains salary information by industry, age group, location, gender, years of experience, and education level. The data is based on approximately 28k user entered responses.
#
# **Features:**
# - `timestamp` - time when the survey was filed
# - `age` - Age range of the person
# - `industry` - Working industry
# - `job_title` - Job title
# - `job_context` - Additional context for the job title
# - `annual_salary` - Annual salary
# - `additional_salary` - Additional monetary compensation
# - `currency` - Salary currency
# - `currency_context` - Other currency
# - `salary_context` - Additional context for salary
# - `country` -  Country in which person is working
# - `state` - State in which person is working
# - `city` - City in which person is working
# - `total_experience` -  Year  range of total work experience
# - `current_experience` - Year range of current field  work experience
# - `education` - Highest level of education completed
# - `gender` - Gender of the person
# - `race` - Race of the person

# # Reading the file

import pandas as pd
import numpy as np
import difflib
import re
import itertools
import enchant
from textblob import TextBlob

FILE = 'Data/salary_responses_clean.csv'

"""
@begin clean_data_with_python
"""

"""
@begin convert_to_df @desc Converts the input file to a pandas dataframe
@in FILE @URI file:'Data/salary_responses_clean.csv'
@out data @AS data
"""
data = pd.read_csv('Data/salary_responses_clean.csv')
data.info()
data['age'].value_counts()
data['age'].isnull().sum()
# @end convert_to_df

"""
@begin age_range_to_min @desc Creates a new column 'age_min' that cleans dashes and strings to output a valid minimum age
@in data
@out data_plus_age_min @desc Age min column
"""
def age_range_to_min(row):
    age_range = row['age']

    if '-' in age_range:
        age_min = age_range.split('-')[0]
    elif 'over' in age_range:
        age_min = age_range.split()[0]
    elif 'under' in age_range:
        return np.nan

    return int(age_min)

data['age_min'] = data.apply(lambda row: age_range_to_min(row), axis=1)
# @end age_range_to_min

"""
@begin age_range_to_max @desc Adjust age range to maximum
@in data_plus_age_min
@out data_plus_age_max
"""
def age_range_to_max(row):
    age_range = row['age']

    if '-' in age_range:
        age_max = age_range.split('-')[1]
    elif 'over' in age_range:
        return np.nan
    elif 'under' in age_range:
        age_max = age_range.split()[-1]

    return int(age_max)

data['age_max'] = data.apply(lambda row: age_range_to_max(row), axis=1)
# @end age_range_to_max

# ## Experience
# Same goes for `total_experience` and `current_experience` attributes.

# In[10]:
data['total_experience'].value_counts()


# In[11]:
data['total_experience'].isnull().sum()

"""
@begin experience_range_to_min @desc Creates a new total_experience_min column from total_experience
@in data_plus_age_max
@param experience
@param attribute
@out data_plus_total_experience_min
"""
# In[12]:
def experience_range_to_min(row, attribute):
    total_exp_range = row[attribute]

    if '-' in total_exp_range:
        total_exp_min = total_exp_range.strip().split('-')[0]
    elif 'more' in total_exp_range:
        total_exp_min = total_exp_range.split()[0]
    elif 'less' in total_exp_range:
        return np.nan

    return int(total_exp_min)
# @end experience_range_to_min

"""
@begin experience_range_to_max @desc Creates a new total_experience_max column from total_experience
@in data_plus_total_experience_min
@param experience
@param attribute
@out data_plus_total_experience_max
"""
def experience_range_to_max(row, attribute):
    total_exp_range = row[attribute]

    if '-' in total_exp_range:
        total_exp_max = total_exp_range.strip().replace('years', '').split('-')[1]
    elif 'more' in total_exp_range:
        return np.nan
    elif 'less' in total_exp_range:
        total_exp_max = total_exp_range.split()[0]

    return int(total_exp_max)
# @end experience_range_to_max

# In[13]:
data['total_experience_min'] = data.apply(lambda row: experience_range_to_min(row, 'total_experience'), axis=1)
data['total_experience_max'] = data.apply(lambda row: experience_range_to_max(row, 'total_experience'), axis=1)
# In[14]:
data['current_experience'].value_counts()

"""
@begin experience_current_to_min @desc Creates a new current_experience_min column from total_experience
@in data_plus_total_experience_max
@param experience
@param attribute
@out data_plus_current_experience_min
@end experience_current_to_min
"""

"""
@begin experience_current_to_max @desc Creates a new current_experience_max column from total_experience
@in data_plus_current_experience_min
@param experience
@param attribute
@out data_plus_current_experience_max
@end experience_current_to_max
"""

# In[13]:
data['current_experience_min'] = data.apply(lambda row: experience_range_to_min(row, 'current_experience'), axis=1)
data['current_experience_max'] = data.apply(lambda row: experience_range_to_max(row, 'current_experience'), axis=1)

# ## Education

# In[14]:


data['education'].value_counts()


# Clean up naming:

# In[15]:

"""
@begin clean_education_naming @desc Replaces all graduate professional degrees with 'Professional degree' string
@in data_plus_current_experience_max
@out data_replaced_education_naming
@end clean_education_naming
"""

data['education'].replace({"Professional degree (MD, JD, etc.)": "Professional degree"}, inplace=True)

# It would be nice to have some kind of knowledge about the actual "level" of education (e.g. 0 - High school, 1 - Some college, etc.). Lets map those values to their level:

# In[16]:

"""
@begin create_education_level @desc Creates a column 'education_lvl' from 'education' that assigns a numerical value to each education level
@in data_replaced_education_naming
@out data_plus_education_lvl
@end create_education_level
"""

data['education_lvl'] = data['education'].map({'High School': 1, 'Some college': 2, 'College degree': 3, "Master's degree": 4, 'Professional degree': 5})


# In[17]:


data[['education', 'education_lvl']].head()


# ## Gender

# In[18]:


data['gender'].value_counts()


# Clean up naming:

# In[19]:

"""
@begin cluster_other_gender @desc Cluster 'Other or prefer not to answer' into 'Other'
@in data_plus_education_lvl
@out data_replaced_other_gender
@end cluster_other_gender
"""

data['gender'].replace({"Other or prefer not to answer": "Other"}, inplace=True)


# Lets create some kind of mapping so it could be easier to use in SQL queries:

# In[20]:

"""
@begin create_gender_idx @desc Creates column 'gender_idx' replacing gender with a string to integer value map
@in data_replaced_other_gender
@out data_plus_gender_idx
@end create_gender_idx
"""

data['gender_idx'] = data['gender'].map({'Woman': 1, 'Man': 2, 'Non-binary': 3, "Other": 4})

# In[21]:

data[['gender', 'gender_idx']].head()

# In[22]:

data.info()

# # Race
# Values in `race` column are lists of all races that the user is identifying to (we are dealing with list of strings).

# In[126]:

data['race'].value_counts().tail()

# Let's create some mapping so it can be easier to analyze the data.

# In[142]:

"""
@begin map_race_to_index @desc Creates column 'race_idx' replacing race with a string to integer value map
@in data_plus_gender_idx
@param row @AS race_values
@out data_plus_race_idx
@end map_race_to_index
"""
race_map = {
    'Asian or Asian American': 1,
    'Black or African American': 2,
    'Hispanic, Latino, or Spanish origin': 4,
    'Middle Eastern or Northern African': 5,
    'Native American or Alaska Native': 6,
    'White': 7,
    "Another option not listed here or prefer not to answer": 8 }

def map_race_to_index(row):
    race = row['race']

    if type(race) != str:
        return np.nan

    races = []
    for r_key in race_map.keys():
        if r_key in race:
            races.append(race_map[r_key])
    return ','.join([str(r) for r in races])

data['race_idx'] = data.apply(lambda row: map_race_to_index(row), axis=1)

# In[143]:

data['race_idx'].value_counts()

# # Address attributes

# ## City

# In[23]:

data['city'].value_counts()

# In[24]:

try:
    from geopy import Nominatim
except:
    get_ipython().system('pip install geopy')
    from geopy import Nominatim

geolocator = Nominatim(user_agent="cs513-final-project")

# In[25]:

try:
    from geotext import GeoText
except:
    get_ipython().system('pip install geotext')
    from geotext import GeoText

"""
@begin get_city_from_text @desc Replaces 'city' columm values by its prefix followed by address type
@in data_plus_race_idx
@param row @AS city_values
@out data_replaced_city
@end get_city_from_text
"""
def get_city_from_text(row):
    city = row['city']

    if type(city) != str:
        return np.nan

    if city.strip().upper() == "REMOTE":
        return "Remote"

    places = GeoText(city)

    if(len(places.cities) > 0):
        return places.cities[0]

    location = geolocator.geocode(city, exactly_one=True, addressdetails=True, timeout=10)

    if location != None:
        location_keys = location.raw['address'].keys()
        if "town" in location_keys:
            return location.raw['address']['town']
        elif "city" in location_keys:
            return location.raw['address']['city']
        elif "hamlet" in location_keys:
            return location.raw['address']['hamlet']
        elif "village" in location_keys:
            return location.raw['address']['village']
        elif "place" in location_keys:
            return location.raw['address']['place']
        elif "municipality" in location_keys:
            return location.raw['address']['municipality']
        elif "township" in location_keys:
            return location.raw['address']['township']
        elif "county" in location_keys:
            return location.raw['address']['county']

    return np.nan

# In[26]:

data['city'] = data.apply(lambda row: get_city_from_text(row), axis=1)
data['city'].value_counts()

# ## Country

# In[27]:

try:
    import pycountry
except:
    get_ipython().system('pip install pycountry')
    import pycountry

"""
@begin get_country_from_text @desc Replaces 'country' columm values by its defined standard geographic location name
@in data_replaced_city
@param row @AS country_values
@out data_replaced_country
@end get_country_from_text
"""
def get_country_from_text(row):
    country = row['country']

    if type(country) != str:
        return np.nan

    if country.strip().upper() == "AMERICA":
        return "United States"

    places = GeoText(country)

    if(len(places.countries) > 0):
        return places.countries[0]

    location = geolocator.geocode(country, exactly_one=True, addressdetails=True, timeout=10)

    if location != None:
        if "country" in location.raw['address'].keys():
            country_code = location.raw['address']['country_code']
            return pycountry.countries.get(alpha_2=country_code).name

    return np.nan

# In[28]:

data['country'] = data.apply(lambda row: get_country_from_text(row), axis=1)

# ## State

# Next, let's examine rows with multiple `state` values and a `city` value containing only one word (not Remote).
# For each row, we will attempt to match the city to one of the states in the column.
# To do so, we will use the **geolocator** module

# In[29]:

try:
    from geopy.geocoders import Nominatim
except ImportError:
    get_ipython().system('pip install geopy')
    from geopy.geocoders import Nominatim

"""
@begin get_state_from_text
@desc Replaces and matches 'state' columm with the corresponding 'city' column
@in data_replaced_country
@param row @AS state
@out data_replaced_state
@end get_state_from_text
"""
def get_state_from_text(row):
    states = row['state']
    country = row['country']
    city = row['city']

    if type(country) != str or country != "United States":
        return np.nan

    if type(states) != str:
        return np.nan

    if type(city) != str or city == "Remote":
        return np.nan

    if ',' not in states:
        return states

    states = [x.strip() for x in states.split(',')]

    for state in states:
        lookup = f"{city}, {state}, {country}"

        location = geolocator.geocode(lookup, exactly_one = True, addressdetails = True, timeout=10)

        if location != None:
            if location.raw['address']['country'] == country:
                return location.raw['address']['state']

    return np.nan

# In[30]:

data['state'] = data.apply(lambda row: get_state_from_text(row), axis=1)
data['state'].value_counts()

# ## Continent
# Based on country value we can add new attribue `continent`

# In[31]:

try:
    from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
except:
    get_ipython().system('pip install pycountry-convert')
    from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2

continent_map = {
    'AF': 'Africa',
    'NA': 'North America',
    'OC': 'Oceania',
    'AN': 'Antarctica',
    'AS': 'Asia',
    'EU': 'Europe',
    'SA': 'South America',
}

"""
@begin get_continent_from_country @desc Creates 'continent' attribute based on the country
@in data_replaced_state
@param row @AS country
@out data_plus_continent
@end get_continent_from_country
"""
def get_continent_from_country(row):
    country = row['country']

    if type(country) != str:
        return np.nan

    country_code = country_name_to_country_alpha2(country)
    continent_code = country_alpha2_to_continent_code(country_code)

    return continent_map[continent_code]

# In[32]:

data['continent'] = data.apply(lambda row: get_continent_from_country(row), axis=1)
data['continent'].value_counts()

# ## Latitude, Longitude

# In[33]:

"""
@begin get_lat_long_from_full_address @desc Creates 'lat_long' attribute based on the address
@in data_plus_continent
@param city
@param country
@out data_plus_lat_long
@end get_lat_long_from_full_address
"""
def get_lat_long_from_full_address(city, country):
    if type(city) != str or type(country) != str:
        return np.nan

    full_address = f"{city}, {country}"

    location = geolocator.geocode(full_address, exactly_one=True, addressdetails=True, timeout=10)
    if location != None:
        return (location.latitude, location.longitude)
    else:
        return np.nan


# In[34]:

data['lat_long'] = data.apply(lambda row: get_lat_long_from_full_address(row['city'], row['country']), axis=1)

# In[35]:

data[['lat', 'long']] = pd.DataFrame(data['lat_long'].tolist(), index=data.index)

# # Text data
# Let's handle the text data.
#
# **Text features include:**
# - `industry`
# - `job_title`
# - `job_context`
# - `salary_context`

# ## Context
# Both `salary_context` and `job_context` features have too much information. It is basically a plain text provided by the user. As it does not help our analysis, we decided to drop those columns.

# In[36]:

"""
@begin drop_salary_context_and_job_context @desc Drops 'salary_context' and 'job_context' columns
@in data_plus_lat_long
@out data_dropped_salary_job_ctx
@end drop_salary_context_and_job_context
"""
data.drop(labels=['salary_context', 'job_context'], axis=1, inplace=True)

# ## Industry and job title

# In[37]:

data['job_title'].value_counts()

# In[38]:

data['industry'].value_counts()

# In[351]:

enchant_dic = enchant.Dict("en_US")

"""
@begin clean_word @desc Cleans strings with a series of replacements
@in data_dropped_salary_job_ctx
@param word @AS word_values
@out cleaned_word_values
@end clean_word
"""
def clean_word(word):
        # replace & with and
        if '&' in word:
            word = word.replace('&', 'and')

        # remove dots
        word = word.replace('.', '')

        # remove strings in parenthesis
        word = re.sub(r"\([^()]*\)", "", word)

        # remove all possible combinations of seniority strings
        seniority_filter = [s_map for s in ['sr', 'jr', 'mid', 'senior', 'junior', 'senor']
                            for s_map in map(''.join, itertools.product(*zip(s.upper(), s.lower())))]
        for seniority in seniority_filter:
            word = word.replace(seniority, '')

        # remove all possible combinations of seniority strings at the end of the string
        seniority_filter = [s_map for s in [' I', ' II', ' III', ' IV']
                            for s_map in map(''.join, itertools.product(*zip(s.upper(), s.lower())))]
        for seniority in seniority_filter:
            if word.endswith(seniority):
                word = word.replace(seniority, '')

        # remove numbers from the end of the string
        word = re.sub(r"\d+$", "", word)

        # remove multiple spaces
        word = re.sub(' +', ' ', word)
        # title and strip
        word = word.title().strip()

        return word

def cluster(possibilities, cutoff = 0.90, n = 10):
    possibilities_copy = possibilities.copy()

    clusters = {}

    for word in possibilities:
        word = clean_word(str(word))

        close_matches = difflib.get_close_matches(word, possibilities_copy, n, cutoff)

        if(len(close_matches) == 0):
            continue

        if(len(close_matches) == 1) & (close_matches[0] == word):
            continue

        for match in close_matches:
            clusters[match] = word
            possibilities_copy.remove(match)
    return clusters

# In[304]:

"""
@begin create_clusters_job_title @desc Creates a 'cluster_job_title' from column 'job_title' unique values
@in data_cluster_job_title
@param job_title
@out clusters_job_title
@end create_clusters_job_title
"""
possibilities = list([str(x) for x in data['job_title'].unique()])
clusters_job_title = cluster(possibilities, 0.95)

# In[306]:

data['job_title_clean'] = data['job_title'].replace(clusters_job_title)

# Second round:

# In[311]:

"""
@begin create_clusters_job_title_second @desc Creates a cluster based on possibilities for the job_title_clean column unique values
@in data_plus_job_title_clean
@in clusters_job_title
@param possibilities @AS possibilities
@param cutoff @AS cutoff
@out clusters_job_title_second
@end cluster_second
"""
possibilities = list([str(x) for x in data['job_title_clean'].unique()])
clusters_job_title = cluster(possibilities, 0.95)

# In[313]:

"""
@begin create_job_title_clean_second @desc Replaces the 'job_title_clean' column values with second word possibilities word clean and cluster
@in data_cluster_job_title
@param job_title_clean
@out data_replaced_job_title_clean
@end create_job_title_clean_second
"""
data['job_title_clean'].replace(clusters_job_title, inplace=True)

# Third round:

# In[314]:

"""
@begin create_job_title_clean_third @desc Replaces the 'job_title_clean' from column 'job_title_clean' with second word possibilities word clean and cluster
@in data_replaced_job_title_clean
@param job_title_clean
@out data_replaced_job_title_clean_3
@end create_job_title_clean_third
"""
possibilities = list([str(x) for x in data['job_title_clean'].unique()])

clusters_job_title = cluster(possibilities, 0.9)


# In[316]:

"""
@begin create_job_title_clean_third @desc Replaces the 'job_title_clean' from column 'job_title_clean' with second word possibilities word clean and cluster
@in data_replaced_job_title_clean_3
@param job_title_clean
@out data_replaced_job_title_clean_3
@end create_job_title_clean_third
"""
data['job_title_clean'].replace(clusters_job_title, inplace=True)


# In[327]:

len(data['job_title_clean'].unique())

# In[337]:

"""
@begin job_title_to_uppercase @desc Convert the 'job_title_clean' to uppercase.
@in data_replaced_job_title_clean_3
@param job_title_clean
@param job_title
@out data_job_title_uppercase
@end job_title_to_uppercase
"""
data[['job_title_clean', 'job_title']][data['job_title_clean'].str.upper() != data['job_title'].str.upper()].head(15)


# In[362]:

"""
@begin create_industry_possibilities @desc Creates a 'possibilities' list of industry unique values
@in data_job_title_uppercase
@out industry_possibilities
@end create_industry_possibilities
"""
possibilities = list([str(x) for x in data['industry'].unique()])

"""
@begin create_cluster_industry @desc Creates a clusters_industry bases on industry unique values
@in data_job_title_uppercase
@in industry_possibilities
@out data_ind
@end create_cluster_industry
"""
clusters_industry = cluster(possibilities, 0.95)


# In[363]:

"""
@begin create_industry_clean @desc Creates an 'industry_clean' from industry clustering
@in clusters_industry
@in data_job_title_uppercase
@out data_plus_industry_clean
@end create_industry_clean
"""
data['industry_clean'] = data['industry'].replace(clusters_industry)

# In[364]:


possibilities = list([str(x) for x in data['industry_clean'].unique()])

clusters_industry = cluster(possibilities, 0.90)


# In[365]:


data['industry_clean'].replace(clusters_industry, inplace=True)


# In[366]:


possibilities = list([str(x) for x in data['industry_clean'].unique()])

clusters_industry = cluster(possibilities, 0.8)


# In[367]:


data['industry_clean'].replace(clusters_industry, inplace=True)


# In[376]:


possibilities = list([str(x) for x in data['industry_clean'].unique()])

clusters_industry = cluster(possibilities, 0.86)


# In[379]:


data['industry_clean'].replace(clusters_industry, inplace=True)


# In[384]:


possibilities = list([str(x) for x in data['industry_clean'].unique()])

clusters_industry = cluster(possibilities, 0.85)


# In[386]:


data['industry_clean'].replace(clusters_industry, inplace=True)


# In[387]:


len(data['industry_clean'].unique())


# In[394]:


data[['industry_clean', 'industry']][data['industry_clean'].str.upper() != data['industry'].str.upper()].head(15)


# # Currency
# Here we have a little bit more work to do. The currency of the salary is defined by the `currency` attribute, but sometimes it can be also defined by the `currency_context`. We need to clean those 2 columns and merge them into one.

# ## Merge columns 

# In[44]:


data[['currency', 'currency_context']]


# In[45]:


data['currency'].value_counts()


# In[46]:


data['currency_context'].value_counts()


# As we can see, when the `currency` feature has value of 'Other', then the currency is defined by `currency_context`. Lets clean this up

# In[47]:


data['currency'] = np.where(data["currency"] == "Other", data['currency_context'], data["currency"])


# In[48]:


data.drop(labels=['currency_context'], axis=1, inplace=True)


# ## Handle AUD/NZD values

# In[49]:


def split_currencies(row):
    currency = row['currency']

    if currency != 'AUD/NZD':
        return currency

    country = row['country']

    if country == 'Australia':
        return 'AUD'
    if country == 'New Zealand':
        return 'NZD'
    return np.nan


# In[50]:


data['currency'] = data.apply(lambda row: split_currencies(row), axis=1)


# ## Clean manually

# In[51]:


data['currency'][data['currency'].str.len() > 3].value_counts()


# In[52]:


messy_currencies = data['currency'][data['currency'].str.len() > 3].to_list() + ['BR$', 'ARP']
right_currencies = ['ARP', 'INR', 'BRL', 'MXN', 'USD', 'PLN', 'CZK', 'NOK', 'ILS', 'USD', 'NIS', 'RMB', 'TWD', 'PHP', 'KRW', 'IDR', 'ILS', 'DKK', 'RMB', 'AUD', 'PLN', 'PHP', 'AUD', np.nan, 'ARS', 'ILS', 'PHP', 'ARP', 'PHP', 'INR', 'DKK', 'KRW', 'EUR', 'SGD', 'MXN', 'THB', 'THB', 'HRK', 'PLN', 'INR', 'SGD', 'BRL', 'ARS']

data['currency'] = data['currency'].replace(messy_currencies, right_currencies)


# In[71]:


data['currency'] = data['currency'].str.upper()


# ## USD rate
# To have a consistent analysis for the salary values, we need to have only one currency (e.g. USD).

# In[53]:


get_ipython().system('pip install forex_python')
from forex_python.converter import CurrencyRates


# In[54]:


currency_rates = CurrencyRates()


# In[427]:


import datetime


def to_USD_rate(row):
    currency = row['currency']
    datatime = datetime.datetime(2021, 4, 27) # day of publishing the form

    try:
        return currency_rates.get_rate(currency, 'USD', datatime)
    except:
        return np.nan


# In[428]:


data['USD_rate'] = data.apply(lambda row: to_USD_rate(row), axis=1)


# In[429]:


data[['currency', 'USD_rate']]


# ## Clean manually

# In[433]:


messy_rates = data['currency'][~data['USD_rate'].notna()].unique()

USD_rates_correct = {
    'ARP': 0.01,
    'ARS': 0.01,
    'TTD': 0.15,
    'BDT': 0.012,
    'NIS': 0.29,
    'RMB': 0.16,
    'TWD': 0.033,
    'LKR': 0.0048,
    'SAR': 0.27,
    'RM': 0.22,
    'CAD': 0.78,
    'NTD': 0.033,
    'GBP': 1.38,
    'NGN': 0.0024,
    'CHF': 1.094,
    'EUR': 1.2090
}


# In[434]:


for currency, rate in USD_rates_correct.items():
    data.loc[data.currency == currency, 'USD_rate'] = rate


# # Numeric data
# Lets handle numeric data now.
#
# **Numeric features include:**
# - `annual_salary`
# - `additional_salary`

# ## Annual salary

# In[79]:


data['annual_salary'].describe()


# In[62]:


def salary_to_USD(USD_rate, salary):

    if USD_rate == np.nan:
        return np.nan

    return salary * USD_rate


# In[63]:


data['annual_salary_USD'] = data.apply(lambda row: salary_to_USD(row['USD_rate'], row['annual_salary']), axis=1)


# ## Additional salary

# In[64]:


data['additional_salary_USD'] = data.apply(lambda row: salary_to_USD(row['USD_rate'], row['additional_salary']), axis=1)


# ## Total salary

# In[542]:


data['total_salary'] = data[['additional_salary', 'annual_salary']].sum(axis=1)


# In[65]:


data['total_salary_USD'] = data[['additional_salary_USD', 'annual_salary_USD']].sum(axis=1)


# ## Clean outliers

# In[97]:


fig = plt.figure(figsize=(10,5))
sns.boxplot(data.total_salary_USD)
plt.title('Box Plot', fontsize=15)
plt.xlabel('Total salary in USD', fontsize=14)
plt.show()


# In[102]:


# calculate upper and lower limits
upper_limit = data.total_salary_USD.mean() + 3 * data.total_salary_USD.std()
lower_limit = data.total_salary_USD.mean() -3 * data.total_salary_USD.std()

# show outliers
data[~((data.total_salary_USD < upper_limit) & (data.total_salary_USD > lower_limit))]


# In[105]:


# remove outliers
data = data.drop(data[~((data.total_salary_USD < upper_limit) & (data.total_salary_USD > lower_limit))].index)


# In[106]:


fig = plt.figure(figsize=(10,5))
sns.boxplot(data.total_salary_USD)
plt.title('Box Plot', fontsize=15)
plt.xlabel('Total salary in USD', fontsize=14)
plt.show()


# # Indexing
# We are forced to treat each record separately.

# In[586]:


data['index'] = data.index


# # Export data

# In[792]:


data.info()


# In[793]:


data.to_csv("Data/data_full.csv", index=False)


# ## Currency table
# ![image.png](attachment:image.png)

# In[681]:


currency_table = data[['currency', 'USD_rate']].copy()


# In[682]:


currency_table = currency_table.drop_duplicates()


# In[683]:


currency_table['currency_ID'] = range(1, len(currency_table) + 1)


# In[684]:


currency_table.head()


# ## Place table
# ![image.png](attachment:image.png)

# In[685]:


place_table = data[['continent', 'country', 'state', 'city', 'lat', 'long']].copy()


# In[686]:


place_table = place_table.drop_duplicates()


# In[687]:


place_table['place_ID'] = range(1, len(place_table) + 1)


# In[688]:


place_table.head()


# ## Position table
# ![image.png](attachment:image.png)

# In[689]:


position_table = data[['job_title_clean', 'industry_clean']].copy()


# In[690]:


position_table = position_table.rename(columns={"job_title_clean": "job_title", "industry_clean": "industry"})


# In[691]:


position_table = position_table.drop_duplicates()


# In[692]:


position_table['position_ID'] = range(1, len(position_table) + 1)


# In[693]:


position_table.head()


# ## Person table
# ![image.png](attachment:image.png)

# In[694]:


person_table = data[['gender_idx', 'race_idx', 'education_lvl', 'age', 'age_min', 'age_max', 'total_experience', 'total_experience_min', 'total_experience_max', 'current_experience', 'current_experience_min', 'current_experience_max']].copy()


# In[695]:


person_table = person_table.drop_duplicates()


# In[696]:


person_table['person_ID'] = range(1, len(person_table) + 1)


# In[697]:


person_table.head()


# ### Gender lookup table

# In[698]:


gender_lookup_table = pd.DataFrame({"Name": ['Woman', 'Man', 'Non-binary', 'Other'], "Index": [1, 2, 3, 4]})


# In[699]:


gender_lookup_table.head()


# ### Race lookup table

# In[700]:


race_lookup_table = pd.DataFrame({"Name": list(race_map.keys()), "Index": list(race_map.values())})


# In[701]:


race_lookup_table


# ## Employee table
# ![image.png](attachment:image.png)

# In[765]:


employee_table = pd.DataFrame()


# In[789]:


def to_index(row, table, columns, id_column):
    mask = pd.DataFrame(table[columns] == row[columns])
    results = mask.all(axis=1)
    print(table[results])
    return table[results][id_column]


# In[790]:


person_table_columns = ['gender_idx', 'race_idx', 'education_lvl', 'age', 'age_min', 'age_max', 'total_experience', 'total_experience_min', 'total_experience_max', 'current_experience', 'current_experience_min', 'current_experience_max']

employee_table['person_ID'] = data.head(20).apply(lambda row: to_index(row, person_table, person_table_columns, 'person_ID'), axis=1)


# In[791]:


employee_table


# In[779]:


position_table_columns = ['job_title', 'industry']

employee_table['position_ID'] = data.apply(lambda row: to_index(row, position_table, position_table_columns, 'position_ID'), axis=1)


# In[ ]:


place_table_columns = ['continent', 'country', 'state', 'city', 'lat', 'long']

employee_table['place_ID'] = data.apply(lambda row: to_index(row, place_table, place_table_columns, 'place_ID'), axis=1)


# In[ ]:


currency_table_columns = ['currency', 'USD_rate']

employee_table['currency_ID'] = data.apply(lambda row: to_index(row, currency_table, currency_table_columns, 'currency_ID'), axis=1)


# In[ ]:


salary_columns = ['annual_salary', 'annual_salary_USD', 'additional_salary', 'additional_salary_USD', 'total_salary', 'total_salary_USD']

employee_table[salary_columns] = data[salary_columns].copy()


# In[ ]:


employee_table


# In[555]:


data[person_table_columns]
"""
@end clean_data_with_python
"""
