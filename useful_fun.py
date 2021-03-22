import numpy as np
import pandas as pd

# EXAMPLE 
# Using list comprehension - https://stackoverflow.com/questions/52705809/index-of-substring-in-a-python-list-of-strings
# Numpy indexing documentation - https://numpy.org/devdocs/reference/arrays.indexing.html
# l = ['abc','day','ghi']
# [e.find('a') for e in l]

def findColumn(containsString,dataFrame):
    """
    \n Function returns a list of column names containing the passed in string.

    containsString - string to be searched.
    dataFrame - dataframe to find the column in. \n

    """
    dataFrame_copy = pd.DataFrame()
    dataFrame_copy = dataFrame.copy()
    col_names = np.array(dataFrame_copy.columns) # Column Names
    has_string = [i.find(containsString)==0 for i in col_names] # Boolean list of column name locations
    col_names = col_names[has_string] # return strings containg this string
    return col_names

# To build the dataframe, region definitions used in the report are given.

class Regions:
    def __init__ (self):
        # Define regions used in the report.

        # Firstly, is it a transition country or not?
        # The "transition region" as defined in the LiTS III Report. This includes Cyprus, Greece, and Turkey, which are not seen as 
        # historically part of the Eastern Bloc. This definition of "transition region" is used for the purposes of replication.
        self.transition = ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herz.', 'Bulgaria', 'Croatia', 'Cyprus', 'Estonia', 
        'FYR Macedonia', 'Georgia', 'Greece', 'Hungary', 'Kazakhstan', 'Kosovo', 'Kyrgyz Rep.', 'Latvia', 'Lithuania', 'Moldova', 'Mongolia', 
        'Montenegro', 'Poland', 'Romania', 'Russia', 'Serbia', 'Slovak Rep.', 'Slovenia', 'Tajikistan', 'Turkey', 'Ukraine', 'Uzbekistan']

        # And secondly, which region does it belong to?
        # Countries of Central Asia (CA)
        self.ca = ['Uzbekistan', 'Tajikistan', 'Kyrgyz Rep.', 'Kazakhstan', 'Mongolia']
        # Countries in central Europe and the Baltic states (CEB)
        self.ceb = ['Estonia', 'Slovenia', 'Latvia', 'Poland', 'Croatia', 'Lithuania', 'Slovak Rep.', 'Hungary']
        # Countries in south-eastern Europe (SEE)
        self.see = ['Albania', 'Cyprus', 'Montenegro', 'Serbia', 'Romania', 'Kosovo', 'Bosnia and Herz.', 'Bulgaria', 'FYR Macedonia', 'Greece']
        # Countries in eastern Europe and the Caucasus (EEC)
        self.eec = ['Azerbaijan', 'Georgia', 'Belarus', 'Armenia', 'Ukraine', 'Moldova']
        # A couple of western European countries for comparison
        self.we = ['Germany', 'Italy']
        # Standalone countries
        self.standalone = ['Czech Rep.', 'Russia', 'Turkey']

        # All countries
        self.allCountries = ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herz.', 'Bulgaria', 'Croatia', 'Cyprus', 'Estonia', 
        'FYR Macedonia', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Italy', 'Kazakhstan', 'Kosovo', 'Kyrgyz Rep.', 'Latvia', 'Lithuania', 'Moldova', 
        'Mongolia', 'Montenegro', 'Poland', 'Romania', 'Russia', 'Serbia', 'Slovak Rep.', 'Slovenia', 'Tajikistan', 'Turkey', 'Ukraine', 'Uzbekistan']

        # Ex soviet states
        self.ussr = ['Armenia', 'Azerbaijan', 'Belarus', 'Estonia', 'Georgia', 'Kazakhstan', 'Kyrgyz Rep.', 'Latvia', 'Lithuania', 'Moldova',
         'Russia', 'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan']


regions = Regions()

def transition_recoder(country):
    """ Return a boolean for whether a country is in the transition region or not.

    Parameters
    ----------
    country : string
        The country for which it is to be determined whether it is in the transition region or not.
    
    Returns
    -------
    in_transition : boolean
        Whether the country is in the transition region or not.
    """

    if country in regions.transition:
        in_transition = True
    else:
        in_transition = False
    
    return in_transition


def region_recoder(country):
    """ Return a string that represents the region in which a country is located.
    
    Parameters
    ----------
    country : string
        The country whose region is to be determined.
    
    Returns
    -------
    region : string
        The region of the country.
    """

    if country in regions.ca:
        region = 'C Asia'
    elif country in regions.ceb:
        region = 'C Europe & Baltics'
    elif country in regions.see:
        region = 'SE Europe'
    elif country in regions.eec:
        region = 'E Europe & Caucasus'
    elif country in regions.we:
        region = 'W Europe'
    elif country in regions.standalone:
        region = country
    
    return region


def ussr_recoder(country):
    """ Return a boolean for whether a country was in the Soviet Union or not.
    
    Parameters
    ----------
    country : string
        The country for which it is to be determined whether it was in the Soviet Union or not.
    
    Returns
    -------
    in_ussr : boolean
        Whether the country was in the Soviet Union or not.
    """

    if country in regions.ussr:
        in_ussr = True
    else:
        in_ussr = False

    return in_ussr


# Define a function that will calculate the survey-weighted life satisfaction level for each country or a group of countries.
def calc_mean_percentage(countries, df, measure, exclusions, criteria):
    """ Return the mean percentage of respondents in given countries who fulfilled certain criteria on a given measure.
    
    Paramaters
    ----------
    countries : array
        Countries for which we are finding the percentage of respondents who fulfilled certain criteria.
    df : dataframe
        The dataframe from which we draw our results.
    measure: string
        The measure of interest.
    exclusions: array
        Values of the measure of interest that are excluded from further analysis.
    criteria: array
        Values of the measure of interest that are included for calculating the mean percentage.
    
    Returns
    -------
    mean_percentage : number
        The mean percentage of respondents in given countries who fulfilled certain criteria on a given measure.
    
    """
    
    df_overall = df.copy()
    
    # Clean the dataframe by removing rows with the excluded values in the measure column.
    for i in np.arange(len(exclusions)):
        indices = df_overall[df_overall[measure] == exclusions[i]].index
        df_overall = df_overall.drop(indices)
    
    # Create a new dataframe that contains only those rows that passed the criteria on the measure column.
    df_select = pd.DataFrame()
    
    for j in np.arange(len(criteria)):
        df_select = df_select.append(df_overall.copy()[df_overall[measure] == criteria[j]])
    
    # If a string was passed to countries, convert it into an array.
    if type(countries) == np.str:
        countries = [countries]
    
    # Calculate the mean percentage(s).
    percentages = np.zeros(len(countries))

    for k in np.arange(len(countries)):
        percentages[k] = np.sum(df_select[df_select['Country'] == countries[k]]['Weight 1']) / np.sum(df_overall[df_overall['Country'] == countries[k]]['Weight 1']) * 100
    
    mean_percentage = np.sum(percentages) / len(percentages)
    
    return mean_percentage
