import numpy as np
import pandas as pd

# THIS FILE CONTAINS FUNCTIONS AND DEFINITIONS THAT WERE USED ACCROSS MULTIPLE NOTEBOOKS 
# (Primarily in the classification task)

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

# # Define a function that will calculate the survey-weighted life satisfaction level for each country or a group of countries.
def calc_mean_percentage(countries, df, measure, exclusions, criteria):
    """
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
    # Clean the dataframe by removing rows with the excluded values in the measure column.
    df_overall = exclude_values(df, measure, exclusions)
    # Create a new dataframe that contains only those rows that passed the criteria on the measure column.
    df_select = include_values(df, measure, criteria)
    
    # If a string was passed to countries, convert it into an array.
    if type(countries) == np.str:
        countries = [countries]
    
    # Calculate the mean percentage(s).
    percentages = np.zeros(len(countries))
    for k in np.arange(len(countries)):
        percentages[k] = np.sum(df_select[df_select['Country'] == countries[k]]['Weight 1']) / np.sum(df_overall[df_overall['Country'] == countries[k]]['Weight 1']) * 100
    
    mean_percentage = np.sum(percentages) / len(percentages)
    
    return mean_percentage


def exclude_values(df, measure, exclusions):
    # Clean the dataframe by removing rows with the excluded values in the measure column.
    df_overall = df.copy()

    # if more than one meassure is passed in.
    if len(measure) > 0:
        for elem in measure:
            for i in np.arange(len(exclusions)):
                indices = df_overall[df_overall[elem] == exclusions[i]].index
                df_overall = df_overall.drop(indices)
    # if there is only one measure.
    else:
        for i in np.arange(len(exclusions)):
            indices = df[df[measure] == exclusions[i]].index
            df_overall = df_overall.drop(indices)
    
    return df_overall


def include_values(df, measure, criteria):
    # Create a new dataframe that contains only those rows that passed the criteria on the measure column.
    df_select = pd.DataFrame()
    for j in np.arange(len(criteria)):
        df_select = df_select.append(df[df[measure] == criteria[j]])
    return df_select

def standard_units(x):
    return (x - np.mean(x))/np.std(x)



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Functions used for k-NN classifier implementation
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# I'm using slightly modded versions of the functions from class exercises

def distances(training, example, output):
    """Compute the distance from example for each row in training."""

    training_drop_output = training.drop(columns=output)
    example_drop_output = example.drop(index=output)

    out = training.copy()
    out['Distance'] = fast_distances(example_drop_output,training_drop_output)

    return out

def fast_distances(test_row, train_rows):
    """Array of distances between `test_row` and each row in `train_rows`.

    Parameters
    ----------
    test_row: attribute series / array
        A row of a table containing features of one test song (e.g.,
        test_20.iloc[0]).
    train_rows: data frame
        A table of features (for example, the whole table train_20).

    Returns
    -------
    distances : array
        One distance per row in `train_rows`.
    """
    # Convert the test row to an array of floating point values.
    test_row = np.array(test_row).astype(np.float64)
    # Convert the training attributes data frame to an array of floating point
    # values.
    train_attrs = np.array(train_rows).astype(np.float64)
    # Make an array of the same shape as `train_attrs` by repeating the
    repeated_test_row = np.tile(test_row, [len(train_rows), 1])
    # Now we can do the subtractions all at once.
    diff = repeated_test_row - train_attrs
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances

def closest(training, example, k, output):
    """Return a table of the k closest neighbors to example."""
    dist_df = distances(training, example, output)
    return dist_df.sort_values('Distance').head(k)


def predict_nn(example, training, k, output):
    """Return average of the price across the 5 nearest neighbors.
    """
    k_nearest = closest(training, example, k, output)

    result = k_nearest[output].value_counts().index[0]

    return result


def calculate_accuracy(resultsDF,Output):
    """
    Function calculates and prints class and total accuracy given the dataframe
    and the output class column
    """
    # Create a column, that indicates if the guess was correct
    resultsDF['Correct'] = resultsDF['Guesses'] == resultsDF[Output]
    

    # Count correct, incorrect guesses per class
    accuracy = resultsDF.value_counts(['Guesses', 'Correct'])
    
    # For each class print the achieved accuaracy
    cat = np.unique(resultsDF[Output])
    for value in cat:
        # Calculate accuracy fot this class
        achieved_accuracy =  (accuracy[value][True]/accuracy[value].sum())*100
        print('{} - {:2.2f}%:'.format(value,achieved_accuracy))

    # Calculate and print total accuracy
    accuracy = resultsDF.value_counts('Correct')
    achieved_accuracy =  (accuracy[True]/accuracy.sum())*100
    print('\nTotal Accuracy - {:2.2f}%:'.format(achieved_accuracy))


def train_test_valid(dataframe,train_prop = 7/10 ,valid_prop = 1/10):
    """
    Function splits the dataset into training, validation and tests sets and retruns dataframes for each set.
    """
    participants = len(dataframe)
    num_train = int(round(participants * train_prop))
    num_valid = int(round(participants * (train_prop+valid_prop)))

    # Shuffle the dataframe
    dataframe = dataframe.sample(frac=1,replace=False)

    # Assign parts of the shuffled dataframe to training and test sets
    train = dataframe.iloc[:num_train,:].copy()
    valid = dataframe.iloc[num_train:num_valid,:].copy()
    test= dataframe.iloc[num_valid:,:].copy()

    print("Training set:\t", len(train), "examples")
    print("Valid set:\t", len(valid), "examples")
    print("Test set:\t", len(test), "examples")
    return train, test, valid


def balance_df(dataframe,Output,bootstrap = False):
    """
    Passing in the dataframe and the output category, returns a dataframe with a 
    balanced number of output clases
    """
    categories = np.unique(dataframe[Output]) # the unique classes

    balancedDf = pd.DataFrame()
    if bootstrap == False:
        # Find the maximum number of rows without replacement
        max_len = dataframe[Output].value_counts(ascending=True)[0]

        for cat in categories:
            df = dataframe[dataframe[Output]==cat].copy()
            df = df.sample(frac=1,replace=False).iloc[:max_len,:]
            balancedDf = pd.concat([balancedDf,df])
    else:
        # Find the maximum number of rows with replacement
        max_len = dataframe[Output].value_counts()[0]

        for cat in categories:
            df = dataframe[dataframe[Output]==cat].copy()
            df = df.sample(frac=2,replace=True).iloc[:max_len,:]
            balancedDf = pd.concat([balancedDf,df])
    return balancedDf


def column_values(df):
    """
    Function prints the unique column values in each column given a df.
    """
    for col in df.columns:
        # unique = np.unique(lits_2016_selected[col])
        unique = df[col].unique()
        # for i in 
        if len(unique) < 14:
            print('{}\n {}, Members: ({})'.format(col,len(unique),unique))
        else:
            print('{}\n {}, Members: (mora than 10)'.format(col,len(unique)))