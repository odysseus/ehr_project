import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import feature_column

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_code_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''

    lookup = df.ndc_code.map(lambda x: ndc_code_df[ndc_code_df.NDC_Code == x]['Non-proprietary Name'])
    newcol = pd.Series([r.values[0] if len(r.values) > 0 else 'None' for r in lookup])
    df['generic_drug_name'] = newcol

    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    eids = df.groupby('patient_nbr').encounter_id.agg(lambda x: list([y for y in x if y is not None]))
    fenc = [sorted(lst)[0] for lst in eids]
    first_encounter_df = df[df.encounter_id.isin(fenc)]

    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train, holdout = train_test_split(df, train_size=0.6, shuffle=True)
    validation, test = train_test_split(holdout, train_size=0.5, shuffle=True)

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......
        '''
        cat = feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path)
        col = feature_column.indicator_column(cat)
        output_tf_list.append(col)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    zscore = lambda c: normalize_numeric_with_zscore(c, MEAN, STD)
    tf_numeric_feature = feature_column.numeric_column(col, normalizer_fn=zscore, default_value=default_value)

    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = (df[col] >= 5).astype(int).to_numpy()

    return student_binary_prediction
