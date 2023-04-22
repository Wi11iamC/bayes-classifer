import pandas as pd
import numpy as np


def main():

    # Define the directory where the CSV files are stored
    digits_directory = './results/digits/'
    
    # Initialize an empty list to store each individual CSV file as a dataframe
    dataframes = []
    
    for i in range(10):
        file_path = digits_directory + 'metrics_no_validation_digits_' + str((i+1) * 10) + '_percent_of_data.csv'
        df = pd.read_csv(file_path)
        df = df.iloc[:-4]
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        dataframes.append(df)
    

    df = pd.concat(dataframes, ignore_index=True)
    
    
    # Use pivot_table function to convert Metric column to headers
    df = df.pivot(columns='Metric', values='Value')
    
    print("Analyzing data for training on only Digits training data")
    print("Average Accuracy: ", df['Accuracy'].mean(skipna=True))
    print("Standard Deviation: ", df['Accuracy'].std(skipna=True))
    print("Average Prediction Error: ", 1 - df['Accuracy'].mean(skipna=True))
    print("Average Precision: ", df['Precision'].mean(skipna=True))
    print("Average Recall: ", df['Recall'].mean(skipna=True))
    print("Average F1 Score: ", df['F1 Score'].mean(skipna=True))
    print("Finished analyzing data for training on only Digits training data\n")


    # Define the directory where the CSV files are stored
    digits_directory = './results/digits/'
    
    # Initialize an empty list to store each individual CSV file as a dataframe
    dataframes = []
    
    for i in range(10):
        file_path = digits_directory + 'metrics_with_validation_digits_' + str((i+1) * 10) + '_percent_of_data.csv'
        df = pd.read_csv(file_path)
        df = df.iloc[:-4]
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        dataframes.append(df)
    

    df = pd.concat(dataframes, ignore_index=True)
    
    
    # Use pivot_table function to convert Metric column to headers
    df = df.pivot(columns='Metric', values='Value')
    
    print("Analyzing data for training on Digits training + validation data")
    print("Average Accuracy: ", df['Accuracy'].mean(skipna=True))
    print("Standard Deviation: ", df['Accuracy'].std(skipna=True))
    print("Average Prediction Error: ", 1 - df['Accuracy'].mean(skipna=True))
    print("Average Precision: ", df['Precision'].mean(skipna=True))
    print("Average Recall: ", df['Recall'].mean(skipna=True))
    print("Average F1 Score: ", df['F1 Score'].mean(skipna=True))
    print("Finished analyzing data for training on Digits training + validation data\n")


    # Define the directory where the CSV files are stored
    faces_directory = './results/faces/'
    
    # Initialize an empty list to store each individual CSV file as a dataframe
    dataframes = []
    
    for i in range(10):
        file_path = faces_directory + 'metrics_no_validation_faces_' + str((i+1) * 10) + '_percent_of_data.csv'
        df = pd.read_csv(file_path)
        df = df.iloc[:-4]
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        dataframes.append(df)
    

    df = pd.concat(dataframes, ignore_index=True)
    
    
    # Use pivot_table function to convert Metric column to headers
    df = df.pivot(columns='Metric', values='Value')
    
    print("Analyzing data for training on only Faces training data")
    print("Average Accuracy: ", df['Accuracy'].mean(skipna=True))
    print("Standard Deviation: ", df['Accuracy'].std(skipna=True))
    print("Average Prediction Error: ", 1 - df['Accuracy'].mean(skipna=True))
    print("Average Precision: ", df['Precision'].mean(skipna=True))
    print("Average Recall: ", df['Recall'].mean(skipna=True))
    print("Average F1 Score: ", df['F1 Score'].mean(skipna=True))
    print("Finished analyzing data for training on only Faces training data\n")


    # Define the directory where the CSV files are stored
    faces_directory = './results/faces/'
    
    # Initialize an empty list to store each individual CSV file as a dataframe
    dataframes = []
    
    for i in range(10):
        file_path = faces_directory + 'metrics_with_validation_faces_' + str((i+1) * 10) + '_percent_of_data.csv'
        df = pd.read_csv(file_path)
        df = df.iloc[:-4]
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        dataframes.append(df)
    

    df = pd.concat(dataframes, ignore_index=True)
    
    
    # Use pivot_table function to convert Metric column to headers
    df = df.pivot(columns='Metric', values='Value')
    
    print("Analyzing data for training on Faces training + validation data")
    print("Average Accuracy: ", df['Accuracy'].mean(skipna=True))
    print("Standard Deviation: ", df['Accuracy'].std(skipna=True))
    print("Average Prediction Error: ", 1 - df['Accuracy'].mean(skipna=True))
    print("Average Precision: ", df['Precision'].mean(skipna=True))
    print("Average Recall: ", df['Recall'].mean(skipna=True))
    print("Average F1 Score: ", df['F1 Score'].mean(skipna=True))
    print("Finished analyzing data for training on Faces training + validation data\n")


if __name__ == '__main__':
    main()
