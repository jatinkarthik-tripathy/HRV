"""
Takes the row values for the item for whom the KNN is to be done to calculate CKT Distance as input_arr.
Take the .csv file's path as csv_file and reads the file to store as a list. 
Uses the calcDistance function to calculate the overall distance for all the rows (items present in the .csv file).
Appends all the data 
"""

from csv import *
from array import *

def calcDistance(input_arr, data_row):
    mean_factor = float(len(input_arr)**(-1))
    sum = 0.0
    for i in range(len(input_arr)):
        x = input_arr[i]
        y = data_row[i]
        sum += float((abs(x**3-y**3)**(1/2))/(x+y))
    Distance =  float(mean_factor * sum)
    return Distance

def CKTD(csv_file, input_arr):
    distance_col = []
    data = []
    with open(csv_file, 'r') as csvfile:
        data = list(csv.reader(csvfile))
    data[0].append("CKT_Distance")
    header = next(data)
    for i in range(len(input_arr)):
        data[i+1].append(calcDistance(input_arr, data[i+1]))
    dict_data_list = [dict(zip(header, map(float, row))) for row in data]
    return dict_data_list



