# ml-assignment2

## Problem statement
The goal of this assignment is to build and evaluate machine learning models for a given dataset. 

## Dataset description
Following data set is used for this assignment:
- Estimation of Obesity Levels Based on Eating Habits and Physical Condition
  - https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
  - Further more https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub this was referred to understand the data set and its features.
- This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.
- Dataset: data/ObesityDataSet_raw_and_data_sinthetic.csv 
- Details of data basis EDA:

| Attribute             | Value                      |
|-----------------------|----------------------------|
| Total no. of rows     | 2111                       |
| Total no. of features | 16                         |
| Any missing value     | No                         |
| Problem type          | Multi-class classification |
| Target Variable       | NObeyesdad                 |
| No. of Target Classes | 7                          |

- Feature Description

| #  | Feature                        | Description                               |
|----|--------------------------------|-------------------------------------------|
| 1  | Gender                         | Gender of person                          |
| 2  | Age                            | Age of person                             |
| 3  | Height                         | Height of person                          |
| 4  | Weight                         | Weight of person                          |
| 5  | family_history_with_overweight | any family history                        |
| 6  | FAVC                           | Frequent consumption of high caloric food |
| 7  | FCVC                           | Frequency of consumption of vegetables    |
| 8  | NCP                            | Number of main meals                      |
| 9  | CAEC                           | Consumption of food between meals         |
| 10 | SMOKE                          | weather person smokes                     |
| 11 | CH20                           | Consumption of water daily                |
| 12 | SCC                            | Calories consumption monitoring                                           |
| 13 | FAF                            | Physical activity frequency                                          |
| 14 | TUE                            | Time using technology devices                                          |
| 15 | CALC                           |   Consumption of alcohol                                          |
| 16 | MTRANS                         |   Transportation used                                        |

- Target Variable Description
    - NObeyesdad (Obesity Level) - Target Variable
    - Values: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III