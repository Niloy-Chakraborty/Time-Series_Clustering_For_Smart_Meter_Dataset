### Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset
This project implements a prototype of time-series clustering of Smart Meter Dataset using different clustering techniques and distance metrics for better understanding of the smart meter distribution among different clusters.

#### DATASET
* The London Smart Meters dataset contains 1130 smart meters data from 1/1/2013 0:00 to 1/1/2014 0:00 at 30 minutes interval. Along with the smart meters, the dataset contains the acorn distribution of every smart meter.
* For each smart meter, the respective Tariff category (i.e. Standard, Dynamic Time of Use (dToU) etc.), Acorn Category (i.e. ACORN-H, ACORN-N, ACORN-Q, etc.: Total: 19) and Acorn Group (i.e. Comfortable, Adversity, Affluent, and ACORN-U, Total:4 ) are mentioned in the dataset.
----------------------------------------------------------

#### DATA ANALYSIS AND DATA CREATION

* The main dataset contains the smart meter data and their acorn distribution and it seems that for clustering only the smart meter usage data over time is needed. For this reason, The main dataset has been split into two DataFrames (DF).

* The first DF contains the smart meter time-series data (17568 rows, 1132 columns) and the second DF contains the acorn distribution of each smart meter (1130 rows, 4 columns).

* The intention is to cluster the smart meters according to their energy consumption. Hence, a "Transpose" of the dataset is created. This step increases the dimensionality of the dataset to 17568, which increases the resource complexity of the clustering process.

* To reduce the dimensionality and increase the performance of the whole process, a new DF is created, which contains few new features, weekly-total- consumption (53 columns), Monthly-total-consumption (13 columns) for smart meters.

* Difference between two weeks consumption and two months' consumption are added, which can improve the quality of the clusters.
-----------------------------------------------------------------

#### DATA PRE-PROCESSING
* At first it is found that one Meter ID is miss classified with Acorn Group Name "Acorn-". From the group details, it can be assumed that it belongs to "ACORN-U", as there is not any other acorn group name started with "ACORN". Still, this entry has been removed from the dataset
* It is very important to check whether there are any missing values in the dataset. Missing values are imputed with "linear interpolation" or the mean values. Although there are no missing values in the dataset, this can be done as a part of the standard approach.
* If some features contain large numbers, that can dominate the outcome while computing the distance metric between two data points, resulting in misleading results. So, Scaling has been done using MinMax Scaler so that no data can dominate others while computing the distance metric. All the monthly and weekly consumption related data have been scaled in such way.
* Some algorithms may be unable to tackle the categorical data, although they might carry some meaningful information. For this reason, encoding has been also added which can take care of the categorical values. The London dataset contains only float values, but the first column Meter Id is categorical. So this encoding script will affect the categorical meter IDs and change them to integer 1,2,3,4...
-----------------------------------------------------------


#### CLUSTERING ALGORITHMS
For clustering of the processed dataset, 4 unsupervised clustering algorithms are implemented as follows:

1.	K-Means Clustering
2.	Agglomerative Hierarchical Clustering
3.	DB Scan Clustering
4.	Clustering Using Auto Encoder
----------------------------------------------------------


#### EXPERIMENTS

The data along with the acorn groups and acorn categories are plotted for having an insight about how the Meter Data are distributed over the acorn groups and categories.

![image1](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image1.jpeg)


![image2](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image2.jpeg)

The first figure shows the acorn group-wise line plots for all the meter. The second plot is for the mean-median distribution of the data. It's quite interesting from the boxplot (second plot) that for Adversity, affluent and comfortable groups, the distribution is almost similar. Now, the same thing has been done for different acorn categories (19 in total). The following figure is the acorn category distribution, which reveals the fact that many categories are also distributed in the same manner.

--------------------------------------------------------------------------------------------------------------------------


