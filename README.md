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


![image2](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image2.png)

The first figure shows the acorn group-wise line plots for all the meter. The second plot is for the mean-median distribution of the data. It's quite interesting from the boxplot (second plot) that for Adversity, affluent and comfortable groups, the distribution is almost similar. Now, the same thing has been done for different acorn categories (19 in total). The following figure is the acorn category distribution, which reveals the fact that many categories are also distributed in the same manner.

![image3](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image3.png)

Each algorithm has been applied for the unsupervised clustering of the Time-Series data and finally, different visualization techniques have been applied for deriving information from the clusters.

All the algorithms have been implemented for the cluster numbers 3-15 and the silhouette coefficient has been calculated for each clustering technique. It has been found that cluster number= 5 gives the best silhouette coefficient value.


All the algorithms, i.e. K-means, Hierarchical Clustering, DB Scan, and Autoencoder based K-means have been implemented with cluster number 5. Among these all, the Autoencoder based K-means shows the most promising result with the silhouette coefficient= 0.52, reconstruction accuracy of the autoencoder= 85% (approximately), and reconstruction loss = 0.002. The input dimension is (1130,132) and the output dimension for the encoder is (1130,10). The pre-processed Dataset has been used for this training.

The result obtained from the Autoencoder based K-means is shown below:

1. **Mean Cluster Visualisation** : The following figure shows the plot for the mean of the clustered data. The clusters seem well divided with a little overlapping.

![4](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image4.jpeg)


2. **Season Wise plot for the Clustered Meters:** The figures below shows the season-wise distribution of the clustered data. Two seasons are considered, i.e. Summer and Winter.

![5](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image5.jpeg)

![6](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image6.jpeg)

It has been found that during winter the energy consumption becomes almost Twice for cluster No 3. For other clusters also, the consumption increases significantly, which is reflected in the plots.

3. **Day wise Consumption for all the clusters:** This analysis shows how the clusters behave if the hours-wise average is calculated for the whole year and the data is plotted for a single week.

--------------------------------------------------------------------------------------------------------------------------


