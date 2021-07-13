### Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset
This project implements a prototype of time-series clustering of Smart Meter Dataset using different clustering techniques and distance metrics for better understanding of the smart meter distribution among different clusters.

#### DATASET
* The London Smart Meters dataset contains 1130 smart meters data from 1/1/2013 0:00 to 1/1/2014 0:00 at 30 minutes interval. Along with the smart meters, the dataset contains the acorn distribution of every smart meter.
* For each smart meter, the respective Tariff category (i.e. Standard, Dynamic Time of Use (dToU) etc.), Acorn Category (i.e. ACORN-H, ACORN-N, ACORN-Q, etc.: Total: 19) and Acorn Group (i.e. Comfortable, Adversity, Affluent, and ACORN-U, Total:4 ) are mentioned in the dataset.
----------------------------------------------------------

#### DATA ANALYSIS AND DATA CREATION


* The first DF contains the smart meter time-series data (17568 rows, 1132 columns) and the second DF contains the acorn distribution of each smart meter (1130 rows, 4 columns).

* The intention is to cluster the smart meters according to their energy consumption. Hence, a "Transpose" of the dataset is created. This step increases the dimensionality of the dataset to 17568, which increases the resource complexity of the clustering process.

* To reduce the dimensionality and increase the performance of the whole process, a new DF is created, which contains few new features, weekly-total- consumption (53 columns), Monthly-total-consumption (13 columns) for smart meters.

* Difference between two weeks consumption and two months' consumption are added, which can improve the quality of the clusters.
-----------------------------------------------------------------

#### DATA PRE-PROCESSING
* At first it is found that one Meter ID is miss classified with Acorn Group Name "Acorn-". From the group details, it can be assumed that it belongs to "ACORN-U", as there is not any other acorn group name started with "ACORN". Still, this entry has been removed from the dataset
* It is very important to check whether there are any missing values in the dataset. Missing values are imputed with "linear interpolation" or the mean values. Although there are no missing values in the dataset, this can be done as a part of the standard approach.
* If some features contain large numbers, that can dominate the outcome while computing the distance metric between two data points, resulting in misleading results. So, Scaling has been done using MinMax Scaler so that no data can dominate others while computing the distance metric. All the monthly and weekly consumption related data have been scaled in such way.
* Some algorithms may be unable to tackle the categorical data, although they might carry some meaningful information. For this reason, encoding has been also added which can take care of the categorical values. The London dataset contains only float values, but the first column Meter Id is categorical. So this encoding script will affect the categorical meter IDs and change them to integer 1,2,3,4.....
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

![7](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image7.jpeg)

It has been found that, on Friday, Cluster 3 consumes the highest energy. Cluster 1 consumes the lowest energy throughout the week. The peak consumption hour has been recorded between 16:40 to 22:13:20, where cluster 3 consumes the highest energy between 0.7-0.8 units.

4. **Resampled daily consumption for each cluster:** The clustered data has been resampled for a daily basis and the following figure shows the overall consumption throughout the year.

![8](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image8.jpeg)

The following figure shows the Quarterly distribution of the consumption, where it can be noted that in the quarter= 3, all the clusters records the lowest consumption.

![9](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image9.png)

5. **Acorn Group Wise Distribution of clusters:** Now let's look at the acorn-wise distribution of each cluster.

![10](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image10.png)

This figure shows the acorn-group distribution of the clusters. It is found that every cluster contains Meters from every acorn group. This is because of the fact the clustering method has been implemented without considering the acorn groups and there are many Meters from all the acorn groups where they follow similar consumption patterns.

6. **Pearson Correlation Analysis with the Global Radiation Pattern Data:** The following box plot shows the correlation between the Global Radiation pattern data and the clustered data. The correlation found is not satisfactory.

![11](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image11.png)

7. **Self-Consumption Computation for all the meter IDs:** The Meter IDs have been scaled to a value to 1000KWh, the Radiation data (PV field) has also been scaled in the same manner. Finally, for each meter ID, the self-consumption amount has been calculated and stored in a csv file. For each acorn group and acorn category, the self-consumption data has been plotted in the Violin plot. As it is not possible to show the whole data in an interactive manner, a Bokeh-HoloView based plotting API has been used here. 


8. **Percentage of Self-Consumption for each Acorn Group and Acorn Category:** For every Acorn group and Acorn Category, the percentage of the self-consumption has been plotted. The plots are shown below:

![12](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image12.jpeg)

This plot shows the percent of self-consumption of all the acorn groups. It is visible that, for all the acorn groups the density distribution of selfconsumption is between 30-40%. Even the median values also lie between this range.

![13](https://github.com/Niloy-Chakraborty/Time-Series_Stream_Clustering_on_London_Smart_Meter_Dataset/blob/master/IMAGES/image13.jpeg)

This plot shows the percent of self-consumption of all the acorn categories. It is also evident that, for all the acorn categories, the density distribution of self-consumption is between 30-40%. Even the median values also lie between this range. There is one exception in case of ACORN-B, where the selfconsumption lies below other categories.


--------------------------------------------------------------------------------------------------------------------------


#### REMARKS

* In this project, the London Time Series dataset has been studied thoroughly. The initial steps include the cleaning and pre-processing of data and later;several algorithms have been implemented for grouping the Meter IDs into clusters, using Unsupervised Learning methods.

* Among different methods, the Autoencoder based K-Means performs the best with the silhouette Score = 0.52, Acuracy= 85% and Loss= 0.002.

* Although the Autoencoder based K-means clustering method clusters the Meters into several distinct groups, the mapping between the clusters and the Acorn Groups remain unsatisfactory. It is found that the clustered Meter IDs don't correlate to the ACorn Groups or Acorn Categories. Meters from different Acorn groups or Acorn Categories form the same clusters.

* One possible reason for this could be, The energy consumption records from the acorn groups or acorn categories are not distinct from each other. In fact, the initial plots show that the mean, median, and the density distribution of the Meters IDs (Meter Ids from different Acorn groups and Categories) are similar. So, while clustering, these Meter IDs form the same clusters

--------------------------------------------------------------------------------------------------------------------------


