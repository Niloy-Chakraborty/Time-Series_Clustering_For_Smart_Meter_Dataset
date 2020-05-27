"""
Script : Visualization.py
Definition: This script is written for the following functionality :
    1. Basic visualization of the dataset
    2. Subplot of clustered data
    3. Overlay plot of the clustered data
    4. Acorn distribution plot based on the clusters
    5. Hourly mean plot of all the clusters
    6. Creation of directory and saving the plots

"""

# import libraries
import pandas as pd    # version 0.23.4 , because of some bugs in the latest version 0.24
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""

Function : visualiseData
Definition: this function takes the dataset 'df1' as argument, and shows the basic visualization of the data.
Return: none

"""


def visualise_data(dict_cluster,df1,n_cluster):
    # df1= df1.sort_values(by=['Date'])
    cluster_list1 = dict_cluster["3"]
    print (cluster_list1)
    cluster_list2 = dict_cluster["1"]
    print(cluster_list2)

    cols = df1.columns


    plt.matshow(df1.corr())
    plt.xticks(range(df1.shape[1]), df1.columns, fontsize=14, rotation=45)
    plt.yticks(range(df1.shape[1]), df1.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

    '''
    for i in range(n_cluster):
        df = pd.DataFrame(df1[dict_cluster[str(i)]])
        plt.matshow(df.corr())
        plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16);
        plt.show()
    #df1[cluster_list1].corr()
    '''

    '''
    for i in housename:

        df = df1[df1['Housename'] ==i]

        # df1["time"] = pd.to_datetime(df1["time"], format='%Y-%m-%d %H:%M:%S')
        # df.set_index(['time'], inplace=True)

        df.plot()
        plt.title("Plot for Housename: "+i)

        plt.savefig("./Figs/visualise_data/visualisation.png", dpi = 150)
        plt.show()
        # print(df.info())
    '''

"""
Function : cluster_visualisation_subplot
Definition: this function takes processed df and labels as arguments, and does the following:
            1. plots the labeled clusters in a subplot.
Return: 

"""


def cluster_visualisation_subplot(df1, labels,choice,n_clusters):
    df1= df1.set_index("time")
    labels = labels.tolist()

    fig = plt.figure()
    for j in range(len(set(labels))):  # use set() to remove duplicates of list
        color_map = {0: 'r',
                     1: 'k',
                     2: 'y',
                     3: 'b',
                     4: 'g',
                     5: 'c',
                     6: 'm'
                     }
        lis = []
        for i in range(len(labels)):
            # print("i", labels[i])
            if labels[i] == j:
                lis.append(df1.columns[i])
            else:
                # print("false")
                pass

        ax = fig.add_subplot(2, 3, j+1)
        ax.plot(df1[lis], c=color_map[j])  # , c=color_map[j]

        ax.set_title("Cluster"+str(j))
        print("cluster"+str(j), lis)

    plt.tight_layout()

    plt.savefig("./Figs/cluster_visualisation_subplot/" + str(choice) + "/n_cluster" + str(n_clusters) + ".png", dpi = 150)
    plt.show()

"""
Function : cluster_visualisation_single_plot
Definition: this function takes processed df and labels as arguments, and does the following:
            1. plots the labeled clusters in a overlay plot.
           
Return: 

"""


def cluster_visualisation_single_plot(dict, df1,choice,n_clusters):

    df1["time"] = pd.to_datetime(df1["time"], format='%d%b%Y:%H:%M:%S.%f')
    df1 = df1.set_index("time")
    print(df1.head())
    linewidth = 0.9
    alpha = 0.3

    print(dict)
    for i in dict.keys():
        print("keys",i)
        print(type(i))
        color_map = {0: 'k',
                     1: 'r',
                     2: 'g',
                     3: 'y',
                     4: 'b',
                     5: 'c',
                     6: 'm'
                     }

        plt.plot(df1[dict[str(i)]], c=color_map[int(i)], linewidth=linewidth, alpha=alpha)
        patchList = []

        for key in dict.keys():  # patch for holding the plots
            data_key = mpatches.Patch(color=color_map[int(key)], label="Cluster"+str(key))
            patchList.append(data_key)

        plt.legend(handles=patchList)

        linewidth = linewidth - 0.2
        # alpha = alpha-0.3
        # plt.legend()
    plt.savefig("./Figs/cluster_visualisation_single_plot/" + str(choice) + "/n_cluster" + str(n_clusters) + ".png", dpi = 150)
    plt.show()



"""
Function : mean_cluster_visualisation
Definition: this function takes processed df and labels as arguments, and does the following:
            1. plots the mean lines of the clusters in a overlay plot.
Return: Dataframe of mean values of each cluster for the entire timestamp

"""


def mean_cluster_visualisation(dict, df1,choice,n_clusters):

    df1["time"] = pd.to_datetime(df1["time"], format='%d%b%Y:%H:%M:%S.%f')
    df1 = df1.set_index("time")
    NUM_COLORS= n_clusters
    df_mean = pd.DataFrame()
    fig, ax = plt.subplots()
    cm = plt.get_cmap('gist_rainbow')
    color = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    for i in dict.keys():
        #color = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
        color_map_mean = {0: 'm',
                     1: 'c',
                     2: 'g',
                     3: 'b',
                     4: 'y',
                     5: 'k',
                     6: 'r'
                     }

        patchList = []
        df= pd.DataFrame()
        df["mean"]= df1[dict[str(i)]].mean(axis=1)
        # df["mean_new"] = df["mean"].resample("D").sum()
        #print("mean df", df.head())
        df_mean["cluster"+str(i)]= df["mean"]

        #plt.plot(df["mean"], c= color_map_mean[(int(i))], linewidth=0.5, alpha= 0.5 )
        plt.plot(df["mean"], c= color[int(i)], linewidth=0.5, alpha= 0.5 )

        for key in dict.keys():
            #data_key = mpatches.Patch(color=color_map_mean[int(key)], label="Cluster"+str(key))
            data_key = mpatches.Patch(color=(color[int(key)]), label="Cluster"+str(key))

            patchList.append(data_key)
        plt.legend(handles=patchList)
        plt.title('Mean plot of Clusters using ' + str(choice) + ' for N_Cluster = ' + str(n_clusters))
        #plt.ylim(0, 2)
    plt.savefig("./Figs/mean_cluster_visualisation/" + str(choice) + "/n_cluster" + str(n_clusters) + ".png", dpi = 150)
    plt.show()
    df_mean.to_csv("dfmean.csv")

    return df_mean

"""
Function : resampled_cluster_visualisation
Definition: Function for plotting the resampled mean Daily, Weekly and Querterly distribution of clustered data.
Return: 

"""
def resampled_cluster_visualisation(choice,n_clusters):
    df_mean = pd.read_csv("dfmean.csv")
    #print(df_mean.head())
    df_mean["time"] = pd.to_datetime(df_mean["time"])
    #df_mean= df_mean.set_index("time")
    #df1 = df_mean.resample('B', on='time').mean()

    df1 = df_mean.resample('D', on='time').mean()
    print(df1)
    df1.plot()
    plt.legend()
    plt.title('Daily mean plot of Clusters using ' + str(choice)+' for N_Cluster = '+str(n_clusters))
    plt.savefig("./Figs/resampled_cluster_visualisation/"+ str(choice) + "/n_cluster" + str(n_clusters) + "_Daily_Mean.png",dpi = 150)

    df2 = df_mean.resample('W', on='time').mean()
    df2.plot()
    plt.legend()
    plt.title('Weekly mean plot of Clusters using ' + str(choice) + ' for N_Cluster = ' + str(n_clusters))
    plt.savefig("./Figs/resampled_cluster_visualisation/" + str(choice) + "/n_cluster" + str(n_clusters) + "_Weekly_Mean.png", dpi=150)

    df3 = df_mean.resample('Q', on='time').mean()
    df3.plot()
    plt.legend()
    plt.title('Querterly mean plot of Clusters using ' + str(choice) + ' for N_Cluster = ' + str(n_clusters))
    plt.savefig("./Figs/resampled_cluster_visualisation/" + str(choice) + "/n_cluster" + str(n_clusters) + "_Resampled.png", dpi=150)

    plt.show()





"""
Function : acorn_plot
Definition: Function for plotting the cluster-wise acorn distribution in a stacked bar plot.
Return: 

"""


def acorn_plot(df, n_cluster,choice):

    df = df.reset_index()

    df = df.drop("level_1", axis=1)
    print(df.info())

    acorn_list = {}

    for i in range(n_cluster):
        #print (i)

        df2 = df[df["label"].astype(int) == i]

        df1 = df2['Acorn group'].value_counts(dropna=False).to_dict()
        # Uncomment the following line if distribution for acorn category needs to be checked
        # df1 = df2['Acorn category'].value_counts(dropna=False).to_dict()

        acorn_list["Cluster"+str(i)] = df1

    df_acorn = pd.DataFrame(acorn_list).T
    df_acorn= df_acorn.fillna(0)

    df_acorn.plot(kind='bar', stacked=True ,fontsize=6)
    plt.ylabel('numbers')
    plt.title('Acorn Distribution among Clusters using ' + str(choice)+' for N_Cluster = '+str(n_cluster))

    # if number of clusters are more, then less grids are required! This is implemented as follows:
    plt.grid()
    if n_cluster > 8:
        # show 20 grids for the length of data set
        minor_ticks = np.arange(0, len(df["label"])*30/100, round((len(df["label"])*30/100)/20))
    else:
        # show 50 grids for the length of data set
        minor_ticks = np.arange(0, len(df["label"]) * 70 / 100,round((len(df["label"]) * 70 / 100) / 50))

    frame1 = plt.gca()
    # For showing the cluster wise distribution, a table is added with the plot
    frame1.axes.xaxis.set_ticklabels([])
    plt.yticks(minor_ticks)
    plt.table(cellText=df_acorn.T.values,
              rowLabels=df_acorn.T.index,
              colLabels=df_acorn.T.columns,
              cellLoc='center', rowLoc='center',
              loc='bottom')
    # subplot adjustments for different cluster numbers
    if n_cluster > 8:
        plt.subplots_adjust(left=0.2, bottom=0.6)
    else:
        plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.savefig("./Figs/acorn_plot/" + str(choice) + "/n_cluster" + str(n_cluster) + ".png", dpi=600)

    plt.show()


"""
Function : acorn_plot
Definition: Function for plotting the correlation of the Radiation pattern with the clustered dataset using 
            pearson correlation.
Return: 

"""


def correlation_plot(dict_correlation,n_cluster,choice):

    print("dict_correlation",dict_correlation)

    dict_mean = {}
    for i in dict_correlation.keys():
        #print(i)
        mean = sum(dict_correlation[i])/len(dict_correlation[i])
        dict_mean["Cluster "+str(i)+" mean"]= mean
        print(mean)

    print("lenght is",len(dict_correlation.values()))
    labels, data, = dict_correlation.keys(), dict_correlation.values()
    print("data is", data)

    plt.boxplot(data,showmeans=True)
    myxticks= []

    data_list= list(data)
    print(data_list)
    print(len(data_list))
    print(len(data_list[0]))

    for i in range(n_cluster):
        ll = "cluster "+ str(i) + " instances "+ str(len(list(data)[i]))
        print(ll)
        myxticks.append(ll)

    plt.xticks(range(1, len(labels) + 1), myxticks)
    plt.xticks(rotation=90)

    plt.ylabel('pearson correlatoin index')
    plt.savefig("./Figs/Pearson_Corr_with_Radiation_data/" + str(choice) + "/n_cluster" + str(n_cluster) + ".png", dpi=600)
    #plt.text(x,y ,dict_mean)
    plt.show()


"""
Function : Hourlyplot
Definition: Function for plotting the average daily distribution of the clustered data and
            average hourly distribution of the clustered data for winter and summer.
Return: 

"""


def Hourlyplot(choice,n_clusters):
    df= pd.read_csv("dfmean.csv")
    df["time"] = pd.to_datetime(df["time"])
    # df = df.resample('D', on='time').mean()
    # df["time"] = pd.to_datetime(df["time"])

    df1=df
    df1['day_of_week'] = df1['time'].dt.day_name()
    df1= df.groupby(['day_of_week']).mean()
    df1.plot()
    plt.title("Avg day-wise consumption "+str(choice)+" "+str(n_clusters))
    plt.savefig("./Figs/resampled_cluster_visualisation/" + str(choice) + "/n_cluster" + str(n_clusters) + "_avg day-wise consumption.png",
        dpi=150)
    plt.show()

    df = df.resample('H', on='time').mean()
    df= pd.DataFrame(df)
    # print(df)
    df.to_csv("df.csv", index=True)
    df2= pd.read_csv("df.csv",parse_dates=True)

    df2["month"]= df.index.month
    df2["time"] = pd.to_datetime(df2["time"])
    # df2["month"].isin([1,2,3,4,5]):
    # df2= pd.DataFrame()
    df3= df2.loc[df2['month'].isin(['1', '2','3','4','5','10','11','12'])]
    # print("df2",df2)
    df3['Hours'] = df3['time'].dt.time
    df3 = df3.groupby(['Hours']).mean()
    print(df3)
    df3.iloc[:,0:n_clusters].plot()
    plt.title("Avg hour-wise consumption from  to October to May")
    plt.savefig("./Figs/resampled_cluster_visualisation/" + str(choice) + "/n_cluster" + str(
        n_clusters) + "avg_hour-wise_consumption_Oct_to_May.png",
                dpi=150)
    plt.show()

    df4 = df2.loc[df2['month'].isin(['6','7','8','9'])]
    print(df4)
    df4['Hours'] = df4['time'].dt.time
    print(df4)
    df4 = df4.groupby(['Hours']).mean()
    df4.iloc[:, 0:n_clusters].plot()
    plt.title("Avg hour-wise consumption from June to September")

    plt.savefig("./Figs/resampled_cluster_visualisation/" + str(choice) + "/n_cluster" + str(
        n_clusters) + "avg_hour-wise_consumption_June_to_Sept.png",
                dpi=150)
    plt.show()


"""
Function : DayWiseHourlyplot
Definition: Function for plotting the average daily distribution of consumption in individual plots
            for Monday to Sunday.
Return: 

"""


def DayWiseHourlyplot(choice,n_clusters):
    df = pd.read_csv("dfmean.csv")
    df["time"] = pd.to_datetime(df["time"])
    df1 = df
    df = df1.resample('H', on='time').mean()
    df = pd.DataFrame(df)
    df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"])
    df['day_of_week'] = df['time'].dt.day_name()
    # df.to_csv("df.csv", index=True)

    # Now Plot for every individual day
    for i in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:

        df3 = df.loc[df['day_of_week'].isin([i])]
        df3['Hours'] = df3['time'].dt.time
        df4 = df3.groupby(['Hours']).mean()

        df4.iloc[:, 0:n_clusters].plot()
        plt.title("Hour-wise consumption for"+str(i))
        plt.savefig("./Figs/Hour-wise consumption for Everyday/" + str(choice) + "/n_cluster" + str(n_clusters) +
                    "Hour-wise consumption for "+str(i), dpi=150)
        plt.show()


"""
Function : DayWiseGrouppedPlot
Definition: Function for plotting the daily distribution of consumption of all the clusters in a single plot
            from Monday to Sunday.
Return: 

"""

def dayWiseGrouppedPlot(choice,n_clusters):
    df = pd.read_csv("dfmean.csv")
    df["time"] = pd.to_datetime(df["time"])
    df1 = df
    df = df1.resample('H', on='time').mean()
    df = pd.DataFrame(df)
    df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"])
    df['day_of_week'] = df['time'].dt.day_name()
    # df.to_csv("df.csv", index=True)

    # Now Plot for every individual day
    dayList = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    colors = {"Monday": 'r',
              "Tuesday": 'k',
              "Wednesday": 'y',
              "Thursday": 'b',
              "Friday": 'g',
              "Saturday": 'c',
              "Sunday": 'm'
              }
    linestyles = {'cluster0':  '-',
                  'cluster1': '--',
                  'cluster2':  '-.',
                  'cluster3': ':',
                  'cluster4': '-'

                  }
    c=0
    for i in dayList:

        df3 = df.loc[df['day_of_week'].isin([i])]
        df3['Hours'] = df3['time'].dt.time
        df4 = df3.groupby(['Hours']).mean()
        print(df4.head())
        # df4.iloc[:, 0:n_clusters].plot()
        print(df4.columns)
        plt.plot(df4.iloc[:, 0:n_clusters+1], c=colors[i])

        patchList = []

        for key in dayList:  # patch for holding the plots
            data_key = mpatches.Patch(label="Day :" + str(key), color= colors[key])
            patchList.append(data_key)

        plt.legend(handles=patchList)
        c=c+1
        # plt.show()
    plt.show()

#dayWiseGrouppedPlot("Autoencoder",5)

#resampled_cluster_visualisation("hierarchical",5)
#Hourlyplot("hierarchical",5)
#correlation_plot("a",4,"hierarchical")


"""
Function : acornWisePlot
Definition: Function for plotting the Acorn Group wise distribution of the original data in a Line plot
Return: 

"""


def acornWisePlot(df):
    '''
    df = pd.read_csv("London_full_data.csv", index_col=None)  # for London dataset

    df = df.drop("Unnamed: 0", axis=1)
    #print(df1.head())
    df= pd.DataFrame(df)
    '''

    df= df.set_index(df["time"])

    acorn_data = pd.read_csv("acorn_df.csv")  # for full dataset
    acorn_data = acorn_data.reset_index()
    acorn_data = acorn_data.rename(index=str, columns={"Unnamed: 0": "HouseholdID"})

    print(acorn_data.head())
    print(acorn_data.shape)

    df2 = acorn_data.groupby('Acorn group')['HouseholdID'].agg(lambda x: list(x))
    df2 = dict(df2)

    # Call function for Line Plot
    plotAcorn(df, df2,'Acorn Group')

    # Call this function for box plot
    AcornWiseBoxPlot(df,df2,"Acorn Group")


"""
Function : acornCategoryWisePlot
Definition: Function for plotting the Acorn Category wise distribution of the original data in a Line plot
Return: 

"""


def acornCategoryWisePlot(df):
    '''
    df = pd.read_csv("London_full_data.csv", index_col=None)  # for London dataset

    df = df.drop("Unnamed: 0", axis=1)
    # print(df1.head())
    df = pd.DataFrame(df)
    '''
    df = df.set_index(df["time"])

    acorn_data = pd.read_csv("acorn_df.csv")  # for full dataset
    acorn_data = acorn_data.reset_index()
    acorn_data = acorn_data.rename(index=str, columns={"Unnamed: 0": "HouseholdID"})

    # print(acorn_data.shape)
    # print(acorn_data["Acorn group"].to_dict())

    df4 = acorn_data.groupby('Acorn category')['HouseholdID'].agg(lambda x: list(x))
    df4 = dict(df4)
    # print("Length of acporn category"+ str(len(df4.keys())))

    # Call function for Line Plot
    plotAcorn(df,df4, "Acorn Category" )

    # Call this function for box plot
    AcornWiseBoxPlot(df,df4,"Acorn Category")


"""
Function : plotAcorn
Definition: Function for plotting the line plot for both Acorn group and Acorn Category wise distribution 
            of the original data .
Return: 

"""
def plotAcorn(df,df1, label ):
    df= pd.DataFrame(df)
    count = 0

    NUM_COLORS = len(df1.keys())
    cm = plt.get_cmap('gist_rainbow')
    fig, ax = plt.subplots()
    #ax.set_prop_cycle(color= [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    color= [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    patchList = []

    for i in df1.keys():
        print("Meter IDs for "+ str(label)+ " "+str(i)+" are: ",df1[i])
        df3 = df[df1[i]]
        #print(df3.head(10))
        df3.plot(ax=ax, label=i, c= color[count])
        # plt.plot(df3, c=colors[count])
        data_key = mpatches.Patch(label=str(label) +" "+ str(i) + " , Total instances: " + str(len(df3.columns)), color=(color[count]))
        patchList.append(data_key)
        plt.legend(handles=patchList)
        plt.xticks(rotation=90)

        # df3.plot(c=colors[count])
        count = count + 1
    plt.savefig("./Figs/"+ str(label)+" Wise Plot", dpi= 150)
    plt.show()


"""
Function : AcornWiseBoxPlot
Definition: Function for plotting the Box plot for both Acorn group and Acorn Category wise distribution 
            of the original data . This plot shows the mean and median distribution of the data.
Return: 

"""


def AcornWiseBoxPlot(df,df1, label ):

    df3 = pd.DataFrame()
    for i in df1.keys():
        print(df1[i])
        df2 = df[df1[i]]
        print(df2.head(10))
        df3[str(i)] = df2.mean(axis=1)
    df3.boxplot()
    plt.title(str(label)+" Wise Box Plot")
    plt.xticks(rotation=90)

    plt.savefig("./Figs/"+ str(label)+" Wise Box Plot", dpi= 150)

    plt.show()


"""
Function : meterIDBoxPlot
Definition: Function for plotting the Box plot for all the meter ids of original data.
            This plot shows the mean and median distribution of the data.
Return: 

"""


def meterIDBoxPlot():
    df = pd.read_csv("London_full_data.csv", index_col=None)  # for London dataset

    df = df.drop("Unnamed: 0", axis=1)
    # print(df1.head())
    df = pd.DataFrame(df)
    df = df.set_index(df["time"])
    for column in df.loc[2:]:
        plt.figure()
        df.boxplot(df[column])
        plt.show()


# meterIDBoxPlot()
# acornCategoryWisePlot()
# acornWisePlot()