import csv
import numpy as np
import math

from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt

def readCSV(filename):
    """
    Read CSV and return in float
    :param filename:
    :return:
    """
    data = list( csv.reader(open('HW_08_DBScan_Data_NOISY_v300.csv','r'),delimiter=','))
    for dIdx in range(len(data)):
        data[dIdx] = [float(data[dIdx][0]),float(data[dIdx][1]),float(data[dIdx][2])]
    #print(data[0])
    return data

# data = readCSV('HW_08_DBScan_Data_NOISY_v300.csv')
# print(len(data))

def getEps(data):
    """
    Plot points to observe inflection point
    :param data:
    :return:
    """
    X = pdist(data)

    Sq = squareform(X)

    FourthDist = []
    firstDist = []
    k = 10
    kNeighbors = []
    for idx in range(len(Sq)):
        Sq[idx] = np.sort(Sq[idx])

    for i in range(k):
        kNeighbors.append(Sq[:, i + 1])
    for i in range(k):
        kNeighbors[i] = np.sort(kNeighbors[i])

    for i in range(k):
        plt.plot(kNeighbors[i])

    plt.title('10 Nearest Point')
    plt.show()


class Cluster:
    def __init__(self,id):
        self.clusterId = id
        self.neighbors =[]
    def addToCluster(self,point):
        self.neighbors.append(point)
    def getID(self):
        return self.clusterId
    def getNeighbor(self):
        return self.neighbors

class Point:
    def __init__(self,id,coordinate):
        self.id = id
        self.coordinate = coordinate
        self.clusterId = -1

#regionQuery(P, eps)
#   return all points within P's eps-neighborhood (including P)
def addToTheCluster(pointIdx,AllPoints,Allneighbors,DistanceRanks,cluster,visited,eps,minPts):
    """
    Add data points to cluster based on density

    :param pointIdx:
    :param AllPoints:
    :param Allneighbors:
    :param DistanceRanks:
    :param cluster:
    :param visited:
    :param eps:
    :param minPts:
    :return:
    """
    point = AllPoints[pointIdx]
    cluster.addToCluster(point)
    point.clusterId = 1

    temp = Allneighbors[pointIdx]
    count = 0
    for points in temp:
        if visited[points] == 0:
            visited[points] =1
            neighbors = findNeighbors(AllPoints[points],DistanceRanks,eps)
            Allneighbors[points] = neighbors
            if len(neighbors) >= minPts:
                temp += neighbors

        if AllPoints[points].clusterId == -1:
            AllPoints[points].clusterId = 1
            cluster.addToCluster(AllPoints[points])
        count+=1


def findNeighbors(dataPoint,DistanceRanks,eps):
    """
    Find neighbors within epsilon value
    :param dataPoint:
    :param DistanceRanks:
    :param eps:
    :return:
    """
    neighbours = []
    #print(len(DistanceRanks[0]))
    #print(dataPoint.id)
    for idx in range(len(DistanceRanks)):
        temp = DistanceRanks[dataPoint.id][idx]
        if temp[1]< eps:
            neighbours.append(temp[0])
        else: return neighbours


def rankNeighbors(Data):
    """
    Rank the neighbors from each point
    :param Data:
    :return:
    """
    strokeDist = []
    for i in range(len(Data)):
        strokeDist.append([])
    index = 0
    for point1 in Data:
        dist = []
        index1=0
        for point2 in Data:
            #dist.append(math.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2))
            dist.append((index1,math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)))
            index1+=1
        #x = copy.deepcopy(dist)
        #print(x)
        dist.sort(key= lambda x:x[1])
        #print(x)
        # Get rank for each element
        idx1 =0
        for e in dist:
            #i = x.index(e)
            strokeDist[index].append(e)
            idx1 +=1
        index+=1
    return strokeDist

def DBScan(Data, eps, MinPts):
    """
    Cluster each point
    :param Data:
    :param eps:
    :param MinPts:
    :return:
    """
    c=0
    DistanceRanks = rankNeighbors(Data)
    clusterList =[]
    visited = np.zeros(len(Data))
    noise = []
    AllNeibours = {}
    AllPoints = []
    Clusters = []
    #Create Points and Initialize dictionary
    for pointIdx in range(len(Data)):
        AllPoints.append(Point(pointIdx,Data[pointIdx]))
        AllNeibours[pointIdx] = []
    #Iterate over all points
    for pointIdx in range(len(Data)):
        if visited[pointIdx]==1:
            continue
        visited[pointIdx] = 1
        #Get neighbors for point curret Point
        AllNeibours[pointIdx] =  findNeighbors(AllPoints[pointIdx],DistanceRanks,eps)

        #Check if point is a noise point else create a cluster and expand it using BFS
        if len(AllNeibours[pointIdx]) < MinPts:
            noise.append(pointIdx)
        else:
            Clusters.append(Cluster(len(Clusters)))

            addToTheCluster(pointIdx,AllPoints,AllNeibours,DistanceRanks,Clusters[-1],visited,eps,MinPts)

    plotPoints = []
    colors = ['b','r','g','y','k']
    count = 0

    #Plot points
    idx =0
    for c in Clusters:
        plotPoints = []
        for e in c.getNeighbor():
        #print(e.coordinate)
            plotPoints.append(e.coordinate[0:2])
        plotPoints = np.array(plotPoints)
        plt.scatter(plotPoints[:,0],plotPoints[:,1],c = colors[idx])
        idx +=1
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.title('CLUSTERS')
    plt.show()
    print('Number of Cluster',len(Clusters))
    tt = []
    for n in noise:
        tt.append( AllPoints[n].coordinate)

    tt = np.array(tt)
    print('Number of noise point',len(tt))
    plt.scatter(tt[:,0],tt[:,1])
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.title('NOISE POINTS')
    plt.show()

    return Clusters


def centerOfMass(data):
    """
    Calculate Center of mass for each cluster
    :param data:
    :return:
    """
    dd = []
    for d in data:
        dd.append(d.coordinate)

    data = dd
    data = np.array(data)
    n = len(data)
    x = sum(data[:,0])
    y = sum(data[:,1])
    z = sum(data[:,2])
    x/=n
    y/=n
    z/=n
    return x,y,z,n


if __name__ == '__main__':


    X= readCSV('HW_08_DBScan_Data_NOISY_v300.csv')

    #getEps(X)
    Clusters = DBScan(X,1.2,15)
    clusterDict = {}
    for c in Clusters:
        clusterDict[c.getID] = centerOfMass(c.getNeighbor())

    for v in clusterDict:
        print(clusterDict[v])
