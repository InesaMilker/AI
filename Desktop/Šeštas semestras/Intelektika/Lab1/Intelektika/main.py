import pandas
import math
import statistics
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sn


# column_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
data = pandas.read_csv('Churn_Modelling.csv', keep_default_na=False)

CreditScore = data.CreditScore.tolist()
Geography = data.Geography.tolist()
Gender = data.Gender.tolist()
Age = data.Age.tolist()
Tenure = data.Tenure.tolist()
Balance = data.Balance.tolist()
NumOfProducts = data.NumOfProducts.tolist()
HasCrCard = data.HasCrCard.tolist()
IsActiveMember = data.IsActiveMember.tolist()
EstimatedSalary = data.EstimatedSalary.tolist()
Exited = data.Exited.tolist()


def continious(col):
    # Apskaičiuojamas procentas trūkstamų reikšmių lentelėje
    print("All value count: " + str(len(col)))
    missing_data = 0
    for i in col:
        if i == '':
            missing_data += 1
    miss_data = missing_data * 100 / len(col)
    print("Missing data: " + str(miss_data))

    # Pašalinami tušti elementai
    while '' in col:
        col.remove('')
    # string listas paverčiamas į int listą
    for i in range(0, len(col)):
        col[i] = int(col[i])

    print("Cardinality: " + str(len(set(col))))
    print("Minimal value: " + str(min(col)))
    print("Maximal value: " + str(max(col)))
    quantile = np.array(col)
    print('1-st quartile: ' + str(np.quantile(col, .25)))
    print('3-rd quartile: ' + str(np.quantile(col, .75)))
    print('Average: ' + str(np.average(col)))
    print('Median: ' + str(statistics.median(col)))
    print('Standard deviation: ' + str(np.std(col)))

    #Atkomentuoti norit nubėžti grafiką pasirinktam atributui
    #Graph_Continious(col)

    #Į atributą įdedame kiekį trūkstamų reikšmių, kurios yra mediana
    while missing_data:
        col.append(round(statistics.median(col)))
        missing_data -= 1

    #Atsikomentuoti norint nubrėžti grafiką tarp atributų CreditScore, Tenure
    #Scatter(col, Tenure, 'Credit Score','Tenure')
    # Atsikomentuoti norint nubrėžti grafiką tarp atributų Tenure, Age
    #Scatter(col, Age,'Tenure', 'Age')
    # Atsikomentuoti norint nubrėžti grafiką tarp atributų CreditScore, Age
    #Scatter(col, Age, 'Credit Score', 'Age')
    # Atsikomentuoti norint nubrėžti grafiką tarp atributų Balance, Age
    #Scatter(col, Age, 'Balance', 'Age')

    #I6kvie2iamas metodas duomenų noramlizacijai
    #Range_Normalization(col)

    #Matrica
    print(Corr_Matrix(col, Age, Tenure, Balance, EstimatedSalary))


def Graph_Continious(col):
    # Paskaičiuojama bar skaičius
    bar_number = 1 + 3.22 * (math.log(len(col)))
    # Braižoma histograma
    plt.hist(col, edgecolor='purple', bins=round(bar_number))
    plt.xticks(rotation='vertical')
    plt.xlabel('Reikšmės')
    plt.ylabel('Pasikartojimų kiekis')
    plt.show()


def descrete(col):
    lenght = str(len(col))
    print("All value count: " + lenght)
    missing_data = 0
    for i in col:
        if i == '':
            missing_data += 1
    missing_data = missing_data * 100 / len(col)
    print("Missing data: " + str(missing_data))

    # Pašalinami tušti elementai
    while '' in col:
        col.remove('')


    print("Cardinality: " + str(len(set(col))))
    modes = Counter(col)
    freq1 = modes.most_common(2)[0][1]
    print('Most common mode: ' + str((modes.most_common(2)[0][0])))
    print('Mode frequency: ' + str((modes.most_common(2)[0][1])))
    print('Mode percentage: ' + str((freq1 / int(lenght)) * 100))
    freq2 = modes.most_common(2)[1][1]
    print('2-nd most common mode: ' + str((modes.most_common(2)[1][0])))
    print('2-nd mode frequency: ' + str((modes.most_common(2)[1][1])))
    print("2-nd mode percentage: " + str((freq2 / int(lenght)) * 100))
    #Atsikomentuoti norit pamatyti pasirinkto atributo grafiką
    #Graph_Descrete(col)

    #Atsikomentuoti norit pamatyti pasirinkto atributo Bar grafiką
    #Bar_Plot((modes.most_common(2)[0][1]), (modes.most_common(2)[1][1]), (modes.most_common(2)[0][0]), (modes.most_common(2)[1][0]))

    while missing_data:
        col.append((modes.most_common(2)[0][0]))
        missing_data -= 1

    Box_Plot(col, Exited)


def Graph_Descrete(col):
    plt.hist(col, edgecolor='pink', bins=10)
    plt.xlabel('Reikšmės')
    plt.ylabel('Pasikartojimų kiekis')
    plt.show()

#Scatter tipo diagrama,iškviečiama per continious metodą
def Scatter(col1, col2, xname, yname):
    plt.scatter(col1, col2)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

#SPLOM tipo diagramos brėžimas su visais tolydžiojo tipo atributais
def SPLOM(col1, col2,col3,col4,col5):
    datas = pandas.DataFrame({'Credit Score': col1, 'Age': col2, 'Tenure': col3, 'Balance': col4, 'EstimatedSalary': col5})
    pandas.plotting.scatter_matrix(datas, figsize=(15, 15))
    plt.show()

#print(SPLOM(CreditScore,Age,Tenure,Balance,EstimatedSalary))

#Iškviečiamas per metodą descrete
def Bar_Plot(col1,col2, name1, name2):
    df = pandas.DataFrame({name1: col1, name2: col2}, index=[1])
    df.plot(kind='bar')
    plt.show()

def Box_Plot():

    plt.boxplot(data['EstimatedSalary'])
    plt.show()

def Range_Normalization(col):
    min, max = 0, 1
    normal = [min + (max - min) * x for x in col]
    return normal

def Corr_Matrix(col1, col2,col3,col4,col5):
    datas = pandas.DataFrame({'Credit Score': col1, 'Age': col2, 'Tenure': col3, 'Balance': col4, 'EstimatedSalary': col5})
    corrMatrix = datas.corr()
    print('Kovariacija \n')
    print(corrMatrix)
    covMatrix = datas.cov()
    print('Koreliacija \n')
    print(covMatrix)
    corr = corrMatrix.corr()
    sn.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    plt.show()

print(Box_Plot())
#print(Bar_Plot())
#print(continious(CreditScore))
#print(continious(Balance))
#print(continious(EstimatedSalary))
'''
print(continious(Age))
print(continious(Tenure))
print(continious(Balance))
print(continious(EstimatedSalary))
'''
#print(descrete(Exited))
'''


print(descrete(Geography))
print(descrete(NumOfProducts))
print(descrete(HasCrCard))
print(descrete(IsActiveMember))
print(descrete(Exited))
'''
