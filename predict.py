import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas
from sklearn import tree

diff = 1
month=4
dday=26

#Data kaynaktan cekilir
dataframe = pandas.read_csv('datagun.csv')
arrayData = dataframe.values

size = arrayData.shape[0]

#ana data secilir
edd = arrayData[0:size - diff - 1, 1:8]
edd = edd.astype('Float32')

#Hedef data secilir.
edtc = arrayData[1:size - diff, 4]
edtc = edtc.astype('Float64')

#tahminleme algoritmasina gonderilir.
prdc = tree.DecisionTreeRegressor(max_depth=7)
prdc = prdc.fit(edd, edtc)


#tahminleme yaptirilir.
rsltc = prdc.predict(arrayData[size - 60:, 1:8])

#R^2 skoru belirlenir.
print(r2_score(arrayData[size-60+1:,4],rsltc[:rsltc.shape[0]-1]))

sizer = rsltc.shape[0]
print('Close:',rsltc[sizer - 1])


plt.figure(figsize=(14, 6))
plt.subplots_adjust(bottom=0.31)
plt.plot(arrayData[size-60+1:,4], c="k", label="Day's Actual")
plt.plot(rsltc, c="b", label="Predict", linewidth=2)
plt.xlabel("Days")
plt.ylabel("BTC Price - Close")
plt.title("Machine Learn BTC Price Prediction ~ CyberSensei")
plt.grid(True)
plt.xticks(numpy.arange(60),pandas.date_range(pandas.datetime(2018, month, dday), periods=60),rotation='vertical')
#plt.xticks(numpy.arange(day),days,rotation='vertical')
plt.legend()
plt.show()

