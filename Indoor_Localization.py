data = 'D:/Downloads/1478167720_9233432_trainingData.csv'
test = 'D:/Downloads/1478167721_0345678_validationData.csv'
dataset = pd.read_csv(data,index_col=None)
testset = pd.read_csv(test,index_col=None)


from keras.utils import np_utils

X = dataset.iloc[:, :-9].values

y_floor = dataset.iloc[:, 522:-6].values
y_building = dataset.iloc[:, 523:-5].values
y_latitude = dataset.iloc[:, 521:-7].values
y_longitude = dataset.iloc[:, 520:-8].values
X_test = testset.iloc[:, :-9].values
y_test = testset.iloc[:, 520:-5].values



from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

lab_enc = preprocessing.LabelEncoder()
y_building_enc = lab_enc.fit_transform(y_building)
y_floor_enc = lab_enc.fit_transform(y_floor)



from sklearn.neighbors import KNeighborsClassifier

building_classifier = KNeighborsClassifier(n_neighbors=3)
floor_classifier = KNeighborsClassifier(n_neighbors=3)

building_classifier.fit(X, y_building)
floor_classifier.fit(X, y_floor)


from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
latitude_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
longitude_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
latitude_regressor.fit(X, y_latitude) 
longitude_regressor.fit(X, y_longitude)


latitude = latitude_regressor.predict(X_test)
longitude = longitude_regressor.predict(X_test)

floor = floor_classifier.predict(X_test)
building = building_classifier.predict(X_test)



from sklearn.metrics import mean_absolute_error

floor_mae = mean_absolute_error(y_test[:, 2], floor)
building_mae = mean_absolute_error(y_test[:, 3], building)

print("MAE of floor : " + str(floor_mae))
print("MAE of building : "+ str(building_mae))



from sklearn import metrics
print("Floor Acccuracy : "+str(metrics.accuracy_score(y_test[:, 2], floor)))
print("Building Acccuracy : "+str(metrics.accuracy_score(y_test[:, 3], building)))


print(latitude)
np.savetxt('predictions.csv', np.c_[longitude, latitude, floor, building], delimiter=',')


error = []

for j in range(1, 15):
    loss = 0
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(X, y_floor)
    pred_i = knn.predict(X_test)
    for i in range (0, len(floor)-1):
        if pred_i[i]!=y_test[:,2][i]:
            loss = loss + 1
    print("K ="+str(j)+" : "+str(loss/len(floor)))
    error.append(loss/len(floor))
        

print("K ="+str(j)+" : "+str(loss/len(floor)))



print("Latitude error: "+ str(mean_absolute_error(y_test[:, 1],  latitude)))
print("Longitude error: "+ str(mean_absolute_error(y_test[:, 0], longitude)))


plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')



error_lat = []
error_lon = []

for i in range (1, 8):
    test_lat_regression = RandomForestRegressor(n_estimators = i*20, random_state = 0) 
    test_lon_regression = RandomForestRegressor(n_estimators = i*20, random_state = 0) 
    test_lat_regression.fit(X, y_latitude) 
    test_lon_regression.fit(X, y_longitude) 
    latitude = test_lat_regression.predict(X_test)
    longitude = test_lon_regression.predict(X_test)
    error_lat.append(mean_absolute_error(y_test[:, 1],  latitude))
    error_lon.append(mean_absolute_error(y_test[:, 0],  longitude))
    print("Latitude error: "+ str(mean_absolute_error(y_test[:, 1],  latitude)))
    print("Longitude error: "+ str(mean_absolute_error(y_test[:, 0], longitude)))



plt.figure(figsize=(12, 6))
plt.plot(range(1, 8), error_lon, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate N Value for Longitude')
plt.xlabel('N Value')
plt.ylabel('Mean Error')

