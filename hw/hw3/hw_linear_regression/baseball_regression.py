
#for regression stuff
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import time

#read in the training data
f = open('training_data.csv', 'r')

#lists for data
rs_list = []
obp_list = []
slg_list = []
ba_list = []

counter = -1

for line in f.readlines():

    #skip the first line
    counter += 1
    if counter == 0:
        continue

    #split up the line
    line_list = line.split(',')

    rs = int(line_list[3])
    obp = float(line_list[6])
    slg = float(line_list[7])
    ba = float(line_list[8])

    #add stuff to the training data lists
    rs_list.append(rs)
    obp_list.append(obp)
    slg_list.append(slg)
    ba_list.append(ba)

#close the file since we've read it
f.close()

#reverse the order of our lists so they count the earliest data first
rs_list.reverse()
obp_list.reverse()
slg_list.reverse()
ba_list.reverse()

#need to convert each independent variable into a Numpy array,
#then reshape the data so each value is its own array (for some
#reason--it's needed for the ML portion of scikit-learn 
obp_array = np.array(obp_list).reshape(-1, 1)
slg_array = np.array(slg_list).reshape(-1, 1)
ba_array = np.array(ba_list).reshape(-1, 1)

#dependent variable is just an array
rs_array = np.array(rs_list)

#now that we have our arrays, can make our linear regression models
#for each situation

obp_reg = linear_model.LinearRegression()
slg_reg = linear_model.LinearRegression()
ba_reg = linear_model.LinearRegression()

#fit the linear model using each independent variable and rs_list
#as our dependent variable
obp_reg.fit(obp_array, rs_array)
slg_reg.fit(slg_array, rs_array)
ba_reg.fit(ba_array, rs_array)

#evaluate how good of a fit we've gotten
r_squared_obp = obp_reg.score(obp_array, rs_array)
r_squared_slg = slg_reg.score(slg_array, rs_array)
r_squared_ba = ba_reg.score(ba_array, rs_array)

print("\nCoefficients of determination for obp, slg, ba:")
print(r_squared_obp)
print(r_squared_slg)
print(r_squared_ba)

#find weights and biases for the models
print("\nWeights:")
print(obp_reg.coef_)
print(slg_reg.coef_)
print(ba_reg.coef_)

print("\nIntercepts:")
print(obp_reg.intercept_)
print(slg_reg.intercept_)
print(ba_reg.intercept_)

#first, create arrays 
#plot all our relationships, plus the lines of best fit

#----------------------------------------------#

ba_x = np.arange(min(ba_list), max(ba_list), (max(ba_list) - min(ba_list))/500)
ba_y = ba_x * ba_reg.coef_ + ba_reg.intercept_ 

plt.figure(1)
plt.scatter(ba_list, rs_list, s=2, color='black')
plt.plot(ba_x, ba_y, color='blue', linewidth=3,
         label='R_squared: ' + str(round(r_squared_ba, 2)))
plt.title('Batting Average vs. Runs Scored')
plt.xlabel("Batting average of team")
plt.ylabel("Runs scored by team")
plt.legend()

#-----------------------------------------------#

obp_x = np.arange(min(obp_list), max(obp_list), (max(obp_list) - min(obp_list))/500)
obp_y = obp_x * obp_reg.coef_ + obp_reg.intercept_

plt.figure(2)
plt.scatter(obp_list, rs_list, s=2, color='black')
plt.plot(obp_x, obp_y, color='magenta', linewidth=3, 
         label='R_squared: ' + str(round(r_squared_obp, 2)))
plt.title('On-Base Percentage vs. Runs Scored')
plt.xlabel("On-base percentage of team")
plt.ylabel("Runs scored by team")
plt.legend()

#------------------------------------------------#

slg_x = np.arange(min(slg_list), max(slg_list), (max(slg_list) - min(slg_list))/500)
slg_y = slg_x * slg_reg.coef_ + slg_reg.intercept_

plt.figure(3)
plt.scatter(slg_list, rs_list, s=2, color='black')
plt.plot(slg_x, slg_y, color='orange', linewidth=3,
         label='R_squared: ' + str(round(r_squared_slg, 2)))
plt.title('Slugging Percentage vs. Runs Scored')
plt.xlabel("Slugging percentage of team")
plt.ylabel("Runs scored by team")
plt.legend()

plt.show()





