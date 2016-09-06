# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:34:24 2016
python project source code

@author: yizhe ge, yg2kj
         fandi lin, fl3bf
         greg gardner, gmg5dc
         tianye song, ts7fx
"""

# import statements:
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.api as sm



class insurance:
    def __init__(self, filename):
        self.data = self.readData(filename) 
                              
        
    def readData(self, filename):
        #read-in data from an external csv file
        temp = pd.read_csv(filename, 
                   names=["symboling", "normalized-losses", 
                          "make", "fuel-type", "aspiration", 
                          "num-of-doors", "body-style", "drive-wheels", 
                          "engine-location", "wheel-base", "length", 
                          "width", "height", "curb-weight", "engine-type", 
                          "num-of-cylinders", "engine-size", "fuel-system", 
                          "bore", "stroke", "compression-ratio", "horsepower", 
                          "peak-rpm", "city-mpg", "highway-mpg", "price"])
        return temp
    
        
    def cleanData(self):
        # take out all missing values of data
        self.data = auto[(auto["normalized-losses"]!="?") & (auto["num-of-doors"]!="?") & (auto["bore"]!="?") &
                  (auto["stroke"]!="?") & (auto["horsepower"]!="?") & (auto["peak-rpm"]!="?") & 
                  (auto["price"]!="?")]
                  
    def export(self, fileName):
        # output data to fileName
        self.data.to_csv(fileName, index=False)
    
    
    def checkEff(self, cat, eff): 
        # gives top three most eff = efficient/inefficient cars for 
        # cat = city/highway
        temp = self.data[['make', 'city-mpg', 'highway-mpg']]
        if (cat == 'city'):
            if (eff == 'efficient'):
                temp = temp.sort_values(by = ['city-mpg'], ascending = [False])
                return temp[['make','city-mpg']].head(3)
            elif (eff == 'inefficient'):
                temp = temp.sort_values(by = ['city-mpg'], ascending = [True])
                return temp[['make','city-mpg']].head(3)
            else:
                print('invalid input!')
                return None
        elif (cat == 'highway'):
            if (eff == 'efficient'):
                temp = temp.sort_values(by = ['highway-mpg'], ascending = [False])
                return temp[['make','highway-mpg']].head(3)
            elif (eff == 'inefficient'):
                temp = temp.sort_values(by = ['highway-mpg'], ascending = [True])
                return temp[['make','highway-mpg']].head(3)
            else:
                print('invalid input!')
                return None
        else:
            print('invalid input!')
            return None
            
    def compMpg(self):
        cyl = auto[['num-of-cylinders','city-mpg','highway-mpg']]

        # frequencies of number of cylinders
        cyl.groupby('num-of-cylinders').count() # Most cylinder sizes are either four or six. Only one is twelve. We will exclude twelve.
        
        # because we don't have enough data to account for individual categories,
        # we chose to group data based on the number of cylinders into two major groups
        # function to group number of cylinders
        def numgroup(word):
            if word in ['two','three','four']:
                return('twothreefour')
            if word in ['five','six','eight']:
                return('fivesixeight')
            else:
                return('twelve')
        
        # apply function and create new column
        cyl['number'] = cyl['num-of-cylinders'].apply(numgroup)
        
        # average mpg for two, three, or four cylinders
        fewercyl = cyl[cyl['number'] == 'twothreefour']
        fewercyl.mean()

    
    def avgMpg(self):
        # 3. avg mpg for different drive wheels (with same cylinder count)
        # First group all automobiles by type of drive wheels
        # then further group automobiles in each drive-wheel group
        # by their number of cylinders
        dwGroupped = self.data.groupby(["drive-wheels","num-of-cylinders"])
        # apply mean function to each subgroup (num-of-cylinders) within each
        # parent group (drive-wheels)
        dwAnalysis = dwGroupped.mean()[["city-mpg","highway-mpg"]]
        # print result
        return dwAnalysis
        
    def compHp(self):
        # 4. compare avg hp for fwd, 4wd and rwd
        # process the dataset, get rid of rows with missing hp values
        hpCleaned = self.data[(self.data['horsepower']!='?')]
        # in order to apply mean function to horsepowers, have to cast
        # horsepowers from string to integers.
        hpCleaned['horsepower'] = hpCleaned['horsepower'].astype('int')
        # group obsvations based on drive-wheel types
        hpGroupped = hpCleaned.groupby("drive-wheels")
        # apply mean function to all horsepowers within each categorical 
        # variable 'drive-wheel'
        hpAnalysis = hpGroupped.mean()[["horsepower"]]
        # print result
        return hpAnalysis
    def advance1(self, display):
        # data frame for symboling
        sym = self.data[['symboling','fuel-type','aspiration',
            'num-of-doors','body-style','drive-wheels',
            'engine-location','wheel-base','length',
            'width','height','curb-weight',
            'engine-type','num-of-cylinders','engine-size',
            'fuel-system','bore','stroke',
            'compression-ratio','horsepower','peak-rpm',
            'price']][
        	(self.data['num-of-doors'] != '?') & 
        	(self.data['price'] != '?') & 
        	(self.data['peak-rpm'] != '?') & 
        	(self.data['stroke'] != '?')]
        	# convert to floats
        sym.bore = sym.bore.astype(float)
        sym.stroke = sym.stroke.astype(float)
        sym.horsepower = sym.horsepower.astype(float)
        sym['peak-rpm'] = sym['peak-rpm'].astype(float)
        sym.price = sym.price.astype(float)
        
        # create dummy variables from categorical variables
        sym = pd.get_dummies(sym)
        
        # randomize data frame
        sym = sym.sample(frac=1).reset_index(drop=True)
        
        # decision tree classifier
        clf = tree.DecisionTreeClassifier(criterion = 'gini')
        
        # recursive feature elimination with cross validation
        rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(sym[:180]['symboling'],2), scoring='accuracy') # We use stratified k-fold because some classes are underrepresented.
        rfecv.fit(sym[:180].drop('symboling',axis = 1),sym[:180]['symboling'])
        
        # selected features
        sym.drop('symboling',axis = 1).columns[rfecv.support_]
        # The data set is small, resulting in highly variable numbers of optimal features, so the above procedure was repeated many times. The most frequent optimal number of features was 3. The most frequently occuring features were "wheel-base", "width", and "price", with "wheel-base" almost always a selected feature.
        
        # plot CV scores
        if display == True:
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (accuracy)")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.show()
        
        # data frame with selected features
        sym_select = sym[['symboling','wheel-base','width','price']]
        sym_select = sym_select.sample(frac=1).reset_index(drop=True)
        sym_select.to_csv('symbolingfeatures.csv',index = False)
        
        # training decision tree classifier
        clf.fit(sym_select[:180].drop('symboling',axis = 1),sym_select[:180]['symboling'])
        
        # scoring test set
        clf.score(sym_select[180:].drop('symboling',axis = 1),sym_select[180:]['symboling'])
        # We shuffle the data and obtain the score each time. With 180 data points, this classifier averages an accuracy of about 80%.
        
        # confusion matrix: rows are true classes and columns are predicted classes
        sym_select = sym_select.sample(frac=1).reset_index(drop=True)
        confusion_matrix(sym_select[180:]['symboling'],clf.predict(sym_select[180:].drop('symboling',axis = 1)),labels = [-3,-2,-1,0,1,2,3])
        
    def uiadvance1(self):
        clf = tree.DecisionTreeClassifier(criterion = 'gini')
        sym_select = pd.read_csv('symbolingfeatures.csv')
        clf.fit(sym_select.drop('symboling',axis = 1),sym_select['symboling'])
        cars = ['Hyundai Sonata','Ford Focus','Chevrolet Cruze','Hyundai Elantra','Ford Fusion','Honda Civic','Nissan Altima','Honda Accord','Toyota Corolla','Toyota Camry']
        wheelbases = [110.4,104.3,105.7,106.3,112.2,106.3,109.3,109.3,106.3,109.3]
        widths = [73.4,71.8,70.70,70.9,72.9,70.8,72,72.8,69.9,71.7]
        prices = [22135,18100,16995,17985,23485,19475,23335,23190,18135,23905]
        
        db = pd.DataFrame(cars,columns = ['car'])
        db['wheelbase'] = wheelbases
        db['width'] = widths
        db['price'] = prices
        
        print('Cars in database: \n')
        for car in cars:
            print(car)
        car = input('Select a car, or enter anything else to input specifications (required: wheelbase, width, and price):\n')
        
        if car.title() in cars:
            print('The predicted risk factor symbol is: ' + str(clf.predict(db[db['car'] == car.title()].drop('car',axis = 1))[0]))
        else:
            print('\nEnter specifications.')
            wheelbase = input('wheelbase (inches): ')
            width = input('width (inches): ')
            price = input('price (inches): ')
            print('\nThe predicted risk factor symbol is: ' + str(clf.predict([[wheelbase,width,price]])[0]))


    def advance2(self):
        # data frame for mpg
        mpg = self.data.drop('make',axis = 1)[self.data.bore != '?'][self.data.horsepower != '?']
        # convert to floats
        mpg.bore = mpg.bore.astype(float)
        mpg.stroke = mpg.stroke.astype(float)
        mpg.horsepower = mpg.horsepower.astype(float)
        mpg['peak-rpm'] = mpg['peak-rpm'].astype(float)
        
        # response variable and explanatory variables
        y = mpg['highway-mpg']
        X = mpg[['wheel-base','length','width',
                 'height','curb-weight','engine-size',
                 'bore','stroke','compression-ratio',
                 'horsepower','peak-rpm']]
        
        # add constant for intercept
        X = sm.add_constant(X)
        # fit model
        lm1 = sm.OLS(y,X)
        results1 = lm1.fit()
        # summary
        results1.summary()
        
        # Remove least significant variable and compare AIC. Repeat process until minimum AIC or all variables are signnificant.
        X = X[['const','wheel-base','length',
               'width','curb-weight','engine-size',
               'bore','stroke','compression-ratio',
               'horsepower','peak-rpm']]
        lm2 = sm.OLS(y,X)
        results2 = lm2.fit()
        results2.summary()
        
        X = X[['const','wheel-base','length',
               'width','curb-weight','engine-size',
               'bore','compression-ratio',
               'horsepower','peak-rpm']]
        lm3 = sm.OLS(y,X)
        results3 = lm3.fit()
        results3.summary()
        
        X = X[['const','wheel-base','length',
               'curb-weight','engine-size','bore',
               'compression-ratio','horsepower','peak-rpm']]
        lm4 = sm.OLS(y,X)
        results4 = lm4.fit()
        results4.summary()
        
        X = X[['const','length','curb-weight',
               'engine-size','bore','compression-ratio',
               'horsepower','peak-rpm']]
        lm5 = sm.OLS(y,X)
        results5 = lm5.fit()
        results5.summary()
        
        X = X[['const','length','curb-weight',
               'engine-size','compression-ratio','horsepower',
               'peak-rpm']]
        lm6 = sm.OLS(y,X)
        results6 = lm6.fit()
        results6.summary()
        
        X = X[['const','length','curb-weight',
               'engine-size','compression-ratio','horsepower']]
        lm7 = sm.OLS(y,X)
        results7 = lm7.fit()
        results7.summary()
        # Remaining variables are all at least somewhat significatn (p < 0.5). 'length' has p = 0.40, so we will remove it and check how AIC changes.
        
        # X = X[['const','curb-weight','engine-size',
        #        'compression-ratio','horsepower']]
        # lm8 = sm.OLS(y,X)
        # results8 = lm8.fit()
        # results8.summary()
        # AIC increased
        # The preferred model is 'lm7'. Highway mpg is best predicted by length, curb weight, engine size, compression ratio, and horsepower.
        
        
        # Now we fit a linear regression for subsets of the mpg data: front wheel drive (fwd) and rear wheel drive (rwd). Four wheel drive exists in the data, but the sample size is too small.
        
        # data frame
        mpg_fwd = auto.drop('make',axis = 1)[auto.bore != '?'][auto.horsepower != '?'][auto['drive-wheels'] == 'fwd']
        mpg_fwd.bore = mpg.bore.astype(float)
        mpg_fwd.stroke = mpg.stroke.astype(float)
        mpg_fwd.horsepower = mpg.horsepower.astype(float)
        mpg_fwd['peak-rpm'] = mpg['peak-rpm'].astype(float)
        
        # response variable and explanatory variables
        y = mpg_fwd['highway-mpg']
        X = mpg_fwd[['wheel-base','length','width',
                 'height','curb-weight','engine-size',
                 'bore','stroke','compression-ratio',
                 'horsepower','peak-rpm']]
        
        # add constant for intercept
        X = sm.add_constant(X)
        # fit model
        lm1 = sm.OLS(y,X)
        results1 = lm1.fit()
        # summary
        results1.summary()
        
        # Remove least significant variable and compare AIC. Repeat process until minimum AIC or all variables are signnificant.
        X = X[['const','wheel-base','length',
               'height','curb-weight','engine-size',
               'bore','stroke','compression-ratio',
               'horsepower','peak-rpm']]
        lm2 = sm.OLS(y,X)
        results2 = lm2.fit()
        results2.summary()
        
        X = X[['const','wheel-base','length',
               'height','curb-weight','engine-size',
               'bore','compression-ratio','horsepower',
               'peak-rpm']]
        lm3 = sm.OLS(y,X)
        results3 = lm3.fit()
        results3.summary()
        
        X = X[['const','wheel-base','length',
               'curb-weight','engine-size','bore',
               'compression-ratio','horsepower','peak-rpm']]
        lm4 = sm.OLS(y,X)
        results4 = lm4.fit()
        results4.summary()
        
        X = X[['const','wheel-base','length',
               'curb-weight','bore','compression-ratio',
               'horsepower','peak-rpm']]
        lm5 = sm.OLS(y,X)
        results5 = lm5.fit()
        results5.summary()
        
        X = X[['const','wheel-base','curb-weight',
               'bore','compression-ratio','horsepower',
               'peak-rpm']]
        lm6 = sm.OLS(y,X)
        results6 = lm6.fit()
        results6.summary()
        
        X = X[['const','wheel-base','curb-weight',
               'compression-ratio','horsepower','peak-rpm']]
        lm7 = sm.OLS(y,X)
        results7 = lm7.fit()
        results7.summary()
        # Remaining variables are all at least somewhat significant (p < 0.05). 'horsepower' has p = 0.040, so we will remove it and check how AIC changes.
        
        # X = X[['const','wheel-base','curb-weight',
        #        'compression-ratio,'peak-rpm']]
        # lm8 = sm.OLS(y,X)
        # results8 = lm8.fit()
        # results8.summary()
        # AIC increased
        # The preferred model is 'lm7'. Highway mpg for front wheel drives are best predicted by wheel base, curb weight, compression ratio, horsepower, and peak rpm. Significant differences from the general model are that length has been replaced with wheel base and engine size has been replaced with peak rpm.
        
        
        # data frame
        mpg_rwd = auto.drop('make',axis = 1)[auto.bore != '?'][auto.horsepower != '?'][auto['drive-wheels'] == 'rwd']
        mpg_rwd.bore = mpg.bore.astype(float)
        mpg_rwd.stroke = mpg.stroke.astype(float)
        mpg_rwd.horsepower = mpg.horsepower.astype(float)
        mpg_rwd['peak-rpm'] = mpg['peak-rpm'].astype(float)
        
        # response variable and explanatory variables
        y = mpg_rwd['highway-mpg']
        X = mpg_rwd[['wheel-base','length','width',
                     'height','curb-weight','engine-size',
                     'bore','stroke','compression-ratio',
                     'horsepower','peak-rpm']]
        
        # add constant for intercept
        X = sm.add_constant(X)
        # fit model
        lm1 = sm.OLS(y,X)
        results1 = lm1.fit()
        # summary
        results1.summary()
        
        # Remove least significant variable and compare AIC. Repeat process until minimum AIC or all variables are signnificant.
        X = X[['const','wheel-base','length',
               'height','curb-weight','engine-size',
               'bore','stroke','compression-ratio',
               'horsepower','peak-rpm']]
        lm2 = sm.OLS(y,X)
        results2 = lm2.fit()
        results2.summary()
        
        X = X[['const','wheel-base','length',
               'height','curb-weight','engine-size',
               'bore','stroke','compression-ratio',
               'peak-rpm']]
        lm3 = sm.OLS(y,X)
        results3 = lm3.fit()
        results3.summary()
        
        X = X[['const','wheel-base','length',
               'curb-weight','engine-size','bore',
               'stroke','compression-ratio','peak-rpm']]
        lm4 = sm.OLS(y,X)
        results4 = lm4.fit()
        results4.summary()
        
        X = X[['const','length','curb-weight',
               'engine-size','bore','stroke',
               'compression-ratio','peak-rpm']]
        lm5 = sm.OLS(y,X)
        results5 = lm5.fit()
        results5.summary()
        
        X = X[['const','curb-weight','engine-size',
               'bore','stroke','compression-ratio',
               'peak-rpm']]
        lm6 = sm.OLS(y,X)
        results6 = lm6.fit()
        results6.summary()
def main():
    status = True
    while status == True:
        print("\nWelcome to the automobile characteristics inspection program...\n")
        proj = insurance("imports-85.data.txt")
    
    
        print("What character would you like to inspect? Enter 1 or 2")
        print("1. Efficiency rank")
        print("2. ui cv")
        print("3. Exit")
    
        option = str(input("Your choice is: "))
        while option not in ("1", "2"):
            option = str(input("Invalid input! Enter again: "))
        
        if option == "1":
            cat = str(input("Enter a mpg category (city or highway): "))
            while cat not in ("city", "highway"):
                cat = str(input("Invalid input! Enter again: "))
            eff = str(input("Would you like to see the most efficient or inefficient cars? (efficient or inefficient): "))
            while eff not in ("efficient", "inefficient"):
                eff = str(input("Invalid input! Enter again: "))
            print(proj.checkEff(cat, eff))
            
            option1 = str(input("Continue? (y or n): "))
            while option1 not in ("y", "n"):
                option1 = str(input("Invalid input! Enter again: "))
        elif option == "2":
            proj.advance1(False)
            proj.uiadvance1()
            break
            
        elif option == "3":
            break
        
        if option1 == "y":
            continue
        else:
            break
    

if __name__ == "__main__": main()

