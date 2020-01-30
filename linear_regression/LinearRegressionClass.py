'''
this file defines a class which calculates linear regression weight w,
and its R-Squared score for comparison
'''

import numpy  as np
import pandas as pd
import itertools
import math
from matplotlib           import pyplot as plt
from numpy                import matlib as mlib
from mpl_toolkits.mplot3d import axis3d

class LinearRegression():
    '''
    this class calculates each types (single variable, double, polynomial) of
    linear regression for comparing each method for explaining the data
    linear regression is basically calculates weights w
    and predicts Y by following equation
    Y = Xw
    '''

    def __init__( self, X, Y, namesX=[], nameY=[] ):
        '''
        initialize this class
        input:
            X : explanatory variables (should be 'number of records' x 'Demensionarity' )
                (caution: don't set bias term (ones(N)) to X)
            Y : objective varialbes (should be 'number of records' x 1)
        '''
        # check violation of data size
        if X.shape[1] != len( namesX ):
            print( '3rd argument, namesX, should be same size of columns of X' )
            return
        if X.shape[0] != len(Y):
            print( 'data record of X and Y should be same' )
            return
        if np.isscalar( nameY ) == False:    # this check should be modified cause str is not scalar
            print( '4th argument, nameY, should has one value' )
            return

        # set initial variables
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.origX = X
        self.origY = Y
        self.namesX = namesX
        self.nameY  = nameY

        # split the data to train and test data
        self.testN  = round( self.N / 5 )
        self.trainN = self.N - self.testN
        self.trainX = self.origX[ :self.trainN, : ]
        self.testX  = self.origX[ self.trainN:, : ]
        self.trainY = self.origY[ :self.trainN ]
        self.testY  = self.origY[ self.trainN: ]
    # define function
    def calcR2( self, Y, Yhat ):
        '''
        calculate R-Squared value for input
        input
            Y    : answer of Y (objective variable)
            Yhat : predicted value of Y
        output
            R2   : R-Squared value for Yhat
        '''
        d1 = Y - Yhat
        d2 = Y - Y.mean()
        R2 = 1 - d1.dot(d1) / d2.dot(d2)
        return R2

    def combinations_count( self, n, r ):
        '''
        calculate the combination count of nCr
        input
            n : number of candidates
            r : number of choice
        '''
        r = min( r, n - r )
        return math.factorial( n ) // ( math.factorial( n-r ) * math.factorial( r ) )
    
    def plotAll3DScatter( self ):
        '''
        display scatter plots for all cimbinations of 2 of X variables and Y
        for example, if X has 3 variables, A, B, C,
        this function shows 3D scatter plot 3 times,
        (A and B vs Y, A and C vs Y, and B and C vs Y)
        how many times 3D plot will be shown is nCr (n: num of dimensionarity, r is 2)
        '''
        if self.combinations_count( self.D, 2 ) > 3:
            nCr = self.combinations_count( self.D, 2 )
            print( 'the number of plot will be {0}, which means you have to close plot window {0} times'.format(nCr) )
            answer = []
            while answer == []:
                print( 'are you ok for {0} times showing? [Y/N]'.format( nCr ) )
                val = input()
                if val.lower()=='y' or val.lower()=='n':
                    answer = val
            if answer.lower() == 'n':
                return

        if self.D < 2:
            print( 'the assined data X is one dimensional' )
            print( 'try to use plotAll2DScatter instead' )
            return
        for i in range( self.D ):
            for j in range( self.D ):
                if i >= j:
                    continue
                fig = plt.figure()
                ax  = fig.add_subplot( 111, projection='3d' )
                ax.scatter( self.origX[:,i], self.origX[:,j], self.origY )
                ax.set_xlabel( self.namesX[i] )
                ax.set_ylabel( self.namesX[j] )
                ax.set_zlabel( self.nameY )
                plt.grid()
                plt.show()
    
    def plotAll2DScatter( self ):
        '''
        display scatter plots for each column of X and Y
        for example, if X has 3 variables, A, B, C,
        this function shows 2D scatter plot 3 times,
        (A vs Y, B vs Y, and C vs Y)
        '''
        if self.D > 3:
            print( 'the number of plot will be {0}, which means you have to close plot window {0} times'.format( self.D ) )
            answer = []
            while answer == []:
                print( 'are you ok for {0} times showing? [Y/N]'.format( self.D ) )
                val = input()
                if val.lower()=='y' or val.lower()=='n':
                    answer = val
            if answer.lower() == 'n':
                return
        
        for i in range( self.D ):
            plt.scatter( self.origX[:,i], self.origY )
            plt.xlabel( self.namesX[i] )
            plt.ylabel( self.nameY )
            plt.grid()
            plt.show()
    
    def calcSingleRegression( self ):
        '''
        calculate weight (including bias term) for each column of X to explain the objective value Y
        cause result including bias term, the size of w will be Dx2
        the w is calculated for trainX (check self.trainX) for explaining trainY
        output R2 is calculated for testX and testY
        output
            w  : list of weights which is sequence of row vectors
            R2 : R-Squared values for each w which evaluates each w in list above w
                 each R2 is how good the No.i row of w explanes testY
                 max is 1 (the best explanation)
        '''
        # assume trainX doesn't have bias term
        D = self.D
        trainX = self.trainX
        testX  = self.testX
        trainY = self.trainY
        testY  = self.testY
        w  = np.zeros( ( D, 2 ) )
        for i in range( D ):
            tmpX = np.vstack([ trainX[:,i], np.ones( self.trainN ) ]).T
            w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
        
        R2_single = np.zeros( D )
        for i in range( D ):
            tmpX = np.vstack([ testX[:,i], np.ones( self.testN ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_single[i] = self.calcR2( testY, Yhat )
        return w, R2_single
    
    def calcDoubleRegression( self ):
        '''
        calculate weight (including bias term) for each column of X to explain the objective value Y
        cause result including bias term, the size of w will be Dx3
        the w is calculated for trainX (check self.trainX) for explaining trainY
        output R2 is calculated for testX and testY
        output
            w  : list of weights which is sequence of row vectors
            R2 : R-Squared values for each w which evaluates each w in list above w
                 each R2 is how good the No.i row of w explanes testY
                 max is 1 (the best explanation)
            combi_pattern : combination of X columns for each row of w
                            w[:,i] is calclated from X columns of combi_pattern[i] and bias term
        '''
        D = self.D
        trainX = self.trainX
        trainY = self.trainY
        testX  = self.testX
        testY  = self.testY
        w = np.zeros( ( self.combinations_count( D,2 ), 3 ) )
        i = 0
        combi_list = list( itertools.combinations( list( range(D) ), 2 ) )
        for combi in combi_list:
            tmpX = np.vstack([ trainX[:, combi[0]], trainX[:, combi[1]], np.ones( self.trainN ) ]).T
            w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
            i += 1
        
        R2_double = np.zeros( self.combinations_count( D,2 ) )
        i = 0
        for combi in combi_list:
            tmpX = np.vstack([ testX[:,combi[0]], testX[:,combi[1]], np.ones( self.testN ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_double[i] = self.calcR2( testY, Yhat )
            i += 1
        return w, R2_double, np.array(combi_list)
    
    def calcSinglePolyRegression( self ):
        '''
        calculate weight (including bias term) for each column of X to explain the objective value Y
        the one column vector X will be comverted to [ X**2, X, 1 ] which is 2nd order polynomial
        because of this conversion, the size of w will be Dx3
        the w is calculated for trainX (check self.trainX) for explaining trainY
        output R2 is calculated for testX and testY
        output
            w  : list of weights for 2nd order polynomial of X[:,i] which is sequence of row vectors
            R2 : R-Squared values for each w which evaluates each weights in the list above w
                 each R2 is how good the No.i row of w explanes testY
                 max is 1 (the best explanation)
            combi_pattern : combination of X columns for each row of w
                            w[:,i] is calclated from X columns of combi_pattern[i] and bias term
        '''
        D = self.D
        trainX = self.trainX
        trainY = self.trainY
        testX  = self.testX
        testY  = self.testY
        w  = np.zeros( ( D, 3 ) )
        for i in range( D ):
            tmpX = np.vstack([ trainX[:,i]**2, trainX[:,i], np.ones( self.trainN ) ]).T
            w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
        
        R2_singlepoly = np.zeros( D )
        for i in range( D ):
            tmpX = np.vstack([ testX[:,i]**2, testX[:,i], np.ones( self.testN ) ]).T
            Yhat = tmpX.dot( w[i,:] )
            R2_singlepoly[i] = self.calcR2( testY, Yhat )
        return w, R2_singlepoly
    
    def calcDoublePolyRegression( self ):
        '''
        calculate weight (including bias term) for 2 columns in X to explain the objective value Y
        the two columns vector X1 and X2 will be comverted to
        [ X1**2, X2**2, X1*X2, X1, X2, 1 ] which is 2nd order polynomial
        because of this conversion, the size of w will be Dx6
        so the number of records for training (check self.trainN) have to be equal or larger than 6
        the w is calculated for trainX (check self.trainX) for explaining trainY
        output R2 is calculated for testX and testY
        output
            w  : list of weights for 2nd order polynomial of X[:,i and j] which is sequence of row vectors
            R2 : R-Squared values for each w which evaluates each weights in the list above w
                 each R2 is how good the No.i row of w explanes testY
                 max is 1 (the best explanation)
            combi_pattern : combination of X columns for each row of w
                            w[:,i] is calclated from X columns of combi_pattern[i] and bias term
        '''
        D = self.D
        trainX = self.trainX
        trainY = self.trainY
        testX  = self.testX
        testY  = self.testY
        if trainX.shape[0] < 6:
            print( 'the number of samples should be more than 6!!' )
            return
        D = trainX.shape[1]
        w = np.zeros( ( self.combinations_count( D,2 ), 6 ) )
        combi_list = list( itertools.combinations( list( range(D) ), 2 ) )
        i = 0
        for combi in combi_list:
            tX1 = trainX[:,combi[0]]
            tX2 = trainX[:,combi[1]]
            tmpX = np.vstack([ tX1**2, tX2**2, tX1*tX2, tX1, tX2, np.ones( self.trainN ) ]).T
            w[i,:] = np.linalg.solve( tmpX.T.dot( tmpX ), tmpX.T.dot( trainY ) )
            i += 1
        if np.isscalar( testX ) == True:
            return w
        else:
            R2_doubpoly = np.zeros( self.combinations_count( D,2 ) )
            i = 0
            for combi in combi_list:
                tX1 = testX[:,combi[0]]
                tX2 = testX[:,combi[1]]
                tmpX = np.vstack([ tX1**2, tX2**2, tX1*tX2, tX1, tX2, np.ones( self.testN ) ]).T
                Yhat = tmpX.dot( w[i,:] )
                R2_doubpoly[i] = self.calcR2( testY, Yhat )
                i += 1
            return w, R2_doubpoly, np.array(combi_list)
        
    def makeSingleData( self, colidx ):
        '''
        creates a Data Matrix for single variable regression
        input
            colidx : index number for target X column
        output
            totalX  : all records X has is returned with bias term
            traiinX : first self.trainN records of totalX
            testX   : last self.testN records of totalX
        '''
        X = self.origX
        totalX = np.vstack([ X[:, colidx], np.ones( self.N ) ]).T
        trainX = totalX[ :self.trainN, : ]
        testX  = totalX[ self.trainN:, : ]
        return totalX, trainX, testX
    
    def makeDoubleData( self, colidx1, colidx2 ):
        '''
        creates a Data Matrix for double variables regression
        input
            colidx1 : 1st index number for target X column
            colidx2 : 2nd index number for target X column
        output
            totalX  : all records X has is returned with bias term
            traiinX : first self.trainN records of totalX
            testX   : last self.testN records of totalX
        '''
        X = self.origX
        totalX = np.vstack([ X[:,colidx1],
                             X[:,colidx2],
                             np.ones( self.N ) ]).T
        trainX = totalX[ :self.trainN, : ]
        testX  = totalX[ self.trainN:, : ]
        return totalX, trainX, testX
    
    def makeSinglePolyData( self, colidx ):
        '''
        creates a Data Matrix for single variable polynomial regression
        input
            colidx : index number for target X column
        output
            totalX  : all records X has is returned with bias term (2nd order polynomial)
            traiinX : first self.trainN records of totalX
            testX   : last self.testN records of totalX
        '''
        X = self.origX
        totalX = np.vstack([ X[:,colidx]**2,
                             X[:,colidx],
                             np.ones( self.N ) ]).T
        trainX = totalX[ :self.trainN, : ]
        testX  = totalX[ self.trainN:, : ]
        return totalX, trainX, testX
    
    def makeDoublePolyData( self, colidx1, colidx2 ):
        '''
        creates a Data Matrix for double variables polynomial regression
        input
            colidx1 : 1st index number for target X column
            colidx2 : 2nd index number for target X column
        output
            totalX  : all records X has is returned with bias term (2nd order polynomial)
            traiinX : first self.trainN records of totalX
            testX   : last self.testN records of totalX
        '''
        X = self.origX
        totalX = np.vstack([ X[:,colidx1]**2,
                             X[:,colidx2]**2,
                             X[:,colidx1]*X[:,colidx2],
                             X[:,colidx1],
                             X[:,colidx2],
                             np.ones( self.N ) ]).T
        trainX = totalX[ :self.trainN, : ]
        testX  = totalX[ self.trainN:, : ]
        return totalX, trainX, testX
    
    def findBestR2Score( self ):
        '''
        try all regression on this class and find the best method (and its construction)
        output
            best_w     : the best weight for the regression
            best_R2    : the best R-Squared score for the regression
            best_combi : (if it has double variable) the best combination of variables
        '''
        w_single,   R2_single             = self.calcSingleRegression()
        w_singpoly, R2_singpoly           = self.calcSinglePolyRegression()
        w_double,   R2_double,   d_combi  = self.calcDoubleRegression()
        w_doubpoly, R2_doubpoly, dp_combi = self.calcDoublePolyRegression()

        max_single   = ( R2_single.max(),   np.argmax( R2_single ) )
        max_singpoly = ( R2_singpoly.max(), np.argmax( R2_singpoly ) )
        max_double   = ( R2_double.max(),   np.argmax( R2_double ) )
        max_doubpoly = ( R2_doubpoly.max(), np.argmax( R2_doubpoly ) )

        catNo = np.argmax( [max_single[0], max_singpoly[0], max_double[0], max_doubpoly[0]] )
        best_w     = []
        best_R2    = []
        best_combi = []
        if   catNo == 0:    # single variable regression
            print( 'Single Variable Regression is the best score of R-Squared' )
            print( 'the idx:{0} column of X is the best explanatory veriable'.format( max_single[1] ) )
            best_w     = w_single[ max_single[1], : ]
            best_R2    = max_single[0]
        elif catNo == 1:    # single polynomial regression
            print( 'Polynomial of Single Variable Regression is the best score of R-Squared' )
            print( 'the idx:{0} column of X is the best combination for polynomial'.format( max_single[1] ) )
            best_w     = w_singpoly[ max_single[1], : ]
            best_R2    = max_singpoly[0]
        elif catNo == 2:    # double variables regression
            bestpair = d_combi[ max_double[1] ]
            print( 'Double Variables Regression is the best score of R-Squared' )
            print( 'the idx:{0} and idx:{1} columns of X are the best combination for explanatory variables'.format( bestpair[0], bestpair[1] ) )
            best_w     = w_double[ max_double[1], : ]
            best_R2    = max_double[0]
            best_combi = bestpair
        elif catNo == 3:    # double polynomial regression
            bestpair = dp_combi[ max_doubpoly[1] ]
            print( 'Polynomial of Double Variables Regression is the best score of R-Squared' )
            print( 'the idx:{0} and idx:{1} columns of X are the best combination for polynomial'.format( bestpair[0], bestpair[1] ) )
            best_w     = w_doubpoly[ max_doubpoly[1], : ]
            best_R2    = max_doubpoly[0]
            best_combi = bestpair
        else:
            print( 'some error was occured' )
        
        return best_w, best_R2, best_combi









