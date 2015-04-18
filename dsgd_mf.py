import sys
import math
import random
import csv
import numpy
from pyspark import SparkContext
from scipy import sparse

spark = 0 
factors = 0
workers = 0
iterations = 0
betaConst = 0
lambdaConst = 0
N = []


# stochastic gradient descent using l2 loss function
# input matrix: V, matrix: W, matrix: H, total iterations: totalIter
# number of elements in row i and j of whole matrix V: Ni, Nj, block range blockRange
def SGD(V,W,H, totalIter , Ni, Nj, blockRange):
    global lambdaConst, betaConst
    nonzero = V.nonzero()
    t0 = 100
    I = V.nonzero()[0].size
    for i in xrange(I):
        rand = random.randint(0, I-1)
        row,col = nonzero[0][rand], nonzero[1][rand]
        value = V[row,col]
        learning = (t0 + i+ totalIter )**(-betaConst)
        Wr = W[row,:]
        Hc = H[:,col]
        common = (value - numpy.dot(Wr, Hc))
        gradientW = -2 * common * Hc + (2*lambdaConst*Wr.T)/Ni[row]
        gradientH = -2 * common * Wr.T + (2*lambdaConst*Hc)/Nj[col]
        W[row,:] = Wr - learning*gradientW
        H[:,col] = Hc - learning*gradientH
    return (W,H,I,blockRange)



# matrixFactorization takes matrix V and factorizes into two matrices W and H. Such that W*H = V
# Created blocks and distributed the block to different workers to do stochastic gradient descent on
# input V
# output W,H
def matrixFactorization(V):
    global iterations, workers, factors, spark
    W = numpy.random.random_sample((V.shape[0], factors))
    H = numpy.random.random_sample((factors, V.shape[1]))
    blockDim = (V.shape[0]/workers, V.shape[1]/workers)
    
    #Block row and col range lists
    rowRangeList = [[block*blockDim[0],(block+1)*blockDim[0]] for block in xrange(workers)]
    colRangeList = [[block*blockDim[1],(block+1)*blockDim[1]] for block in xrange(workers)]

    rowRangeList[-1][1] += V.shape[0]%workers
    colRangeList[-1][1] += V.shape[1]%workers


    for iteration in xrange(iterations):

        iterations = 0
        for strata in xrange(workers):
            items = []
            variables = []
            for block in xrange(workers):

                rowRange = [rowRangeList[block][0], rowRangeList[block][1]]
                colRange = [colRangeList[block][0], colRangeList[block][1]]
                Vn = V[rowRange[0]:rowRange[1], colRange[0]:colRange[1]]
                Wn = W[rowRange[0]:rowRange[1],:]
                Hn = H[:, colRange[0]:colRange[1]]
                Ni = {}
                for i in xrange(rowRange[0],rowRange[1]):
                    Ni[i-rowRange[0]] = V[i,:].nonzero()[0].size 
                Nj = {}
                for i in xrange(colRange[0],colRange[1]):
                    Nj[i-colRange[0]] = V[:,i].nonzero()[0].size 
                if (Vn.nonzero()[0].size != 0):
                    variables.append([Vn, Wn, Hn, iterations, Ni, Nj, (rowRange, colRange)])
            results = spark.parallelize(variables, workers).map(lambda x: SGD(x[0],x[1],x[2],x[3],x[4],x[5],x[6])).collect()
            for result in results:
                iterations += result[2]
                rowRange,colRange = result[3]
                W[rowRange[0]:rowRange[1],:] = result[0]
                H[:,colRange[0]:colRange[1]] = result[1]
            colRangeList.insert(0,colRangeList.pop())
            

    return W,H

            
#Given path opens file and returns matrix as list of (rowIndices, colIndices, values)
def openFile(path):
    fi = open(path[0].replace("file:",""))
    csvReader = csv.reader(fi)
    rows = []
    cols = []
    values = []
    for line in csvReader:
        # Data is 1 indexed
        rows.append(int(line[0]) - 1)
        cols.append(int(line[1]) - 1)
        values.append(float(line[2]))
    fi.close()
    return (rows,cols,values)

#Writes matrix to file in dense form
def writeFile(matrix, path):
    f = open(path, 'w', 100)
    rows= matrix.shape[0]
    cols = matrix.shape[1]
    for row in xrange(rows):
        for col in xrange(cols):
            
            if col == cols-1:
                f.write(str(matrix[row,col])) 
            else:
                f.write(str(matrix[row,col]) + ",")
        f.write("\n")

    f.flush()
    f.close()

def main():
    global spark, factors, workers, iterations, betaConst, lambdaConst
    spark = SparkContext("local", "Matrix-Factorization")
    factors = int(sys.argv[1])
    workers = int(sys.argv[2])
    iterations = int(sys.argv[3])
    betaConst = float(sys.argv[4])
    lambdaConst = float(sys.argv[5])
    inputPath = sys.argv[6]
    outputPathW = sys.argv[7]
    outputPathH = sys.argv[8]
    #load data

    matrixParts = spark.wholeTextFiles(inputPath).map(lambda x:openFile(x)).collect()

    allRows = []
    allCols = []
    allValues = []
    for part in matrixParts:
        allRows += part[0]
        allCols += part[1]
        allValues += part[2]

    V = sparse.csr_matrix((allValues, (allRows, allCols)))
    W,H = matrixFactorization(V)
    writeFile(W,outputPathW)
    writeFile(H,outputPathH)

if __name__ == '__main__':
    main()
    