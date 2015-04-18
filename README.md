# Distributed-Matrix-Factorization
 A Machine Learning project: Distributed Stochastic Gradient Descent method for Matrix Factorization in pySpark
 by Sudev Bohra
 



## How to Run
* Run
```
$ $(SPARK) dsgd_mf.py $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TRAINV) $(OUTPUTW) $(OUTPUTH)  
```
Example
```
$ spark-submit dsgd_mf.py 20 5 100 0.9 1.0 test.csv w.csv h.csv  
```
## Input
Takes input of matrix V in sparse format. One row, col, value per line
eg. the 2x2 identity matrix would look like
$ 0,0,1
$ 1,1,1


## Output
Outputs file w.csv and h.csv in dense matrix format

