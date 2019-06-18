# Comparison of sparse coding algorithms

Authors: Joseph Paul Cohen, Shawn Tan, Giancarlo Kerg, Yannick Pouliot, and Yoshua Bengio.
 
(Much of the code comes from other sources)

Here there are implementations of the following methods:

- Random
- Iterative Shrinkage-Thresholding Algorithm (ISTA)
- Iterative Shrinkage-Thresholding Algorithm (ISTA) Non-Negative
- Orthogonal Matching Pursuit
- Lasso Coordinate Descent
- Non-Negative Least Squares
- Bounded-Variable Least-Squares
- Differential Evolution
- Non-negative Matrix Factorization (NMF)
- Sparse Pseudo Inverse

![](eval.png)

The above plot was made with the following code:

```python
errorss = []
solvers_list = [solvers.DifferentialEvolution(strategy="best1bin",max_h=90),
                solvers.Random(num_tries=10000, max_h=10),
                solvers.ISTA(),
                solvers.ISTA_NN(),
                solvers.OrthogonalMatchingPursuit(),
                solvers.ISTA_NN(0.001,50.0),
                solvers.NMF(),
                solvers.NMF(beta_loss="kullback-leibler", solver="mu"),
                solvers.BVLS(max_h=90),
                solvers.Random(num_tries=1000, max_h=90),
                solvers.SparsePseudoInverse(),
                solvers.LassoCoordinateDescent(alpha=0.01)
               ]

for solver in solvers_list:
    errors = []
    for i in range(30):

        dataset = datasets.NoisySparseRandomDataset(n_in=5, n_out=30, sparsity=0.2, noise=i)

        hp = solver.solve(dataset.get_D(),dataset.get_signals())
        hp = np.maximum(hp,0)

        l1 = np.abs((dataset.get_h() - hp)).mean()
        l2 = (((dataset.get_h() - hp))**2).mean()
        
        errors.append(l2)
    errorss.append(errors)
    
plt.rcParams['figure.figsize'] = (15, 5)
linestyles = ['_', '-', '--', ':']
for lines in errorss:
    plt.plot(lines,np.random.choice(linestyles))
plt.legend([str(e) for e in solvers_list])
plt.xlabel("Gaussian Noise");
plt.ylabel("Mean Squared Error");
```
