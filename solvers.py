import numpy as np

class Solver:
    def __init__(self):
        pass

    def solve(self, D, signal):
        print "Must implement solve"
        pass
    
    def __str__(self):
         return str(self.__class__)

class Random(Solver):
    
    def __init__(self, num_tries, max_h):
        self.num_tries = num_tries
        self.max_h = max_h
        
    def __str__(self):
         return Solver.__str__(self) + " " + str({"num_tries":self.num_tries, "max_h": self.max_h})
    
    def solve(self, D, signals):

        def sub_solve(signal):
            best_error = 999999
            best_h = 0
            for i in range(self.num_tries):
                h = np.random.rand(D.shape[1])*self.max_h
                error = np.linalg.norm(np.dot(D,h)-signal)
                if error < best_error:
                    best_error = error
                    best_h = h
            return best_h
        return np.asarray([sub_solve(e) for e in signals])

class ISTA(Solver):

    def __init__(self, alpha=0.01, lamb=0.5):
        self.alpha = alpha
        self.lamb = lamb
        
    def __str__(self):
         return Solver.__str__(self) + " " + str({"alpha":self.alpha, "lamb": self.lamb})
    
    def solve(self, D, signals):

        def shrink(x, alpha):
            return np.sign(x)*np.maximum(np.abs(x)-alpha,0) 

        def ISTA_core(D, x, h, alpha, lamb):

            critical_value = max(np.linalg.eigvals(np.dot(D.T,D)))
            if (alpha*critical_value > 1):
                raise ValueError('Your alpha is too big for ISTA to converge.')

            converged = False
            while not converged:
                h_prime = h
                h = h - np.dot(alpha*D.T,np.dot(D,h)-x)
                h = shrink(h,alpha*lamb)
                if np.linalg.norm(h-h_prime) < 0.01:
                    converged = True 
            return h 
        
        def sub_solve(signal):
            init_h = np.ones(D.shape[1])
            return ISTA_core(D, signal, h=init_h, alpha=self.alpha, lamb=self.lamb)
    
        return np.asarray([sub_solve(e) for e in signals])
    
    
class ISTA_NN(Solver):

    def __init__(self, alpha=0.01, lamb=0.5):
        self.alpha = alpha
        self.lamb = lamb
        
    def __str__(self):
         return Solver.__str__(self) + " " + str({"alpha":self.alpha, "lamb": self.lamb})
            
    def solve(self, D, signals):

        def shrink(x, alpha):
            return np.sign(x)*np.maximum(np.abs(x)-alpha,0) 

        def ISTA_core(D, x, h, alpha, lamb):

            critical_value = max(np.linalg.eigvals(np.dot(D.T,D)))
            if (alpha*critical_value > 1):
                raise ValueError('Your alpha is too big for ISTA to converge.')

            converged = False
            while not converged:
                h_prime = h
                h = h - np.dot(alpha*D.T,np.dot(D,h)-x)
                h = shrink(h,alpha*lamb)
                h = np.maximum(h,0) ## every iter clamp negative values to 0
                if np.linalg.norm(h-h_prime) < 0.01:
                    converged = True 
            return h 
        
        def sub_solve(signal):
            init_h = np.ones(D.shape[1])
            return ISTA_core(D, signal, h=init_h, alpha=self.alpha, lamb=self.lamb)
    
        return np.asarray([sub_solve(e) for e in signals])

class OrthogonalMatchingPursuit(Solver):
    """
    Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang, Matching pursuits with time-frequency dictionaries, IEEE Transactions on Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415. (http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical Report - CS Technion, April 2008. http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    """
    def __init__(self, n_nonzero_coefs=None):
        self.n_nonzero_coefs = n_nonzero_coefs

    def solve(self, D, signals):
        
        if self.n_nonzero_coefs is None:
            self.n_nonzero_coefs = D.shape[1]
            
        from sklearn.linear_model import OrthogonalMatchingPursuit
        self.sksolver = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
        
        def sub_solve(signal):
            self.sksolver.fit(D, signal)
            return self.sksolver.coef_
    
        return np.asarray([sub_solve(e) for e in signals])
        
        
class LassoCoordinateDescent(Solver):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def solve(self, D, signals):
            
        from sklearn.linear_model import Lasso
        self.sksolver = Lasso(alpha=self.alpha)
        
        def sub_solve(signal):
            self.sksolver.fit(D, signal)
            return self.sksolver.coef_
    
        return np.asarray([sub_solve(e) for e in signals])
    
class NNLS(Solver):
    """
    Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM
    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls
    """
    def __init__(self):
        pass

    def solve(self, D, signals):
            
        import scipy, scipy.optimize
        
        def sub_solve(signal):
            return scipy.optimize.nnls(D, signal)[0]
    
        return np.asarray([sub_solve(e) for e in signals])
    
    
class BVLS(Solver):
    """
M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior, and Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization Problems," SIAM Journal on Scientific Computing, 1999.
P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares: an Algorithm and Applications", Computational Statistics, 1995.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
    """
    def __init__(self, max_h=10):
        self.max_h = max_h
        
    def __str__(self):
         return Solver.__str__(self) + " " + str({"max_h": self.max_h})

    def solve(self, D, signals):
            
        import scipy, scipy.optimize
        
        def sub_solve(signal):
            res = scipy.optimize.lsq_linear(D, signal, bounds=(0,self.max_h))
            return res['x']
    
        return np.asarray([sub_solve(e) for e in signals])

class DifferentialEvolution(Solver):
    """
Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """
    def __init__(self, strategy='best1bin', max_h=10):
        self.max_h = max_h
        self.strategy = strategy

    def __str__(self):
         return Solver.__str__(self) + " " + str({"max_h": self.max_h, "strategy":self.strategy})

    def solve(self, D, signals):
            
        import scipy, scipy.optimize
        
        def sub_solve(signal):
            
            def f(x):
                return ((np.dot(D,x) - signal)**2).mean()
            
            bounds = [(0, self.max_h)]*D.shape[1]
            return scipy.optimize.differential_evolution(f,bounds=bounds, strategy=self.strategy)['x']
    
        return np.asarray([sub_solve(e) for e in signals])

class NMF(Solver):
    def __init__(self, beta_loss='frobenius', solver='cd'):
        self.beta_loss = beta_loss
        self.solver = solver
    
    def __str__(self):
         return Solver.__str__(self) + " " + str({"beta_loss":self.beta_loss, "solver": self.solver})

    def solve(self, D, signals):
        from sklearn.decomposition import nmf
        ##print D.shape
        ##print signals.shape
        signals = np.array(signals, dtype=np.float32)
        signals[signals < 0] = 0
        W, _, _ = nmf.non_negative_factorization(
            signals, W=None, H=D.T, n_components=D.shape[1],
            init='random', update_H=False, solver=self.solver,
            beta_loss=self.beta_loss, tol=1e-4,
            max_iter=200, alpha=1., l1_ratio=1.,
            regularization="both", random_state=None,
            verbose=0, shuffle=False
        )
        return W

class OrthogonalMatchingPursuit2(Solver):
    """
        From https://github.com/davebiagioni/pyomp/blob/master/omp.py
        JPC: added ceil using NNLS
    """
    def __init__(self, max_h=None):
        self.max_h = max_h
        pass

    def solve(self, D, signals):
        
        def sub_solve(signal):
            return self.omp(D, signal, ceil=self.max_h).coef
    
        return np.asarray([sub_solve(e) for e in signals])
        
    def omp(self, X, y, nonneg=True, ceil=None, ncoef=None, maxit=200, tol=1e-3, ztol=1e-12, verbose=False):
        '''Compute sparse orthogonal matching pursuit solution with unconstrained
        or non-negative coefficients.

        Args:
            X: Dictionary array of size n_samples x n_features. 
            y: Reponse array of size n_samples x 1.
            nonneg: Enforce non-negative coefficients.
            ncoef: Max number of coefficients.  Set to n_features/2 by default.
            tol: Convergence tolerance.  If relative error is less than
                tol * ||y||_2, exit.
            ztol: Residual covariance threshold.  If all coefficients are less 
                than ztol * ||y||_2, exit.
            verbose: Boolean, print some info at each iteration.

        Returns:
            result:  Result object.  See Result.__doc__
        '''

        class Result(object):
            '''Result object for storing input and output data for omp.  When called from 
            `omp`, runtime parameters are passed as keyword arguments and stored in the 
            `params` dictionary.
            Attributes:
                X:  Predictor array after (optional) standardization.
                y:  Response array after (optional) standarization.
                ypred:  Predicted response.
                residual:  Residual vector.
                coef:  Solution coefficients.
                active:  Indices of the active (non-zero) coefficient set.
                err:  Relative error per iteration.
                params:  Dictionary of runtime parameters passed as keyword args.   
            '''

            def __init__(self, **kwargs):

                # to be computed
                self.X = None
                self.y = None
                self.ypred = None
                self.residual = None
                self.coef = None
                self.active = None
                self.err = None

                # runtime parameters
                self.params = {}
                for key, val in kwargs.iteritems():
                    self.params[key] = val

            def update(self, coef, active, err, residual, ypred):
                '''Update the solution attributes.
                '''
                self.coef = coef
                self.active = active
                self.err = err
                self.residual = residual
                self.ypred = ypred
        
        import scipy, scipy.optimize
        
        def norm2(x):
            return np.linalg.norm(x) / np.sqrt(len(x))

        # initialize result object
        result = Result(nnoneg=nonneg, ncoef=ncoef, maxit=maxit, tol=tol, ztol=ztol)
        if verbose:
            print(result.params)

        # check types, try to make somewhat user friendly
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(y) is not np.ndarray:
            y = np.array(y)

        # check that n_samples match
        if X.shape[0] != len(y):
            print('X and y must have same number of rows (samples)')
            return result

        # store arrays in result object    
        result.y = y
        result.X = X

        # for rest of call, want y to have ndim=1
        if np.ndim(y) > 1:
            y = np.reshape(y, (len(y),))

        # by default set max number of coef to half of total possible
        if ncoef is None:
            ncoef = int(X.shape[1]/2)

        # initialize things
        X_transpose = X.T                        # store for repeated use
        #active = np.array([], dtype=int)         # initialize list of active set
        active = []
        coef = np.zeros(X.shape[1], dtype=float) # solution vector
        residual = y                             # residual vector
        ypred = np.zeros(y.shape, dtype=float)
        ynorm = norm2(y)                         # store for computing relative err
        err = np.zeros(maxit, dtype=float)       # relative err vector

        # Check if response has zero norm, because then we're done. This can happen
        # in the corner case where the response is constant and you normalize it.
        if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
            if verbose:
                print('Norm of the response is less than convergence tolerance.')
            result.update(coef, active, err[0], residual, ypred)
            return result

        # convert tolerances to relative
        tol = tol * ynorm       # convergence tolerance
        ztol = ztol * ynorm     # threshold for residual covariance

        if verbose:
            print('\nIteration, relative error, number of non-zeros')

        # main iteration
        for it in range(maxit):

            # compute residual covariance vector and check threshold
            rcov = np.dot(X_transpose, residual)
            if nonneg:
                i = np.argmax(rcov)
                rc = rcov[i]
            else:
                i = np.argmax(np.abs(rcov))
                rc = np.abs(rcov[i])
            if rc < ztol:
                if verbose:
                    print('All residual covariances are below threshold.')
                break

            # update active set
            if i not in active:
                #active = np.concatenate([active, [i]], axis=1)
                active.append(i)

            # solve for new coefficients on active set
            if nonneg:
                if ceil is None:
                    coefi, _ = scipy.optimize.nnls(X[:, active], y)
                else:
                    #print "BVLS"
                    res = scipy.optimize.lsq_linear(X[:, active], y, bounds=(0,ceil))
                    coefi = res['x']
            elif ceil:
                print "Must be nonneg and ceil"
                break
            else:
                coefi, _, _, _ = np.linalg.lstsq(X[:, active], y)
            coef[active] = coefi   # update solution

            # update residual vector and error
            residual = y - np.dot(X[:,active], coefi)
            ypred = y - residual
            err[it] = norm2(residual) / ynorm  

            # print status
            if verbose:
                print('{}, {}, {}'.format(it, err[it], len(active)))

            # check stopping criteria
            if err[it] < tol:  # converged
                if verbose:
                    print('\nConverged.')
                break
            if len(active) >= ncoef:   # hit max coefficients
                if verbose:
                    print('\nFound solution with max number of coefficients.')
                break
            if it == maxit-1:  # max iterations
                if verbose:
                    print('\nHit max iterations.')

        result.update(coef, active, err[:(it+1)], residual, ypred)
        return result
    

class SparsePseudoInverse(Solver):
    def __init__(self, l1=1.):
        self.l1 = l1

    def __str__(self):
        return Solver.__str__(self) + " " + str({"l1":self.l1})

    def solve(self, D, signals):
        W = np.dot(signals, np.linalg.pinv(D).T)
        W = np.sign(W) * (abs(W) - self.l1)
        return W
    
class Baseline(Solver):
    def __init__(self, method="mean"):
        self.method = method
        
    def __str__(self):
        return Solver.__str__(self) + " " + str({"method":self.method})

    def solve(self, D, signals):

        import scipy, scipy.stats
        solved = []
        for transcript in range(D.shape[1]):
            masked = (signals)*D[:,transcript]
            
            if self.method == "mean":
                agg = masked.mean(axis=1)
            elif self.method == "median":
                agg = np.median(masked, axis=1)
            elif self.method == "gmean":
                agg = scipy.stats.mstats.gmean(masked, axis=1)
            solved.append(agg)
        solved = np.stack(solved, axis=1)
        
        return solved
