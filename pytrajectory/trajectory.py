
import numpy as np
import sympy as sp
import scipy as scp
from scipy.sparse.csr import csr_matrix
import pickle

from spline import CubicSpline, fdiff
from solver import Solver
from simulation import Simulation
from utilities import IntegChain, plotsim
import log

# DEBUGGING
DEBUG = True

if DEBUG:
    from IPython import embed as IPS


class Trajectory():
    '''
    Base class of the PyTrajectory project.

    Trajectory manages everything from analysing the given system over
    initialising the spline functions, setting up and solving the collocation
    equation system up to the simulation of the resulting initial value problem.

    After the iteration has finished, it provides access to callable functions
    for the system and input variables as well as some capabilities for
    visualising the systems dynamic.


    Parameters
    ----------

    ff :  callable
        Vectorfield (rhs) of the control system
    
    a : float
        Left border
    
    b : float
        Right border
    
    xa : list
        Boundary values at the left border
    
    xb : list
        Boundary values at the right border
    
    g : list
        Boundary values of the input variables
    
    
    Attributes
    ----------
    
    mparam : dict
        Dictionary with method parameters
        
        ==========  =============   =======================================================
        key         default value   meaning
        ==========  =============   =======================================================
        sx          5               Initial number of spline parts for the system variables
        su          5               Initial number of spline parts for the input variables
        kx          2               Factor for raising the number of spline parts
        delta       2               Constant for calculation of collocation points
        maxIt       7               Maximum number of iteration steps
        eps         1e-2            Tolerance for the solution of the initial value problem
        ierr        0.0             Tolerance for the error on the whole interval
        tol         1e-5            Tolerance for the solver of the equation system
        method      'leven'         The solver algorithm to use
        use_chains  True            Whether or not to use integrator chains
        colltype    'equidistant'   The type of the collocation points
        use_sparse  True            Whether or not to use sparse matrices
        ==========  =============   =======================================================
    '''
    
    def __init__(self, ff, a=0.0, b=1.0, xa=None, xb=None, g=None, constraints=None, **kwargs):
        # Enable logging
        if not DEBUG:
            log.log_on(verbosity=1)
        
        # Save the symbolic vectorfield
        self.ff_sym = ff
        
        # The borders of the considered time interval
        self.a = a
        self.b = b
        
        # Boundary values for the input variable(s)
        self.uab = g
        
        # Set default values for method parameters
        self.mparam = {'sx' : 5,
                        'su' : 5,
                        'kx' : 2,
                        'delta' : 2,
                        'maxIt' : 10,
                        'eps' : 1e-2,
                        'ierr' : 0.0,
                        'tol' : 1e-5,
                        'method' : 'leven',
                        'use_chains' : True,
                        'colltype' : 'equidistant',
                        'use_sparse' : True,
                        'ncoll' : 10}
        
        # Change default values of given kwargs
        for k, v in kwargs.items():
            self.setParam(k, v)
        
        # Initialise dimensions --> they will be set afterwards
        self.n = 0
        self.m = 0

        # Analyse the given system to set some parameters
        self.analyseSystem()
        
        # A little check
        if not (len(xa) == len(xb) == self.n):
            raise ValueError('Dimension mismatch xa,xb')

        # Reset iteration number
        self.nIt = 0

        # Dictionary for spline objects
        #   key: variable   value: CubicSpline-object
        self.splines = dict()

        # Dictionaries for callable (solution) functions
        self.x_fnc = dict()
        self.u_fnc = dict()
        self.dx_fnc = dict()

        # Create symbolic variables
        self.x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1,self.n+1)])
        self.u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1,self.m+1)])
        
        # Dictionaries for boundary conditions
        self.xa = dict()
        self.xb = dict()
        
        for i,xx in enumerate(self.x_sym):
            self.xa[xx] = xa[i]
            self.xb[xx] = xb[i]
        
        # HERE WE HANDLE THE STATE CONSTRAINTS
        self.constraints = constraints
        if self.constraints:
            self.constrain(constraints)
            
            # Create a numeric vectorfield function of the original vectorfield
            F_orig = sp.Matrix(self.ff_sym_orig(self.x_sym_orig,self.u_sym))
            _ff_num_orig = sp.lambdify(self.x_sym_orig+self.u_sym, F_orig, modules='numpy')
            
            def ff_num_orig(x, u):
                xu = np.hstack((x, u))
                return np.array(_ff_num_orig(*xu)).squeeze()
            
            self.ff_orig = ff_num_orig
        
        # Now we transform the symbolic function of the vectorfield to
        # a numeric one for faster evaluation

        # Create sympy matrix of the vectorfield
        F = sp.Matrix(self.ff_sym(self.x_sym,self.u_sym))

        # Use lambdify to replace sympy functions in the vectorfield with
        # numpy equivalents
        _ff_num = sp.lambdify(self.x_sym+self.u_sym, F, modules='numpy')

        # Create a wrapper as the actual function because of the behaviour
        # of lambdify()
        def ff_num(x, u):
            xu = np.hstack((x, u))
            return np.array(_ff_num(*xu)).squeeze()
        
        # This is now the callable (numerical) vectorfield of the system
        self.ff = ff_num
        
        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

        # and likewise this should not be existent yet
        self.sol = None

        # just for me
        #print np.__version__
        #print sp.__version__
        #print scp.__version__
    
    
    def constrain(self, constraints):
        '''
        This method is used to enable compliance with the desired box constraints
        by employing appropriate saturation functions for the state variables.
        '''
        ff = self.ff_sym(self.x_sym, self.u_sym)
        ff = sp.Matrix(ff)
        
        xa = self.xa
        xb = self.xb
        
        x_sym = self.x_sym
        
        # backup original state variables and their boundary values
        x_sym_orig = 1*x_sym
        xa_orig = xa.copy()
        xb_orig = xb.copy()
        
        ff_sym_orig = self.ff_sym
        
        for k, v in constraints.items():
            # check if boundary values are within constraints
            #if not( v[0] <= xa[x_sym[k]] <= v[1] and v[0] <= xb[x_sym[k]] <= v[1] ):
            #    print "ERROR: boundary value not within constrints"
            assert( v[0] <= xa[x_sym[k]] <= v[1] )
            assert( v[0] <= xb[x_sym[k]] <= v[1] )
            
            # replace constrained state variable with new unconstrained one
            x_sym[k] = sp.Symbol('y%d'%(k+1))
            
            # calculate saturation function expression and its derivative
            yk = x_sym[k]
            m = 4.0/(v[1] - v[0])
            psi = v[1] - (v[1]-v[0])/(1.0+sp.exp(m*yk))
            dpsi = ((v[1]-v[0])*m*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            
            # replace constrained variables in vectorfield with saturation expression
            ff = ff.replace(x_sym_orig[k], psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            ff[k] = ff[k] / dpsi
            
            # replace constrained variable in dictionaries for boundary values
            xk = x_sym_orig[k]
            xa[yk] = xa.pop(xk)
            xb[yk] = xb.pop(xk)
            
            # update boundary values for new unconstrained variable
            wa = xa[yk]
            xa[yk] = (1/m)*np.log( (wa-v[0])/(v[1]-wa) )
            wb = xb[yk]
            xb[yk] = (1/m)*np.log( (wb-v[0])/(v[1]-wb) )
        
        _ff_sym = sp.lambdify(self.x_sym+self.u_sym, ff, modules='sympy')
        
        def ff_sym(x,u):
            xu = np.hstack((x,u))
            return np.array(_ff_sym(*xu)).squeeze()
        
        self.xa = xa
        self.xb = xb
        self.ff_sym = ff_sym
        
        self.ff_sym_orig = ff_sym_orig
        self.x_sym_orig = x_sym_orig
        self.xa_orig = xa_orig
        self.xb_orig = xb_orig
        
    
    def unconstrain(self, constraints):
        x_fnc = self.x_fnc
        dx_fnc = self.dx_fnc
        
        for k, v in self.constraints.items():
            xk = self.x_sym_orig[k]
            yk = self.x_sym[k]
            y0, y1 = v
            
            y_fnc = x_fnc[yk]
            dy_fnc = dx_fnc[yk]
            m = 4.0/(y1-y0)
            
            def psi_y(t):
                y = y_fnc(t)
                return y1 - (y1-y0)/(1.0+np.exp(m*y))
            
            def dpsi_dy(t):
                y = y_fnc(t)
                dy = dy_fnc(t)
                return dy * (4.0*np.exp(m*y))/(1.0+np.exp(m*y))**2
            
            self.x_fnc[xk] = psi_y
            self.x_fnc.pop(yk)
            self.dx_fnc[xk] = dpsi_dy
            self.dx_fnc.pop(yk)
            
            self.xa = self.xa_orig
            self.xb = self.xb_orig
            self.x_sym = self.x_sym_orig
        
        ##########################
        # recreate vectorfield
        F = sp.Matrix(self.ff_sym_orig(self.x_sym,self.u_sym))
        _ff_num = sp.lambdify(self.x_sym+self.u_sym, F, modules='numpy')
        
        def ff_num(x, u):
            xu = np.hstack((x, u))
            return np.array(_ff_num(*xu)).squeeze()
        
        self.ff = ff_num
        ##########################
            
        #IPS()
    

    def startIteration(self):
        '''
        This is the main loop.

        At first the equations that have to be solved by collocation will be
        determined according to the integrator chains.

        Next, one step of the iteration is done by calling :meth:`iterate()`.

        After that, the accuracy of the found solution is checked.
        If it is within the tolerance range the iteration will stop.
        Else, the number of spline parts is raised and another step starts.
        
        
        Returns
        -------
        
        x : callable
            Callable function for the system state.
        
        u : callable
            Callable function for the input variables.
        '''

        #log.info(40*"#", verb=3)
        #log.info("       ---- First Iteration ----")
        #log.info(40*"#", verb=3)
        #log.info("# spline parts: %d"%self.mparam['sx'])
        log.info("1st Iteration: %d spline parts"%self.mparam['sx'], verb=1)
        
        # resetting integrator chains according to value of self.use_chains
        if not self.mparam['use_chains']:
            self.chains = dict()

        # now we determine the equations that have to be solved by collocation
        # (--> lower ends of integrator chains)
        eqind = []

        if self.chains:
            # iterate over all integrator chains
            for ic in self.chains:
                # if lower end is a system variable
                # then its equation has to be solved
                if ic.lower.name.startswith('x'):
                    idx = self.x_sym.index(ic.lower)
                    eqind.append(idx)
            eqind.sort()
        else:
            # if integrator chains should not be used
            # then every equation has to be solved by collocation
            eqind = range(self.n)

        # save equation indices
        self.eqind = np.array(eqind)

        # start first iteration
        self.iterate()
        
        # check if desired accuracy is already reached
        self.checkAccuracy()
        
        # this was the first iteration
        # now we are getting into the loop
        while not self.reached_accuracy and self.nIt < self.mparam['maxIt']:
            # raise the number of spline parts
            self.mparam['sx'] = int(round(self.mparam['kx']*self.mparam['sx']))
            
            #log.info( 40*"#", verb=3)
            #log.info("       ---- Next Iteration ----")
            #log.info( 40*"#", verb=3)
            #log.info("# spline parts: %d"%self.mparam['sx'])
            if self.nIt == 1:
                log.info("2nd Iteration: %d spline parts"%self.mparam['sx'], verb=1)
            elif self.nIt == 2:
                log.info("3rd Iteration: %d spline parts"%self.mparam['sx'], verb=1)
            elif self.nIt >= 3:
                log.info("%dth Iteration: %d spline parts"%(self.nIt+1, self.mparam['sx']), verb=1)

            # store the old spline to calculate the guess later
            self.old_splines = self.splines

            # start next iteration
            self.iterate()

            # check if desired accuracy is reached
            self.checkAccuracy()
            
        # clear workspace
        self.clear()
        
        # HERE WE HAVE TO UNCONSTRAIN THE STATE FUNCTIONS
        if self.constraints:
            self.unconstrain(self.constraints)
        
        return self.x, self.u


    def analyseSystem(self):
        '''
        Analyse the system structure and set values for some of the method
        parameters.

        By now, this method just determines the number of state and input variables
        and searches for integrator chains.
        '''

        #log.info( 40*"#", verb=3)
        #log.info("####   Analysing System Strucutre   ####")
        #log.info( 40*"#", verb=3)
        log.info("  Analysing System Structure", verb=2)
        
        # first, determine system dimensions
        log.info("    Determine system/input dimensions", verb=3)
        i = -1
        found_nm = False

        while not found_nm:
            # iteratively increase system and input dimensions and try to call
            # symbolic vectorfield ff_sym with i/j-dimensional vectors
            i += 1

            for j in xrange(i):
                x = np.ones(i)
                u = np.ones(j)

                try:
                    self.ff_sym(x, u)
                    # if no ValueError is raised, i is the system dimension
                    # and j is the dimension of the inputs
                    found_nm = True
                    n = i
                    m = j
                    break
                except ValueError:
                    # unpacking error inside ff_sym
                    # (that means the dimensions don't match)
                    pass
                    #if j == 0:
                    #    try:
                    #        self.ff_sym(x)
                    #        found_nm = True
                    #        n = i
                    #        m = j
                    #        break
                    #    except:
                    #        pass

        self.n = n
        self.m = m

        log.info("      --> system: %d"%n, verb=3)
        log.info("      --> input : %d"%m, verb=3)

        # next, we look for integrator chains
        log.info("    Looking for integrator chains", verb=3)

        # create symbolic variables to find integrator chains
        x_sym = ([sp.symbols('x%d' % k, type=float) for k in xrange(1,n+1)])
        u_sym = ([sp.symbols('u%d' % k, type=float) for k in xrange(1,m+1)])

        fi = self.ff_sym(x_sym, u_sym)

        chaindict = {}
        for i in xrange(len(fi)):
            # substitution because of sympy difference betw. 1.0 and 1
            if isinstance(fi[i], sp.Basic):
                fi[i] = fi[i].subs(1.0, 1)

            for xx in x_sym:
                if fi[i] == xx:
                    chaindict[xx] = x_sym[i]

            for uu in u_sym:
                if fi[i] == uu:
                    chaindict[uu] = x_sym[i]

        # chaindict looks like this:  {u_1 : x_2, x_4 : x_3, x_2 : x_1}
        # where x_4 = d x_3 / dt and so on

        # find upper ends of integrator chains
        uppers = []
        for vv in chaindict.values():
            if (not chaindict.has_key(vv)):
                uppers.append(vv)

        # create ordered lists that temporarily represent the integrator chains
        tmpchains = []

        # therefor we flip the dictionary to work our way through its keys
        # (former values)
        dictchain = {v:k for k,v in chaindict.items()}

        for var in uppers:
            tmpchain = []
            vv = var
            tmpchain.append(vv)

            while dictchain.has_key(vv):
                vv = dictchain[vv]
                tmpchain.append(vv)

            tmpchains.append(tmpchain)

        # create an integrator chain object for every temporary chain
        chains = []
        for lst in tmpchains:
            ic = IntegChain(lst)
            chains.append(ic)
            log.info("      --> found: " + str(ic), verb=3)

        self.chains = chains

        # get minimal neccessary number of spline parts
        # for the manipulated variables
        # TODO: implement this!?
        # --> (3.35)      ?????
        #nu = -1
        #...
        #self.su = self.n - 3 + 2*(nu + 1)  ?????


    def setParam(self, param='', val=None):
        '''
        Method to assign value :attr:`val` to method parameter :attr:`param`.
        (mainly for didactic purpose)

        Parameters
        ----------

        param : str
            Parameter of which to alter the value

        val : ???
            New value for the passed parameter
        '''
        
        # check if current and new value have the same type
        # --> should they always?
        assert type(val) == type(self.mparam[param])
        
        self.mparam[param] = val


    def iterate(self):
        '''
        This method is used to run one iteration step.

        First, new splines are initialised for the variables that are the upper
        end of an integrator chain.

        Then, a start value for the solver is determined and the equation
        system is build.

        Next, the equation system is solved and the resulting numerical values
        for the free parameters are written back.

        As a last, the initial value problem is simulated.
        '''
        self.nIt += 1
        
        # initialise splines
        with log.Timer("initSplines()"):
            self.initSplines()

        # Get first guess for solver
        with log.Timer("getGuess()"):
            self.getGuess()

        # create equation system
        with log.Timer("buildEQS()"):
            self.buildEQS()

        # solve it
        with log.Timer("solveEQS()"):
            self.solveEQS()

        # write back the coefficients
        with log.Timer("setCoeff()"):
            self.setCoeff()
        
        # solve the initial value problem
        with log.Timer("simulateIVP()"):
            self.simulateIVP()
    
    
    def getGuess(self):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrarily values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        '''

        if (self.nIt == 1):
            self.c_list = np.empty(0)

            for k, v in sorted(self.indep_coeffs.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, v))
            guess = 0.1*np.ones(len(self.c_list))
        else:
            # make splines local
            old_splines = self.old_splines
            new_splines = self.splines

            guess = np.empty(0)
            self.c_list = np.empty(0)

            # get new guess for every independent variable
            for k, v in sorted(self.coeffs_sol.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, self.indep_coeffs[k]))

                if (new_splines[k].type == 'x'):
                    log.info("    Get new guess for spline %s"%k.name, verb=3)

                    # how many unknown coefficients does the new spline have
                    nn = len(self.indep_coeffs[k])

                    # and this will be the points to evaluate the old spline in
                    #   but we don't want to use the borders because they got
                    #   the boundary values already
                    gpts = np.linspace(self.a,self.b,(nn+1),endpoint = False)[1:]

                    # evaluate the old and new spline at all points in gpts
                    #   they should be equal in these points

                    OLD = [None]*len(gpts)
                    NEW = [None]*len(gpts)
                    NEW_abs = [None]*len(gpts)

                    for i, p in enumerate(gpts):
                        OLD[i] = old_splines[k].f(p)
                        NEW[i], NEW_abs[i] = new_splines[k].f(p)

                    OLD = np.array(OLD)
                    NEW = np.array(NEW)
                    NEW_abs = np.array(NEW_abs)

                    TT = np.linalg.solve(NEW,OLD-NEW_abs)

                    guess = np.hstack((guess,TT))
                else:
                    # if it is a manipulated variable, just take the old solution
                    guess = np.hstack((guess, self.coeffs_sol[k]))

        # the new guess
        self.guess = guess


    def initSplines(self):
        '''
        This method is used to initialise the provisionally splines.
        '''
        #log.info( 40*"#", verb=3)
        #log.info( "#########  Initialise Splines  #########")
        #log.info( 40*"#", verb=3)
        log.info("  Initialise Splines", verb=2)
        
        # dictionaries for splines and callable solution function for x,u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()
        
        # make some stuff local
        chains = self.chains
        sx = self.mparam['sx']
        su = self.mparam['su']

        # first handle variables that are part of an integrator chain
        for ic in chains:
            upper = ic.upper

            # here we just create a spline object for the upper ends of every chain
            # w.r.t. its lower end
            if ic.lower.name.startswith('x'):
                splines[upper] = CubicSpline(self.a,self.b,n=sx,bc=[self.xa[upper],self.xb[upper]],steady=False,tag=upper.name)
                splines[upper].type = 'x'
            elif ic.lower.name.startswith('u'):
                splines[upper] = CubicSpline(self.a,self.b,n=su,bc=self.uab,steady=False,tag=upper.name)
                splines[upper].type = 'u'

            for i,elem in enumerate(ic.elements):
                if elem in self.u_sym:
                    if (i == 0):
                        u_fnc[elem] = splines[upper].f
                    if (i == 1):
                        u_fnc[elem] = splines[upper].df
                    if (i == 2):
                        u_fnc[elem] = splines[upper].ddf
                elif elem in self.x_sym:
                    if (i == 0):
                        splines[upper].bc = [self.xa[elem],self.xb[elem]]
                        if ((self.uab != None) and (splines[upper].type == 'u')):
                            splines[upper].bcd = self.uab
                        x_fnc[elem] = splines[upper].f
                    if (i == 1):
                        splines[upper].bcd = [self.xa[elem],self.xb[elem]]
                        if ((self.uab != None) and (splines[upper].type == 'u')):
                            splines[upper].bcdd = self.uab
                        x_fnc[elem] = splines[upper].df
                    if (i == 2):
                        splines[upper].bcdd = [self.xa[elem],self.xb[elem]]
                        x_fnc[elem] = splines[upper].ddf

        # now handle the variables which are not part of any chain
        for xx in self.x_sym:
            if (not x_fnc.has_key(xx)):
                splines[xx] = CubicSpline(self.a,self.b,n=sx,bc=[self.xa[xx],self.xb[xx]],steady=False,tag=str(xx))
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f

        for uu in self.u_sym:
            if (not u_fnc.has_key(uu)):
                splines[uu] = CubicSpline(self.a,self.b,n=su,bc=self.uab,steady=False,tag=str(uu))
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f
        
        # solve smoothness conditions of each spline
        for ss in splines:
            with log.Timer("makesteady()"):
                splines[ss].makesteady()

        for xx in self.x_sym:
            dx_fnc[xx] = fdiff(x_fnc[xx])

        indep_coeffs= dict()
        for ss in splines:
            indep_coeffs[ss] = splines[ss].c_indep

        self.indep_coeffs = indep_coeffs

        self.splines = splines
        self.x_fnc = x_fnc
        self.u_fnc = u_fnc
        self.dx_fnc = dx_fnc


    def buildEQS(self):
        '''
        Builds the collocation equation system.
        '''

        #log.info( 40*"#", verb=3)
        #log.info("####  Building the equation system  ####")
        #log.info( 40*"#", verb=3)
        log.info("  Building Equation System", verb=2)
        
        # make functions local
        x_fnc = self.x_fnc
        dx_fnc = self.dx_fnc
        u_fnc = self.u_fnc

        # make symbols local
        x_sym = self.x_sym
        u_sym = self.u_sym

        a = self.a
        b = self.b
        delta = self.mparam['delta']

        # now we generate the collocation points
        if self.mparam['colltype'] == 'equidistant':
            # get equidistant collocation points
            cpts = np.linspace(a,b,(self.mparam['sx']*delta+1),endpoint=True)
            #cpts = np.linspace(a,b,(self.mparam['sx']*self.mparam['ncoll']+1),endpoint=True)
        elif self.mparam['colltype'] == 'chebychev':
            # determine rank of chebychev polynomial
            # of which to calculate zero points
            nc = int(self.mparam['sx']*delta - 1)

            # calculate zero points of chebychev polynomial --> in [-1,1]
            cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
            cheb_cpts.sort()

            # transfer chebychev nodes from [-1,1] to our interval [a,b]
            a = self.a
            b = self.b
            chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

            # add left and right borders
            cpts = np.hstack((a, chpts, b))
        else:
            log.warn('Unknown type of collocation points.')
            log.warn('--> will use equidistant points!')
            cpts = np.linspace(a,b,(self.mparam['sx']*delta+1),endpoint=True)

        lx = len(cpts)*len(x_sym)
        lu = len(cpts)*len(u_sym)

        Mx = [None]*lx
        Mx_abs = [None]*lx
        Mdx = [None]*lx
        Mdx_abs = [None]*lx
        Mu = [None]*lu
        Mu_abs = [None]*lu

        # here we do something that will be explained after we've done it  ;-)
        indic = dict()
        i = 0
        j = 0
        
        # iterate over spline quantities
        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j

        # iterate over all quantities including inputs
        for sq in x_sym+u_sym:
            for ic in self.chains:
                if sq in ic:
                    indic[sq] = indic[ic.upper]

        # as promised: here comes the explanation
        #
        # now, the dictionary 'indic' looks something like
        #
        # indic = {u1 : (0, 6), x3 : (18, 24), x4 : (24, 30), x1 : (6, 12), x2 : (12, 18)}
        #
        # which means, that in the vector of all independent parameters of all splines
        # the 0th up to the 5th item [remember: Python starts indexing at 0 and leaves out the last]
        # belong to the spline created for u1, the items with indices from 6 to 11 belong to the
        # spline created for x1 and so on...

        # total number of independent coefficients
        c_len = len(self.c_list)

        eqx = 0
        equ = 0
        for p in cpts:
            for xx in x_sym:
                mx = np.zeros(c_len)
                mdx = np.zeros(c_len)

                i,j= indic[xx]

                mx[i:j], Mx_abs[eqx] = x_fnc[xx](p)
                mdx[i:j], Mdx_abs[eqx] = dx_fnc[xx](p)

                Mx[eqx] = mx
                Mdx[eqx] = mdx
                eqx += 1

            for uu in u_sym:
                mu = np.zeros(c_len)

                i,j = indic[uu]

                mu[i:j], Mu_abs[equ] = u_fnc[uu](p)

                Mu[equ] = mu
                equ += 1

        self.Mx = np.array(Mx)
        self.Mx_abs = np.array(Mx_abs)
        self.Mdx = np.array(Mdx)
        self.Mdx_abs = np.array(Mdx_abs)
        self.Mu = np.array(Mu)
        self.Mu_abs = np.array(Mu_abs)
        
        # here we create a callable function for the jacobian matrix of the vectorfield
        # w.r.t. to the system and input variables
        f = self.ff_sym(x_sym,u_sym)
        Df_mat = sp.Matrix(f).jacobian(x_sym+u_sym)
        self.Df = sp.lambdify(x_sym+u_sym, Df_mat, modules='numpy')

        # the following would be created with every call to self.DG but it is possible to
        # only do it once. So we do it here to speed things up.

        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = self.Mdx.reshape((len(cpts),-1,len(self.c_list)))[:,self.eqind,:]
        self.DdX = np.vstack(DdX)

        # here we compute the jacobian matrix of the system/input functions as they also depend on
        # the free parameters
        DXU = []
        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        for i in xrange(len(cpts)):
            DXU.append(np.vstack(( self.Mx[x_len*i:x_len*(i+1)], self.Mu[u_len*i:u_len*(i+1)] )))

        self.DXU = DXU

        if self.mparam['use_sparse']:
            self.Mx = csr_matrix(self.Mx)
            self.Mx_abs = csr_matrix(self.Mx_abs)
            self.Mdx = csr_matrix(self.Mdx)
            self.Mdx_abs = csr_matrix(self.Mdx_abs)
            self.Mu = csr_matrix(self.Mu)
            self.Mu_abs = csr_matrix(self.Mu_abs)

            self.DdX = csr_matrix(self.DdX)


    def solveEQS(self):
        '''
        This method is used to solve the collocation equation system.
        '''

        #log.info( 40*"#", verb=3)
        #log.info("#####  Solving the equation system  ####")
        #log.info( 40*"#", verb=3)
        log.info("  Solving Equation System", verb=2)
        
        # create our solver
        solver = Solver(self.G, self.DG, self.guess, tol= self.mparam['tol'],
                        method=self.mparam['method'])
        
        #IPS()
        
        # solve the equation system
        self.sol = solver.solve()


    def G(self, c):
        '''
        Returns the collocation system evaluated with numeric values for the
        independent parameters.
        '''

        ff = self.ff
        eqind = self.eqind

        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        X = self.Mx.dot(c) + self.Mx_abs
        U = self.Mu.dot(c) + self.Mu_abs

        X = np.array(X).reshape((-1,x_len))
        U = np.array(U).reshape((-1,u_len))

        # evaluate system equations and select those related
        # to lower ends of integrator chains (via eqind)
        # other equations need not to be solved
        F = np.array([ff(x,u) for x,u in zip(X,U)], dtype=float).squeeze()[:,eqind]

        dX = self.Mdx.dot(c) + self.Mdx_abs
        dX = np.array(dX).reshape((-1,x_len))[:,eqind]

        G = F-dX

        return G.flatten()


    def DG(self, c):
        '''
        Returns the jacobian matrix of the collocation system w.r.t. the
        independent parameters evaluated at :attr:`c`.
        '''

        # make callable function for the jacobian matrix of the vectorfield local
        Df = self.Df
        eqind = self.eqind

        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        # first we calculate the x and u values in all collocation points
        # with the current numerical values of the free parameters
        X = self.Mx.dot(c) + self.Mx_abs
        X = np.array(X).reshape((-1,x_len)) # one column for every state component
        U = self.Mu.dot(c) + self.Mu_abs
        U = np.array(U).reshape((-1,u_len)) # one column for every input component

        # now we compute blocks of the jacobian matrix of the vectorfield with those values
        DF_blocks = []
        for x,u in zip(X,U):
            # get one row of X and U respectively
            tmp_xu = np.hstack((x,u))

            # evaluate the jacobian of the vectorfield at current collocation point represented by
            # the row of X and U
            DF_blocks.append(Df(*tmp_xu))

        # because the system/input variables depend on the free parameters we have to multiply each
        # jacobian block with the jacobian matrix of the x/u functions w.r.t. the free parameters
        # --> see buildEQS()
        DF = []
        for i in xrange(len(DF_blocks)):
            res = np.dot(DF_blocks[i], self.DXU[i])
            assert res.shape == (x_len,len(self.c_list))
            DF.append(res)

        DF = np.array(DF)[:,eqind,:]
        # 1st index : collocation point
        # 2nd index : equations that have to be solved --> end of an integrator chain
        # 3rd index : component of c

        # now compute jacobian of x_dot w.r.t. to indep coeffs
        # --> see buildEQS()
        #DdX = self.Mdx.reshape((len(self.cpts),-1,len(self.c_list)))[:,eqind,:]
        DdX = self.DdX

        # stack matrices in vertical direction
        #DG = np.vstack(DF) - np.vstack(DdX)
        DG = np.vstack(DF) - DdX

        return DG


    def setCoeff(self):
        '''
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.
        '''

        log.info("    Set spline coefficients", verb=2)

        sol = self.sol
        subs = dict()

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            i = len(v)
            subs[k] = sol[:i]
            sol = sol[i:]

        for var in self.x_sym + self.u_sym:
            for ic in self.chains:
                if var in ic:
                    subs[var] = subs[ic.upper]

        # set numerical coefficients for each spline
        for cc in self.splines:
            self.splines[cc].set_coeffs(subs[cc])

        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            j += len(v)
            coeffs_sol[k] = self.sol[i:j]
            i = j

        self.coeffs_sol = coeffs_sol


    def simulateIVP(self):
        '''
        This method is used to solve the initial value problem.
        '''

        #log.info( 40*"#", verb=3)
        #log.info("##  Solving the initial value problem ##")
        #log.info( 40*"#", verb=3)
        log.info("  Solving Initial Value Problem", verb=2)
        
        # get list as start value
        start = []
        #for xx in self.x_sym:
            #start.append(self.xa[xx])
        for xx in self.x_sym_orig:
            start.append(self.xa_orig[xx])

        log.info("    start: %s"%str(start), verb=2)

        # calulate simulation time
        T = self.b - self.a

        # create simulation object
        #S = Simulation(self.ff, T, start, self.u)
        S = Simulation(self.ff_orig, T, start, self.u)

        # start forward simulation
        self.sim = S.simulate()


    def checkAccuracy(self):
        '''
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance set by self.eps
        '''

        # this is the solution of the simulation
        xt = self.sim[1]

        # what is the error
        log.info(40*"-", verb=3)
        log.info("Ending up with:   Should Be:  Difference:", verb=3)

        err = np.empty(self.n)
        #for i, xx in enumerate(self.x_sym):
        #    err[i] = abs(self.xb[xx] - xt[-1][i])
        #    log.info(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.xb[xx], err[i]), verb=3)
        for i, xx in enumerate(self.x_sym_orig):
            err[i] = abs(self.xb_orig[xx] - xt[-1][i])
            log.info(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.xb_orig[xx], err[i]), verb=3)
        
        log.info(40*"-", verb=3)
        
        if self.mparam['ierr']:
            # calculate the error functions H_i(t)
            H = dict()
            
            error = []
            for t in self.sim[0]:
                xe = self.x(t)
                ue = self.u(t)
            
                ffe = self.ff(xe, ue)
                dxe = self.dx(t)
                
                error.append(ffe - dxe)
            error = np.array(error)
            
            for i in self.eqind:
                H[i] = error[:,i]
            
            maxH = 0
            for arr in H.values():
                maxH = max(maxH, max(abs(arr)))
            
            self.reached_accuracy = maxH < self.mparam['ierr']
            log.info('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            self.reached_accuracy = max(err) < self.mparam['eps']
        
        #self.maxerr.append((self.mparam['sx'], max(err)))
        
        log.info("  --> reached desired accuracy: "+str(self.reached_accuracy), verb=1)


    def x(self, t):
        '''
        Returns the current system state.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the system at.
        '''
        
        if not self.a <= t <= self.b:
            log.warn("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.x_fnc[xx](t) for xx in self.x_sym])
        
        return arr


    def u(self, t):
        '''
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        '''
        
        if not self.a <= t <= self.b+0.05:
            log.warn("Time point 't' has to be in (a,b)")
            arr = None
            arr = np.array([self.u_fnc[uu](self.b) for uu in self.u_sym])
        else:
            arr = np.array([self.u_fnc[uu](t) for uu in self.u_sym])
        
        return arr


    def dx(self, t):
        '''
        Returns the state of the 1st derivatives of the system variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        '''
        
        if not self.a <= t <= self.b+0.05:
            log.warn("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.dx_fnc[xx](t) for xx in self.x_sym])
        
        return arr


    def plot(self):
        '''
        Plot the calculated trajectories and error functions.

        This method calculates the error functions and then calls
        the :func:`utilities.plot` function.
        '''

        try:
            import matplotlib
        except ImportError:
            log.error('Matplotlib is not available for plotting.')
            return

        # calculate the error functions H_i(t)
        H = dict()

        error = []
        for t in self.sim[0]:
            xe = self.x(t)
            ue = self.u(t)

            ffe = self.ff(xe, ue)
            dxe = self.dx(t)

            error.append(ffe - dxe)
        error = np.array(error)

        for i in self.eqind:
            H[i] = error[:,i]

        # call utilities.plotsim()
        #plotsim(self.sim, H)
        t = self.sim[0]
        xt = np.array([self.x(tt) for tt in t])
        ut = self.sim[2]
        plotsim([t,xt,ut], H)


    def save(self, fname=None):
        '''
        Save system data, callable solution functions and simulation results.
        '''

        save = dict()

        # system data
        #save['ff_sym'] = self.ff_sym
        #save['ff'] = self.ff
        #save['a'] = self.a
        #save['b'] = self.b

        # boundary values
        #save['xa'] = self.xa
        #save['xb'] = self.xb
        #save['uab'] = self.uab

        # solution functions       
        #save['x'] = self.x
        #save['u'] = self.u
        #save['dx'] = self.dx

        # simulation resutls
        save['sim'] = self.sim
        
        if not fname:
            fname = __file__.split('.')[0] + '.pkl'
        elif not fname.endswith('.pkl'):
            fname += '.pkl'
        
        with open(fname, 'wb') as dumpfile:
            pickle.dump(save, dumpfile)


    def clear(self):
        '''
        This method is intended to delete some attributes of the object that
        are no longer neccessary after the iteration has finished.

        TODO: implement this ;-P
        '''

        del self.Mx, self.Mx_abs, self.Mu, self.Mu_abs, self.Mdx, self.Mdx_abs
        del self.Df, self.DXU, self.DdX
        del self.c_list

        return



if __name__ == '__main__':
    from sympy import cos, sin, exp
    from numpy import pi

    # partially linearised inverted pendulum

    #~ def f(x,u):
        #~ x1,x2,x3,x4 = x
        #~ u1, = u
        #~ l = 0.5
        #~ g = 9.81
        #~ ff = np.array([     x2,
                            #~ u1,
                            #~ x4,
                        #~ (1/l)*(g*sin(x3)+u1*cos(x3))])
        #~ return ff
    #~ 
    #~ xa = [0.0, 0.0, pi, 0.0]
    #~ xb = [0.0, 0.0, 0.0, 0.0]
    
    def f(x,u):
        x1, x2 = x
        u1, = u
        
        ff = np.array([ x2,
                        u1])
        return ff
    
    xa = [0.0, 0.0]
    xb = [1.0, 0.0]
    
    a = 0.0
    b = 2.0
    sx = 5
    su = 5
    kx = 3
    maxIt  = 5
    g = [0,0]
    eps = 0.01
    use_chains = False
    
    # NEW
    #constraints = {0:[-0.9, 0.4]}
    constraints = {1:[-0.1, 0.65]}
    #constraints = dict()

    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx,
                    maxIt=maxIt, g=g, eps=eps, use_chains=use_chains, constraints=constraints)
    
    T.setParam('ierr', 1e-2)
    #T.setParam('method', 'new')
    
    with log.Timer("Iteration", verb=0):
        T.startIteration()
    
    if DEBUG:
        IPS()
