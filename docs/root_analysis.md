## 1. Curve fitting problem

### 1.1 One-Sided

We have values x_i, f(x_i), where...
- f represents a shifted function such that f(0) = 0.  
- values x_i are chosen to be in [2/3, 4/3] and include [2/3, 1.0, 4/3]
- we assume f(x_i) >= 0, or at least generally trend upwards

We want to fit a curve g(x) to the provided data points such that we can analyse
root properties.  g(x) will have 3 degrees of freedom a,b,c:

g(x) = a*(b + (1-b)*(x^c))

Where we can calculate that...
- g(0) = a*b
- g(1) = a
- g(2) = a*(b + (1-b)*(2^c))

Going the other way (computing a,b,c from g(0), g(1), g(2)), we get
- a = g(1)
- b = g(0) / g(1)
- c = log2((g(2) - g(0))/(a*(1-b)))

### 1.2. Two-Sided

Given a function f(x) with f(0)=0, we can analyse the local properties of 
around the root at the left side and the right side of the root by solving the
**one-sided** curve fitting problem **twice**:
- once by considering f_left(x) = |f(-x)|, leading to a_l, b_l, c_l
- once by considering f_right(x) = |f(x)|, leading to a_r, b_r, c_r
In both cases we'd only consider positive x_i in [2/3, 4/3]

### 1.3. Cost function

For robustness reasons, we'll use the L1-like Mean-Absolute-Deviation (MAD)
to determine goodness-of-fit.  This should make us less prone to outliers.

We'll call this L(a,b,c)

### 1.4. Uncertainty bounds

Since the provided data is expected to be noisy to some degree (could be very small
or ery significant amount), we want to estimate bounds on the determined parameters.

Generally speaking, we'll approach this as follows:
- determine optimal solution and establish 'L_opt'
- we'll compute 'L_threshold' as e.g. L_threshold = 2*'L_opt'
- determine the parameter ranges (regions in 3D parameter space) that have L(a,b,c) <= L_threshold
- this should give us an idea of the parameter ranges [a_min, a_max] etc...

## 2. Solution approach

### 2.1. 3-point solution

We assume we have 3 data points (which could be based on robust aggregates 
of a larger set of points, such as medians or quantiles) as follows:
- f(x0)
- f(1)
- f(x1)
with 0 < x0 < 1 < x1

Now let's see if we can find a,b,c analytically such that g(x) goes through these 3 points.
As a reminder:

g(x) = a*(b + (1-b)*(x^c))

Then we have:
- f(x0) = a*(b + (1-b)*(x0^c))
- f(1) = a
- f(x1) = a*(b + (1-b)*(x1^c))

#### 1) Determining a

Simple:

    a = f(1)

#### 2) Determining c

We can write:

    (f(x1) - f(1)) / (f(x1) - f(x0)) = ( x1^c - 1) / (x1^c - x0^c) 

For the special case of x0=0.5 and x1=2.0, we get...

    (f(2) - f(1)) / (f(1) - f(0.5)) = ( 2^c - 1) / (1 - 0.5^c)
                                    = ( 2^c - 1) / (1 - 2^-c)
                                    =    (l - 1) / (1 - 1/l) 
                                    =  l*(l - 1) / (l - 1)
                                    =  l
                                    =  2^c     

Hence 

    c = log2((f(2) - f(1)) / (f(1) - f(0.5)))

Note that the above trick can be applied for any x0,x1 as long as x0*x1=1

#### 3) Determining b

From our regular expression

    g(x) = a*(b + (1-b)*(x^c))

we can write:

    (f(x1) - f(x0)) / f(1) = (1-b) * (x1^c - x0^c)

And hence we have

    b = 1 - (f(x1) - f(x0)) / (f(1) * (x1^c - x0^c))


#### Special case: deriving c from a,b

If for some reason we have already determined (or we assume a value for) b,
before compute c, we can more easily compute c:

    (f(x1) - f(1)) / (f(1) * (1-b)) = (x1^c - 1)
hence
    
    x1^c = 1 + (f(x1) - f(1)) / (f(1) * (1-b))

and

    c = log2( 1 + (f(x1) - f(1)) / (f(1) * (1-b)) ) / log2( x1 )

This will find 'c', given 'b', such that g(x1) - g(1) matches f(x1) - f(1).

#### Special case: varying b,c while keeping dgdx(x=1) constant

This means we vary b,c such that the slope at x=1 stays constant.  In other words,
we play with the curvature (2nd order deriv) of the curve such that g(0) = a*b changes,
without changing dgdx(x=1).  This direction of changing the parameters along a curve
is expected to be the most uncertain direction, i.e. the least well-defined direction
based on the provided data.

Let's first determine an expression for dgdx(x=1).  We start from g(x):

g(x) = a*(b + (1-b)*(x^c))

And derive

dg/dx = a*(1-b)*c*x^(c-1)

Which results in 

dg/dx(x=1) = a*(1-b)*c

**SOLUTION**

Assuming a stays constant, we need to keep (1-b)*c constant and = dg/dx(x=1) / a

#### Special case: varying b,c while keeping g(x1) constant

Let's first determine g(x1) for reference parameters a',b',c':

  cte = a'*(b' + (1-b')*(x1^c'))

We will keep a=a' constant as well, so we need to make sure that

b + (1-b)*(x1^c) = (cte/a') 

x1^c = ((cte/a') - b) / (1-b)

and hence

c = log2( ((f(x1)/a') - b) / (1-b) ) / log2(x1)

#### Special case: varying b,c while keeping g(x0) constant

Very similar to previous case

c = log2( ((f(x0)/a') - b) / (1-b) ) / log2(x0)

#### Special case: varying a,c while keeping g(x1) constant

Let's first determine g(x1) for reference parameters a',b',c':

  cte = a'*(b' + (1-b')*(x1^c'))

We will keep b=b' constant as well, so we need to make sure that

a*(b' + (1-b')*(x1^c)) = cte
  hence
a*(1-b')*(x1^c) = cte - (a*b')
  and
x1^c = (cte - (a*b'))/(a*(1-b'))
  and therefore
c = log2(  (cte - (a*b'))/(a*(1-b'))  ) / log2(x1)

#### Special case: varying b, c while keeping g(r) - g(1/r) constant

We could e.g. take r = 2*sqrt(2), if we know that all x-values lie in [1/r, r].

So Let's compute g(r) - g(1/r) for parameters a',b',c'

    g(r) - g(1/r) = a'*(b' + (1-b')*(r^c')) - a'*(b' + (1-b')*((1/r)^c'))
                  = a'*(1-b')*(r^c' - (1/r)^c')

INTERMEZZO:

We can rewrite `r^c' - (1/r)^c'` into something simpler:

    r^c' - (1/r)^c' = r^c' - r^-c'
                    = e^(ln(r)*c') - e^(-ln(r)*c')
                    = 2 * (1/2) * e^(ln(r)*c') - e^(-ln(r)*c')
                    = 2 * sinh(ln(r) * c')

So we get the following expression for `g(r) - g(1/r)`:

    g(r) - g(1/r) = a'*(1-b')*2*sinh(ln(r)*c')

If we now want to vary b,c (keeping a=a' constant) such that g(r) - g(1/r) stays constant, we get:

          a*(1-b)*2*sinh(ln(r)*c)  =      a'*(1-b')*2*sinh(ln(r)*c')
            (1-b)*2*sinh(ln(r)*c)  =         (1-b')*2*sinh(ln(r)*c')
                   sinh(ln(r)*c)   =         (1-b')*2*sinh(ln(r)*c') /     (2*(1-b))
                        ln(r)*c    = asinh(  (1-b')*2*sinh(ln(r)*c') /     (2*(1-b)) )
                              c    = asinh(  (1-b')*2*sinh(ln(r)*c') /     (2*(1-b)) ) / ln(r)
                              c    = asinh(  (1-b') * sinh(ln(r)*c') /        (1-b)  ) / ln(r)
                              c    = asinh(           sinh(ln(r)*c') * (1-b')/(1-b)  ) / ln(r)

In the special case when we choose `r=e` (coincidentally `2*sqrt(2) ~= 2.82 ~= 2.78 ~= e`), we get:

    c   =   asinh( sinh(c') * (1-b')/(1-b) )


#### Special case: varying a, c while keeping g(r) - g(1/r) constant

We'll start from the expression we derived earlier:

    g(r) - g(1/r) = a'*(1-b')*2*sinh(ln(r)*c')

If we now want to vary a,c (keeping b=b' constant) such that g(r) - g(1/r) stays constant, we get:

          a*(1-b)*2*sinh(ln(r)*c)  =         a'*(1-b')*2 * sinh(ln(r)*c')
                  a*sinh(ln(r)*c)  =         a'          * sinh(ln(r)*c')
                    sinh(ln(r)*c)  =        (a'/a)       * sinh(ln(r)*c')
                               c   = asinh( (a'/a)       * sinh(ln(r)*c') ) / ln(r) 

Again, in the special case when we choose `r=e`, we get:

    c   =   asinh( sinh(c') * (a'/a) )


#### Special case: varying a, b while keeping g(r) - g(1/r) constant

We'll start from the expression we derived earlier:

    g(r) - g(1/r) = a'*(1-b')*2*sinh(ln(r)*c')

If we now want to vary a,b (keeping c=c' constant) such that g(r) - g(1/r) stays constant, we get:

          a*(1-b)*2*sinh(ln(r)*c)  =  a'*(1-b')*2 * sinh(ln(r)*c')
                          a*(1-b)  =  a'*(1-b')
                          a        =  a'*(1-b')/(1-b)