"""
This python script solves Laplace Tidal equations including the latitudinal and and vertcial structure equations
The latitudinal structure is presented by Hough functions (eigenvectors) associated with various Hough modes (eigenvalues)
The list of functions includes:
    solve_hough_cheby    : solve hough functions using Chebyshev collection method
    solve_hough_alp      : solve hough functions using associated Legendre polynomials
    solve_tide_v         : solve the vertical structure equation using second-order central differential method
    cheby_boyd           : compute differential matrix for Chebyshev collocation methd
    lgwt                 : computing definite integrals using Legendre-Gauss quadrature
    central_diff         : Central Difference Gradient for unevenly spaced univariate data (not the same as that in solve_tide_v)
    pmn_polynomial_value : normalized Legendre polynomial Pmn(n,m,x), where n >= m

Input (for Laplace Tidal Equations):
    (for latitudinal and vertical structure equations)
    sn         : zonal wavenumbder [number of waves]
    nu         : wave frequency (omega) normalized by the rotation rate (omg_planet) nu = omega/(2*omg_planet)
    n_mode     : number of wave modes (sum of which gives the solution to perturbations)
    omg_planet : rotation rate of the planet [s^{-1}]
    radius     : radius of the planet [m]
    gravity    : gravity [ms^{-2}]

    (for vertical structure equation only)
    smdiv      : dimensionless divergence damping coefficient
    rgas       : gas constant [J kg^{-1} K^{-1}]
    p0         : reference pressure (surface pressure) [Pa]
    T0         : reference temperature at surface [K]
    dT0        : maximum amplitude of near-surface temperature variation (see REMS)
    dts        : model timestep or RK sub-timestep [s]
    res        : model horizontal grid resolution [deg]
    kdiff      : eddy diffusivity in PBL [m^2s^{-1}]
    lambda_x   : longitude [rad]
    heq        : equivalent depth [m]
    dz         : vertical grid resolution [m]
    ztop       : model top (PBL top) [m]
    ubc        : upper boundary condition [zero vertical velocity or radiation BC]

Output (for Laplace Tidal Equations):
    (for latitudinal structure)
    heq        : equivalent depth [m]
    mu         : sin(theta), where theta is latitude in radians
    hough      : hough functions (eigenvectors)

    (for vertical structure)
    x          : solutions (pressure perturbation P' = p'/rho) of the vertical structure equation
    z          : altitude [m]

Examples (for Mars):

    Latitudinal structure:
    sn=1; nu=0.5 # Diurnal
    sn=2; nu=1.  # Semidiurnal
    sn=3; nu=1.5 # Terdiurnal
    n_mode=62; omg_planet=7.27e-5; radius = 3389.e3; gravity=3.727
    heq, mu, hough = solve_hough_cheby(sn,nu,n_mode,omg_planet,radius,gravity)

    Vertical structure:
    omg_planet=7.27e-5; radius = 3389.e3; gravity=3.727; rgas=191.8
    p0=610.; T0=300.; dT0=40.; kdiff=0.1
    dts=40. #RK time step=dyn/3;
    res=4.;
    # the following dz and ztop combo are for the case smdiv=0.1 (ztop needs to be sufficiently large to satisfy upper BC)
    dz=0.02 ; ztop=400. # Diurnal: dz=0.02, 0.03, 0.04 -- same solution, also not affected by increasing ztop
    dz=0.015; ztop=300. # Semidirnal (ztop increase won't significantly affect the solution between 0-300m)
    dz=0.01 ; ztop=200. # Terdiurnal (ztop increase won't significantly affect the solution between 0-200m)

    smdiv=0.1; lambda_x=0.; ubc=1; lbc=0
    x, zc, dz = solve_tide_v(smdiv,nu,omg_planet,radius,gravity,rgas,p0,T0,dT0,dts,res,kdiff,lambda_x,heq,dz,ztop,ubc,lbc)

    (for Earth)
    sn=1; nu=0.5
    n_mode=62; omg_planet=7.27e-5; radius = 6378.e3; gravity=9.81
    p0=1.e5; T0=300.; kdiff=20.
    dts=120.; res=4.; dz=10.; ztop=5.e3
    smdiv=0.1; lambda_x=0.; ubc=1

"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.linalg import eig
from scipy.linalg import pinv
import matplotlib.pyplot as plt


def solve_hough_cheby(sn,nu,n_mode,omg_planet,radius,gravity):
    # solve hough functions using the Chebyshev collection method (Wang et al., 2016)
    # sn: horizontal wavenumber (number of waves)
    # nu: wave frequency (omega) normalized by the rotation rate (omg_planet) nu = omega/(2*omg_planet)
    # n_mod: number of wave modes (sum of which gives the solution to perturbations)
    # omeg_planet: rotation rate of the planet [s^{-1}]
    # radius: radius of the planet [m]
    # gravity: gravity of the planet [m s^{-2}]

    # sn = 1.0; nu = 0.5 (diurnal tide)
    # sn = 2.0; nu = 1.0 (semidiurnal tide)
    # sn = 3.0; nu = 1.5 (terdirunal tide)

    parity_factor = np.mod(sn,2)
    N = n_mode
    D1,D2,mu = cheb_boyd(N,parity_factor)
    a2 = (1.-mu**2)/(nu**2-mu**2)
    a1 = 2.*mu*(1.-nu**2)/(nu**2-mu**2)**2
    a0 = -1./(nu**2-mu**2)*((sn/nu)*(nu**2+mu**2)/(nu**2-mu**2)+sn**2/(1.-mu**2))

    A = np.dot(np.diag(a2),D2) + np.dot(np.diag(a1),D1) + np.diag(a0)
    # calculate eigenvalues d and eigenvectors v of matrix A
    d,v = eig(A) # in matlab, it is v,d=eig(A). Also, d_matlab=np.diag(d_python)
    lamb = np.real(d)
    idx = lamb.argsort()[::-1] # sort eigenvalues from largest to smallest
    lamb=lamb[idx]
    hough=np.real(v[:,idx])
    heq = -4.*radius**2*omg_planet**2/gravity/lamb; # equivalent depth in [m]
    b1 = (nu**2-mu**2)*np.sqrt(1.-mu**2)
    b2 = np.sqrt(1.-mu**2)/(nu**2-mu**2)
    hough_u = np.dot(np.diag(sn/b1),hough) - np.dot(np.dot(np.diag(b2*mu/nu),D1),hough)
    hough_v = np.dot(np.diag((sn/nu)*mu/b1),hough) - np.dot(np.dot(np.diag(b2),D1),hough)

    return heq, mu, hough

def solve_hough_alp(sn,nu,n_mode,omg_planet,radius,gravity):
    # NALP_HOUGH - Compute Hough functions
    # using normalized associated Legendre
    # polynomials (ALP)
    # sn = 1.d0; nu = 0.4986348375d0; % DW1
    #sn = 1.; nu = 0.5; #DW1
    #sn = 2.; nu = 1.0; # SW2 
    #sn = 3.; nu = 1.5; # TW3
    N = n_mode; N2 = int(N/2); sf = sn/nu;
    # define L(r) and M(r)
    L = np.zeros([N])
    M = np.zeros([N])
    for r in range(sn,N+sn):
        i = r-sn
        # define L(r)
        L[i] = np.sqrt((r+sn+1)*(r+sn+2)*(r-sn+1)*(r-sn+2)) \
              /((2*r+3)*np.sqrt((2*r+1)*(2*r+5))*(sf-(r+1)*(r+2)));
        # define M(r)
        if ((sn == 2) and (r == 2)):
           M[i]=-(nu**2.*(sf-r*(r+1.)))/((r*(r+1.))**2) \
                +(r+2.)**2*(r+sn+1.)*(r-sn+1.)/((r+1.)**2*(2*r+3.)*(2*r+1.) \
                *(sf-(r+1.)*(r+2.)))
        else:
           M[i]=-(nu**2*(sf-r*(r+1.)))/((r*(r+1.))**2) \
                +(r+2.)**2*(r+sn+1.)*(r-sn+1.)/((r+1.)**2*(2*r+3.)*(2*r+1.) \
                *(sf-(r+1.)*(r+2.)))+(r-1.)**2*(r**2-sn**2)*np.float64(1.)/(r**2*(4*r**2-1.)*(sf-r*(r-1.)))
        if (M[i] == np.inf):
           M[i] = np.finfo(float).max

    # build F1 & F2 matix
    f1 = np.zeros([N2,N2])
    f2 = np.zeros([N2,N2])
    for i in range(N2):
        f1[i,i] = M[2*i]
        f2[i,i] = M[2*i+1]
        if (i+1 <= N2-1):
           f1[i,i+1] = L[2*i];
           f1[i+1,i] = L[2*i];
           f2[i,i+1] = L[2*i+1];
           f2[i+1,i] = L[2*i+1];
    # symmetric modes
    d1,v1 = eig(f1); lamb1 = d1
    idx = lamb1.argsort()[::-1]
    lamb1=lamb1[idx]; v1 = v1[:,idx]
    heq1 = 4.*radius**2*omg_planet**2/gravity*lamb1
    # antisymmetric modes
    d2,v2 = eig(f2); lamb2 = d2
    idx = lamb2.argsort()[::-1]
    lamb2 = lamb2[idx]; v2 = v2[:,idx];
    heq2 = 4.*radius**2*omg_planet**2/gravity*lamb2
    # Legendre-Gauss quadrature points
    nlat = 184
    mu,w = lgwt(nlat,-1,1)
    # normalized associated Legendre functions
    prs = pmn_polynomial_value(nlat,N+sn,sn,mu); # compute Hough modes
    h1 = np.zeros([nlat,N2])
    h2 = np.zeros([nlat,N2])
    for i in range(N2):
        for j in range(N2):
            i1 = 2*j+sn; i2 = 2*j+sn+1;
            for ii in range(nlat):
                # symmetric modes
                h1[ii,i] = h1[ii,i] + v1[j,i]*prs[ii,i1]
                # anti-symmetric modes
                h2[ii,i] = h2[ii,i] + v2[j,i]*prs[ii,i2]

    # put them together
    lamb = np.zeros([N]); hough = np.zeros([nlat,N]);
    for i in range(N2):
        for j in range(nlat):
            i1 = 2*i; i2 = 2*i+1;
            lamb[i1] = lamb1[i]
            lamb[i2] = lamb2[i]
            hough[j,i1] = h1[j,i]
            hough[j,i2] = h2[j,i]
    idx = lamb.argsort()[::-1]
    lamb = lamb[idx]; hough = hough[:,idx]
    # equivalent depth (m)
    heq = 4.*radius**2*omg_planet**2/gravity*lamb

    # reverse the sign of the eigenvector (or eigenfunction) hough to better match chebyshev method

    # there is not a sign error. Both positive and negative solutions of eigenvectors for a given eigenvalue
    # are valid given the boundary condition where hough=0 at mu=[-1,1]. The choice of sign is determined by 
    # the function eig that calculates the eigenvalues and eigenvectors.

    hough=-hough
    # compute Hough functions for wind components
    b1 = (nu**2-mu**2)*np.sqrt(1.-mu**2)
    b2 = np.sqrt(1.-mu**2)/(nu**2-mu**2)
    dhdx = central_diff(hough,mu);
    hough_u = np.dot(np.diag(sn/b1),hough) - np.dot(np.diag(b2*mu/nu),dhdx)
    hough_v = np.dot(np.diag((sn/nu)*mu/b1),hough) - np.dot(np.diag(b2),dhdx)
    #for j in range(60):
    #    u = hough[:,j]; plt.subplot(10,6,j)
    #    plt.plot(mu, u,'LineWidth',2)

    return heq, mu, hough


def solve_tide_v(smdiv,nu,omg_planet,radius,gravity,rgas,p0,T0,dT0,dts,res,kdiff,lambda_x,heq,dz0,ztop,ubc,lbc):

    # This function solves the second-order ODE w.r.t. y_n = L_n*sqrt(rho)

    # smdiv:      [0.1    ]  dimensionless divergence damping coefficient
    # nu:         [1.0    ]  wave frequency (omega) normalized by the rotation rate (omg_planet) nu = omega/(2*omg_planet)
    # omg_planet: [7.27e-5]  rotation rate of the planet [s^{-1}]
    # radius:     [3389.e3]  radius of the planet [m]
    # gravity:    [3.727  ]  gravity of the planet [ms^{-2}]
    # rgas:       [191.8  ]  gas constant of the atmosphere [Jkg^{-1}K^{-1}]
    # p0:         [610.0  ]  reference pressure (surface pressure) [Pa]
    # T0:         [210.0  ]  reference temperature (surface temperature) [K]
    # heq:        [8.e3   ]  equivalent depth in [m]
    # ztop:       [10.e3  ]  model top (PBL top) in [m]    
    # dz:         [100.   ]  layer thickness in [m]
    # dts:        [120./3.]  GCM runge-kutta sub time step = dt_dynamics/number_of_sub_steps [s] (see MPAS model namelist)
    # res:        [4.     ]  GCM grid resolution in [deg]
    # kdiff:      [0.1    ]  eddy diffusivity [m^2 s^{-1}]. For Mars kdiff=0.1 (Martinez et al., 2009)
    # lambda_x:   [0.     ]  longitude [rad]
    # ubc:        [0/1    ]  flag for upper boundary condition: 0, w'=0; 1, radiation BC

    # omega:      wave frequency = nu*2*omg_planet


    # ===== define some constants ======= #
    #s0=1380. # solar constant
    #ra=1.5   # distance between Mars and Sun
    #rgas=8314./43.34
    cp=7./2.*rgas
    cv=5./2.*rgas
    cpocv=cp/cv
    eta=cpocv/(1.-cpocv)

    # ======= define vertical grid ====== #
    sfac=1.05
    na=200
    dza=np.zeros(na)
    for i in range(na):
        dza[i]=dz0/sfac**(na-i)

    nz=int(ztop/dz0)
    dz=np.zeros(nz)
    if (nz>na):
       dz[0:na]=dza
       dz[na:nz]=dz0
    else:
        raise RuntimeError('model layer number nz needs to be large than the number of fine-res layers')
  
    dzc=(dz[1:]+dz[:-1])/2.

    ze=np.zeros(nz+1)
    for i in range(nz):
        ze[i+1]=ze[i]+dz[i]

    zc=(ze[1:]+ze[:-1])/2.
    
    # ======= tunable parameters ======== #
    #p0=610. # mean surface pressure
    #T0=220. # surface temperature
    H0=rgas*T0/gravity
    # wave frequency
    omega = nu*2.*omg_planet

    # ====== calculate divergence damping coefficient ======= #
    ##cs2=1.4*rgas*T0 # sound speed squared or just use phase speed of thermal tides
    ##tau=2.*np.pi/omega # model time step or wave period as time scale?
    ##alpha_d=simdiv*cs2*tau # divergence damping coefficient on pp 2118, Skamarock and Klemp 1992
    len_disp=radius*res/180.*np.pi
    alpha_d=2.*smdiv*len_disp**2./dts # see coef_divdamp in mpas_atm_time_integration.F
    # calculate the tidal heating source term (Eq. 140, Atmospheric Tides, pp125) 
    # kdiff=0.1 # m^2s^{-1} ( for Mars, Martinez et al., 2009)
    # lambda_x = 0. # lambda_x is the longitude in [rad] (0-2*pi)
    kd=np.sqrt(abs(omega)/kdiff)*np.exp(lambda_x*1j/4.)
    Gn=1j*abs(omega)*cp*dT0*np.exp(-kd*zc) # kd is the vertical wavenumber derived from diffusion equation
    # ignore the Gn vertical profile since kd~1./kdiff where kdiff is very small for Mars
    #Gn[1:]=0.
    dGndz=-kd*Gn

    p=p0*np.exp(-zc/H0)
    rho=p/(rgas*T0)

    epsn=Gn*np.sqrt(rho)
    depsndz=np.sqrt(rho)*(dGndz-1./2.*Gn/H0)
    
    # initialize coefficient arrary for the vertical structure equation
    aa=np.zeros([nz,nz])*1.0j
    bb=np.zeros([nz])*1.0j

    # lower BC:
    k=0
    if (lbc==0): # w'=0
       aa[k,k  ] = 1./H0*(1./2.+1./eta)/2.-1./dz[k]
       aa[k,k+1] = 1./H0*(1./2.+1./eta)/2.+1./dz[k]
       bb[k    ] = 1.0j*epsn[k]/(omega*eta*H0)
    else:        # p'=10Pa
       aa[k,k]=1.
       bb[k  ]= 30./np.sqrt(rho[0])

    # calculate coefficients for the second-order ODE
    a1=1j*alpha_d*omega/(cpocv*gravity*heq*H0)
    a2=1j*alpha_d*omega/(cpocv*gravity*heq*2.*H0**2.) - 1./(4.*H0**2.) - 1./(eta*heq*H0)
    # k = 1, nz
    for k in range(1,nz-1):
        aa[k,k-1] =  1./dz[k]**2.-a1/(2.*dz[k])
        aa[k,k  ] = -2./dz[k]**2.+a2
        aa[k,k+1] =  1./dz[k]**2.+a1/(2.*dz[k])
        bb[k    ] = 1.0j/(eta*omega*H0)*(depsndz[k]+epsn[k]/(2.*H0)) \
                  -(1.0j/omega+alpha_d/(gravity*heq))*epsn[k]/(eta*H0**2.)

    # upper BC
    k=nz-1
    c1=a1/1j
    c2=-a2
    if (ubc==0): # w'=0
        aa[k,k-1] = 1./H0*(1./2.+1./eta)/2.-1./dz[k]
        aa[k,k  ] = 1./H0*(1./2.+1./eta)/2.+1./dz[k]
        bb[k    ] = 1.0j*epsn[k]/(omega*eta*H0)
    else: # radiation BC
        kzu= (-c1+np.sqrt(c1**2.-4.*c2))/2.
        aa[k,k-1] = - (1./dz[k-1] + 1j*kzu/2.)
        aa[k,k  ] =    1./dz[k-1] - 1j*kzu/2.
        bb[k    ] = 0.0
    
    # storage of banded matrix (tridiagnoal)
    
    ld=1; ud=1;
    
    # ab[u + i - j, j] == a[i,j]
    
    ndiag=ld+ud+1
    
    ab=np.zeros([ndiag,nz])*1.0j
    
    # array indices start from 1 to N
    # then -1 to move indices from 0 to N-1
    for icol in range(1,nz+1):
        i1 = max (1,  icol - ud )
        i2 = min (nz, icol + ld )
        for irow in range(i1, i2+1):
            irowb = irow - icol + ndiag-1
            #print('old ', irow,icol,' new ', irowb,icol)
            # -1 so array indices start from 0
            ab[irowb-1,icol-1] = aa[irow-1,icol-1]
    
    # solve the 2nd-oder ODE
    x = solve_banded((1, 1), ab, bb)

    return x, zc, dz

def cheb_boyd(N,pf):
    # compute differential matrix for Chebyshev collocation methd with
    # parity factor (Wang et al., 2016)
    # note that x = cos(phi), see Appendix C in wang et al., 2016
    t  = np.pi/(2.*N)*np.arange(1,2*N,2)
    x  = np.cos(t)
    n  = np.arange(0,N)
    ss = np.sin(t)
    cc = np.cos(t)
    sx = np.tile(ss,(N,1)).transpose() # repeat in columns
    cx = np.tile(cc,(N,1)).transpose() # repeat in columns
    tx = np.tile(t, (N,1)).transpose() # repeat in coloums
    nx = np.tile(n, (N,1))             # repeat in rows
    tn = np.cos(nx*tx)

    if (pf == 0):
       phi2 = tn
       PT = -nx*np.sin(nx*tx)
       phiD2 = -PT/sx
       PTT = -nx**2*tn
       phiDD2 = (sx*PTT-cx*PT)/sx**3
    else:
       phi2 = tn*sx
       PT = -nx*np.sin(nx*tx)*sx+tn*cx
       phiD2 = -PT/sx
       PTT = -nx**2*tn*sx-2.*nx*np.sin(nx*tx)*cx-tn*sx
       phiDD2 = (sx*PTT-cx*PT)/sx**3

    D1 = np.dot(phiD2,  pinv(phi2)) # right division phiD2 /phi2 in matlab
    D2 = np.dot(phiDD2, pinv(phi2)) # right division phiDD2/phi2 in matlab

    return D1, D2, x

def lgwt(N,a,b):
    # lgwt.m
    #
    # This script is for computing definite integrals using Legendre-Gauss 
    # Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
    # [a,b] with truncation order N
    #
    # Suppose you have a continuous function f(x) which is defined on [a,b]
    # which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    # the values contained in the x vector to obtain a vector f. Then compute
    # the definite integral using sum(f.*w);
    #
    # Written by Greg von Winckel - 02/25/2004
    N=N-1;
    N1=N+1; N2=N+2;
    xu=np.linspace(-1,1,N1)
    # Initial guess
    y=np.cos((2*np.arange(N+1)+1.)*np.pi/(2*N+2.))+(0.27/N1)*np.sin(np.pi*xu*N/N2);
    # Legendre-Gauss Vandermonde Matrix
    L=np.zeros([N1,N2])
    # Derivative of LGVM
    Lp=np.zeros([N1])
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0=2.
    # Iterate until new points are uniformly within epsilon of old points
    while (np.max(abs(y-y0))>np.finfo(float).eps):
        L[:,0]=1.
        L[:,1]=y
        
        for k in range(2,N1+1):
            L[:,k]=( (2*k-1.)*y*L[:,k-1]-(k-1)*L[:,k-2] )/np.float(k)
     
        Lp=N2*( L[:,N1-1]-y*L[:,N2-1] )/(1.-y**2)
        
        y0=y
        y=y0-L[:,N2-1]/Lp
        
    # Linear map from[-1,1] to [a,b]
    x=(a*(1.-y)+b*(1.+y))/2.
    # Compute the weights
    w=(b-a)/((1.-y**2)*Lp**2)*(N2/N1)**2

    idx = x.argsort()[::-1]
    x=x[idx]
    w=w[idx]

    return x,w


def central_diff(F, x):
    # Central Difference Gradient for unevenly spaced univariate data
    # in the interior and second-order forward/backward differences 
    # at the left/right ends
    #
    # usage:    gradient = central_diff( F, x )
    #
    # inputs:   F - Values of a function evaluated at x 
    #               to be differentiated with respect to x
    #               (matrix: number of rows = length of x, unless x is scalar)
    #           x - Monotonically increasing coordinate values 
    #               where F is evaluated (vector, length=number rows of F),
    #               or dx spacing for evenly spaced coordinates (scalar)
    #           F[M,N], X[M]
    #
    # output:  gradient - numerically evaluated gradient by:
    #                     forward difference at the left end;
    #                     backward difference at the right end;
    #                     central difference in the interior
    #                     (matrix, same size as F)
    #
    #  Written by:   Robert A. Canfield
    #  email:        bob.canfield@vt.edu
    #  Version:      2.0
    #
    #  Created:      10/19/00
    #  Modified:     10/01/15
    #
    #  Description: The central_diff function calculates a numeric gradient
    #  using second-order accurate difference formula for evenly or unevenly
    #  spaced coordinate data. It operates in a similar fashion to the MATLAB
    #  function, gradient, except that it permits only one independent
    #  variable, x, and correctly handles unevenly spaced values of the
    #  x-coordinate data. Accuracy is increased at the ends relative to the
    #  MATLAB gradient function, which uses only first-order forward or
    #  backward differences at the ends, by instead using second-order forward
    #  difference at the left end and second-order backward difference at the
    #  right end.
    #  MATLAB's gradient function is incorrect for unevenly spaced coordinates.
    #  This central_diff function uses the correct formula.
    #  Tested under MATLAB versions 5.2, 5.3.1, and 8.3.
    #  (logical operators & and | replaced with && and || for 8.3)
    #
    # Alternatively, you may patch MATLAB's gradient function 
    # to make unevenly spaced interior points second-order accurate
    # (leaving left and right ends first-order accurate)
    # by replacing the following lines...
    #
    #>  # Take centered differences on interior points
    #>  if n > 2
    #>     h = h(3:n) - h(1:n-2);
    #>     g(2:n-1,:) = (f(3:n,:)-f(1:n-2,:))./h(:,ones(p,1));
    #>  end
    #
    # with...
    #   # Take centered differences on interior points
    #   if n > 2
    #      if all(abs(diff(h,2)) < eps) # only use for uniform h (RAC)
    #         h = h(3:n) - h(1:n-2);
    #         g(2:n-1,:) = (f(3:n,:)-f(1:n-2,:))./h(:,ones(p,1));
    #      else   # new formula for un-evenly spaced coordinates (RAC)
    #         h = diff(h); h_i=h(1:end-1,ones(p,1)); h_ip1=h(2:end,ones(p,1));
    #         g(2:n-1,:) =  (-(h_ip1./h_i).*f(1:n-2,:) + ...
    #                         (h_i./h_ip1).*f(3:n,:)   )./ (h_i + h_ip1) + ...
    #                         ( 1./h_i - 1./h_ip1 ).*f(2:n-1,:);
    #      end
    #   end
    #--Modifications
    #  10/23/00 
    #  10/01/01 - Copyright (c) 2001 Robert A. Canfield (BSD License)
    #  10/01/15 - Second-order accurate at left and right ends
    ## Ensure compatible vectors and x monotonically increasing or decreasing
    m = len(x);
    n,p = np.shape(F)
    Fx = np.zeros([n,p])
    # Forward difference at left end, and Backward difference at right end
    if (m>1):
       H = x[1] - x[0]
    else:
       H = x

    if (n==2): # First-order difference for a single interval with end values
       Fx[0,:] = (F[1,:] - F[0,:]) / H;
       Fx[1,:] = Fx[0,:]
    else:      # Second-order differences
       # Left end forward difference
       if ((m==1) or abs(np.diff(x[1:3])-H)<=np.finfo(float).eps): # evenly spaced
          Fx[0,:] = np.dot( [-3, 4, -1]/(2*H), F[0:3,:] )
       else:                              # unevenly spaced
          h   = np.diff(x[0:3])
          hph = np.sum(h); # h_1 + h+2
          Fx[0,:] = hph/h[0]/h[1]*F[1,:] - ((2*h[0]+h[1])/h[0]*F[0,:] + h[0]/h[1]*F[2,:])/hph;
       # Right end backward difference
       if (m==1 or abs(np.diff(np.diff(x[m-3:m])))<np.finfo(float).eps): # evenly spaced
          Fx[-1,:] = np.dot( [1, -4, 3]/(2*H), F[m-3:m,:] )
       else:                                   # unevenly spaced
          h   = np.diff(x[m-3:m])
          hph = np.sum(h)
          Fx[-1,:] = ( h[1]/h[0]*F[-3,:] + (h[0]+2*h[1])/h[1]*F[-1,:] )/hph - hph/h[0]/h[1]*F[-2,:]
    # Central Difference in interior (second-order)
    if (n > 2):
       if (m==1 or np.all(abs(np.diff(x)-H)<=np.finfo(float).eps)):
          # Evenly spaced formula used in MATLAB's gradient routine
          Fx[1:n-1] = ( F[2:n,:] - F[0:n-2,:] ) / (2*H);
       else:
          # Unevenly spaced central difference formula
          h    = np.diff(x)
          h_i  = np.tile(h[0:m-2],(p,1)).transpose()
          h_ip1= np.tile(h[1:m-1],(p,1)).transpose()
          Fx[1:n-1,:] =  (-(h_ip1/h_i)*F[0:n-2,:] + (h_i/h_ip1)*F[2:n,:] )/(h_i + h_ip1) + \
                           ( 1./h_i - 1./h_ip1 )*F[1:n-1,:]

    return Fx


def pmn_polynomial_value ( mm, n, m, x ):

    #*****************************************************************************80
    #
    ## PMN_POLYNOMIAL_VALUE: normalized Legendre polynomial Pmn(n,m,x), where n >= m
    #
    #  Discussion:
    #
    #    The unnormalized associated Legendre functions P_N^M(X) have
    #    the property that
    #
    #      Integral ( -1 <= X <= 1 ) ( P_N^M(X) )^2 dX 
    #      = 2 * ( N + M )! / ( ( 2 * N + 1 ) * ( N - M )! )
    #
    #    By dividing the function by the square root of this term,
    #    the normalized associated Legendre functions have norm 1.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license. 
    #
    #  Modified:
    #
    #    12 January 2021
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Reference:
    #
    #    Milton Abramowitz, Irene Stegun,
    #    Handbook of Mathematical Functions,
    #    National Bureau of Standards, 1964,
    #    ISBN: 0-486-61272-4,
    #    LC: QA47.A34.
    #
    #  Input:
    #
    #    integer MM, the number of evaluation points.
    #
    #    integer N, the maximum first index of the Legendre
    #    function, which must be at least 0.
    #
    #    integer M, the second index of the Legendre function,
    #    which must be at least 0, and no greater than N.
    #
    #    real X(MM,1), the evaluation points.
    #
    #  Output:
    #
    #    real CX(MM,N+1), the function values.
    #
    cx = np.zeros([mm, n+1]);
    
    if ( m <= n ):
      cx[0:mm,m] = 1.0; 
      factor = 1.0;
      for j in range(m):
        cx[0:mm,m] = - factor * cx[0:mm,m] * np.sqrt ( 1.0 - x[0:mm]**2 )
        factor = factor + 2.0
    else:
        raise ValueError('n needs to be greater than m')

    if ( m + 1 <= n ):
      cx[0:mm,m+1] = ( 2 * m + 1 ) * x[0:mm] * cx[0:mm,m];
    
    for j in range(m + 2, n+1):
      cx[0:mm,j] = ( ( 2 * j     - 1 ) * x[0:mm] * cx[0:mm,j-1]   \
                   + (   - j - m + 1 ) *           cx[0:mm,j-2] ) \
                   / (     j - m     )
    #
    #  Normalization.
    #
    for j in range( m , n+1):
      factor = np.sqrt ( ( ( 2 * j + 1 ) * np.math.factorial ( j - m ) ) \
        / ( 2.0 * np.math.factorial ( j + m ) ) )
      cx[0:mm,j] = cx[0:mm,j] * factor

    return cx
     
    
