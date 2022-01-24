"""
This python script calculates the thermal tides either from data or from model.
If from the model, the input is in format of [hour, pressure] that is produced
by insight_pressure.py (multiple lander sites can be analyzed)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sg
import scipy.interpolate as interp

"""

# 1. idealized example
  T=86400.        # diurnal period [second]
  total_time=10*T # 10 days
  omg=2.*np.pi/T  # rotation rate
  
  # need 10 days of data for properly fft diurnal, semdiurnal cycles
  omg1=1.*omg # diurnal tide
  omg2=2.*omg # semidiurnal tide
  omg3=3.*omg # terdiurnal tide
  
  sample_interval=3600.     # [second] -- 1 hour
  N=int(total_time/sample_interval)  # number of records
  
  a1=2; a2=1; a3=0.5
  t=np.linspace(0,total_time,N, endpoint=False)
  # assuming phase is zero for all thermal tides
  y=a1*np.cos(omg1*t)+a2*np.cos(omg2*t)+a3*np.cos(omg3*t)
  
  yf=fft.fft(y)
  
  xf=fft.fftfreq(N, total_time/N)[:N//2] # xf has unit of frequency per second sample_interval=total_time/N
  
  plt.plot(xf*T, 2.0/N * np.abs(yf[0:N//2])) # xf*T--> frequency per T [day]


# 2. real application example

  import thermal_tides_mars as ttm

  f0=open('/Users/lian/python/ysu_test_2mom_limitVt/smdiv0/is_pres.txt')
  sm0=f0.readlines(); f0.close()

  y0=[]
  for i in range(len(sm0)):
      x=[float(s) for s in sm0[i].replace('\n','').split()]
      y0.append(x)

  y0=np.asarray(y0)
  hours=y0[:,0]

  y=y0[:,1]-np.mean(y0[:,1])

  t,yf,xs,xd=ttm.decompose_tides(y,hours)
  # t is in hours relative to the start time at t=0


# 3. real data example

  path='/Volumes/Samsung_T5/Downloads/insight_press/'
  files=[]
  for i in range(10):
      files.append(path+'ps_calib_0'+str(380+i)+'_01.csv')

  ps_all=[] # time in 
  ts_all=[] # time in seconds
  n=0
  for fn in files:
      dat=[]
      with open(fn,newline='') as f:
          reader = csv.reader(f)
          for row in reader:
              dat.append(row)
      dat=np.asarray(dat)
      ltd=dat[1:,2]
      pres=dat[1:,5]
      del dat
      time=[]
      ps=[]
      secs=[]
      for i in range(len(ltd)):
          x=ltd[i].split("M")[1].split(":")
          time.append(x)
          if (pres[i]):
             ps.append(float(pres[i]))
             secs.append(float(time[i][0])*3600.+float(time[i][1])*60.+float(time[i][2]))
      secs=np.asarray(secs)
      dp=ps-np.mean(ps)
      del ltd, ps, pres
      pss=sg.savgol_filter(dp,5001,2)
      pss=pss[::100]
      secs=secs[::100]+n*86400
      ps_all=np.concatenate((ps_all,pss))
      ts_all=np.concatenate((ts_all,secs))

      print(n)

      n=n+1

  # project pressure data to evenly spaced time
  dt=10 # 10 seconds, 100 should be fine too
  t1=np.arange(0,864000,dt) # planet seconds
  f = interp.interp1d(ts_all, ps_all,fill_value="extrapolate")
  p1=f(t1)
  t_is,yf_is,xs_is,xd_is=ttm.decompose_tides(p1,t1/3600,1)

"""

def decompose_tides(y,t,local_time):
    # y: input signal
    # t: time in planet hours
    # local_time: wether to use Earth time or local (planet) time
    # Note that xs to xd conversion has p2si factor in it so p2si=1 or p2si=other doesn't make any difference

    # output:
    # yf: FFT of y, abs(y) is the magnitude of thermal tides
    # xs: frequency [per second, Hz]
    # xd: frequency [per day, 1 is diurnal, 2 is semidiurnal etc]

    if (local_time==1):
        p2si=1.
    else:
        p2si=1.027491252

    T=86400.*p2si
    hr2sec=3600.*p2si
    omg=2.*np.pi/T

    omg1=1.*omg # diurnal tide
    omg2=2.*omg # semidiurnal tide
    omg3=3.*omg # terdiurnal tide
    omg4=4.*omg # quadiurnal tide

    indx_day=np.where(t[:-1]-t[1:]> 6.) # indx_day is float
    indx_day=np.asarray(indx_day).transpose()
    if any(indx_day): # if t is counted every 24 hours
       for i in range(np.size(indx_day)-1):
           day_start=int(indx_day[i])+1
           day_end  =int(indx_day[i+1])+1
           t[day_start:day_end]=t[day_start:day_end]+24.*(i+1)
       if (indx_day[-1] < len(t)):
           day_start=int(indx_day[-1])+1
           t[day_start:]=t[day_start:]+24.*np.size(indx_day)

    total_time=(t[-1]-t[0])*hr2sec # calculate total time in seconds
    sample_interval=(t[1]-t[0])*hr2sec # sample interval in seconds
    N=int(total_time/sample_interval)  # number of records

    yf=fft.fft(y)
    xs=fft.fftfreq(N, sample_interval)[:N//2] # xs has unit of frequency per Earth second sample_interval=total_time/N
    xd=xs*T # xd=xs*T --> frequency per day, 1 is diurnal etc

    plt.figure()
    plt.plot(xd, 2.0/N * np.abs(yf[0:N//2]))

    return t, yf, xs, xd
    

