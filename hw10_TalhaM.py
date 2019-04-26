import pylab
import scipy.ndimage as nd
from scipy import misc
import mahotas as mh
import matplotlib.pyplot as plt
import Tkinter
import tkFileDialog
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# the function opens and reads a file, number of rows is hard coded to the data
def imageopen():
    root = Tkinter.Tk()
    root.withdraw()
    imgname = tkFileDialog.askopenfilename(title="Select Image")
    with open(imgname,'rb') as fname:
        try:
            objpic = misc.imread(fname)
        except IOError as e:
            print e
            print 'Error in reading file'
        return objpic

def imageanalysis(objectspic):
    obj_g = nd.gaussian_filter(objectspic, 2)
    T = mh.thresholding.otsu(obj_g)
    pylab.imshow(obj_g)
    pylab.show()
    thres = obj_g.mean()
    obj_thres = obj_g > thres
    labeled, nr_objects = nd.label(obj_thres)
    means_val= nd.measurements.center_of_mass(obj_thres,labeled,range(1,nr_objects+1))
    x1=[x[1] for x in means_val]
    y1=[y[0] for y in means_val]
# create figures and display centers of Mass
    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    ax2.imshow(objectspic)
    ax2.scatter(x1, y1, s=10, facecolor='#FF0000', edgecolor='#FF0000')
    # pylab.imshow(fig2)
    pylab.show()
    return fig2

def linfunc(x,a,b):
    return a*x+b

def quadfunc(x,a,b,c):
    return a*x**2+b*x+c

def gaussfunc(x,a,b,c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def analyze_text(filename):
    dfinal1 = pd.read_csv(filename, sep='(\")', header=None, engine='python', usecols=[2, 4])
    dfinal1.columns = ['request', 'r2']
    dfinal1['h1'] = dfinal1.r2.str.split('([0-9][-]+)').str[0]
    dfinal1['http_codes'] = dfinal1.h1.str.split('\s+').str[1]
    dfinal1['bytes'] = dfinal1.h1.str.split('\s+').str[2]
    dfinal1 = dfinal1[['request', 'http_codes', 'bytes']]
    # extract host data and time
    dfinal3 = pd.read_csv(filename, sep='\s+|\b(\\"\+\\"$)\b|\b(\\[\+\\]$)\b|([0-9]+\:[0-9]+:[0-9]+:[0-9]+)',
                          header=None, engine='python', usecols=[0, 7])
    dfinal3.columns = ['host', 'stime']

    result = pd.concat([dfinal1, dfinal3], axis=1, join='inner')
    print type(result)

    ##time series analysis
    # split string for time analysis
    result['day'] = result.stime.str.split(':').str[0]
    result['hour'] = result.stime.str.split(':').str[1]
    result['minute'] = result.stime.str.split(':').str[2]
    result['second'] = result.stime.str.split(':').str[3]
    # add month and year information
    result['year'] = pd.Series('1995', index=result.index)
    result['month'] = pd.Series('8', index=result.index)
    # convert strings to time format
    result['time'] = pd.to_datetime(result[['month', 'day', 'year', 'hour', 'minute', 'second']])
    ## bytes processing
    # convert bytes to numeric
    result['bytes'] = result['bytes'].str.replace('-', 'NaN')
    result['bytes'] = pd.to_numeric(result['bytes'], errors='coerce')
    # split request detail
    result['req_type'] = result.request.str.split('\s').str[0]
    result['file'] = result.request.str.split('\s').str[1]
    result['http'] = result.request.str.split('\s').str[2]
    result_time=result.set_index(['time'])
    return result_time

if __name__ == "__main__":
    #Part 1
    # analysis of the Objects pic
    #filter and show the Objects pic
    pic=imageopen()
    figure=imageanalysis(pic)
    figure.savefig('obspic.pdf', bbox_inches='tight')
    figure.clear()
    #Part 2
    #  Analyze the cars data and develop plots
    cars= pd.read_csv('cars.data.csv', sep=',',header=None)
    cars.columns=['buying','maint','doors','persons','lug_boot','safety','condition']
    print cars.head(n=10)
    # print buy_counts
    buy_c=cars.groupby('buying',as_index=True).count()
    buy_c=buy_c[['maint']]
    buy_c.columns=['Counts']
    print buy_c
    maint_c = cars.groupby('maint', as_index=True).count()
    maint_c = maint_c[['buying']]
    maint_c.columns=['Counts']
    print maint_c
    #safety of cars
    safety_c = cars.groupby('safety',as_index=True).count()
    safety_c = safety_c[['buying']]
    safety_c.columns = ['Counts']
    print safety_c
    doors_c = cars.groupby('doors', as_index=True).count()
    doors_c = doors_c[['buying']]
    doors_c.columns =['Counts']
    print doors_c
    # plots and subplots of cars data
    fig, axes = plt.subplots(nrows=2, ncols=2,sharey=True)
    maint_c.plot(kind='bar',color='b',alpha=0.7, fontsize=8,ax=axes[0,0],legend=False)
    safety_c.plot(kind='bar',color='r',alpha=0.7, fontsize=8,ax=axes[0,1],legend=False)
    doors_c.plot(kind='bar',color='green',alpha=0.7, fontsize=8,ax=axes[1,0],legend=False)
    buy_c.plot(kind='bar',color='y',alpha=0.7, fontsize=8,ax=axes[1,1],legend=False)
    fig.tight_layout()
    fig.suptitle('Cars Data Category Counts', y=1.0)
    plt.show()
    #Part 3
    # Analyze the EPA data
    # serverdata = pd.read_pickle('result_time.pkl')
    serverdata=analyze_text('epa-http.txt')
    print serverdata.head(n=10)
    #plots and subplots of EPA data
    #summarize and calculate the differences
    r_group=serverdata.groupby(['req_type']).get_group('GET')
    r_group_hrly=r_group.resample('1H').count()
    r_group_hrly['req_change']=r_group_hrly['request'].diff(1)
    r_group_hrly=r_group_hrly[['request','req_change']]
    ##plot results
    plt.style.use('ggplot')
    ax=r_group_hrly.plot(secondary_y=['req_change'])
    ax.set_ylabel('Hourly GET Requests')
    ax.right_ax.set_ylabel('Hourly Change in Requests')
    plt.subplots_adjust(bottom=0.1,right=0.85)
    plt.title('Server GET Requests per Hour')
    plt.show()
    # Part 4
    # Brain and Body data
    brain= pd.read_csv('brainandbody.csv', sep=',',header=0)
    print brain
    #convert data frame format to numpy array
    X=brain.as_matrix(columns=['body'])
    Y=brain.as_matrix(columns=['brain'])
    Y=np.reshape(Y,len(Y))
    X=np.reshape(X,len(X))
    #estimate linear and other fits
    popt, pcov =curve_fit(linfunc,X,Y)
    print "Linear Fit"
    print popt
    popt2, pcov2 =curve_fit(quadfunc,X,Y)
    print "Quadratic Fit"
    print popt2
    print "Gaussian Fit"
    popt3, pcov3 =curve_fit(gaussfunc,X,Y)
    print popt3
    #make plots of brain and body data
    ax = brain.plot.scatter(x='body', y='brain', color='DarkBlue',label="data")
    ax.set_ylabel('Brain Weight (grams)')
    ax.set_xlabel('Body Weight (Kg)')
    plt.title('Relationship Between Body & Brain Weight')
    plt.plot(X, linfunc(X, *popt), 'r-', label="linear fit")
    plt.plot(X, quadfunc(X, *popt2),'b--', label="quad fit")
    plt.plot(X, gaussfunc(X, *popt3),'g--', label="gaussian fit")
    plt.legend()
    plt.show()