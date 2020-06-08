import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.stats as sps
from scipy.optimize import curve_fit
import glob
########
# Inputs:
R = 330           # Tungsten Probe Resistance (ohm)
M = 1             # Amplification Parameter
D = 0.000127      # Probe Diameter [m]
L = 0.00843       # Probe Height [m]
########
T_e = []
n_e = []
n_i = []
EEP_F = []
E_f = []
#########
path_dir = r'path_dir'
files = glob.glob(path_dir + '/*.csv')
for f in files:
    print('___________________________________')
    print(f[32:])
    ## Importing the CSV files
    T = pd.read_csv(f, delimiter= ',')
    # Column #1: Time (ms), Column #3: Avg(A), Column #4: Avg(B)
    # time (s)
    limit = np.argmax(np.array(T.iloc[1:,4]).astype(float)) - 200
    time = np.array(T.iloc[1:limit,0]).astype(float)
    # Average Channel A
    avg_A = - np.array(T.iloc[1:limit,3]).astype(float)
    # Average Channel B
    if T.iloc[0,4] == '(mV)':
        avg_B = np.array(T.iloc[1:limit,4]).astype(float) / 1000
    else:
        avg_B = np.array(T.iloc[1:limit,4]).astype(float)
    ##### 
    # Savitzky-Golay filter: polyorder: 2, window length 5001 of avg_B
    avg_Bs = sp.savgol_filter(avg_B, 5001, 2)
    #####
    # Linear fit avg_A
    slope, intercept, r_value, p_value, std_err = sps.linregress(time, avg_A)
    avg_Al = (slope * time) + intercept
    #####
    real_V = (10 * avg_Al) - (M * avg_Bs)
    #####
    real_I = (M * avg_Bs) / R
    Secd_real_I = np.gradient(np.gradient(real_I, real_V), real_V)
    ##### 
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    #### smoothed Second Derivative of real Current
    smsd_real_I = smooth(Secd_real_I, 1001)
    #### 
    b = np.argmax(smsd_real_I)
    #plt.plot(real_V[b:], abs(np.log10(Secd_real_I[b:])))
    from scipy.signal import find_peaks
    peak_proof = (-np.log10(abs(smsd_real_I[b:])))
    mean_peak = np.mean(peak_proof)
    std_peak = np.std(peak_proof)
    peaks = (find_peaks(peak_proof, height= mean_peak + 4*std_peak))
    ##### Plasma Voltage
    plasma_V = real_V[b:][peaks[0][0]]
    #plt.plot(real_V, smsd_real_I)
    ##### Energy  (eV)
    E = plasma_V - real_V
    #####
    e = 1.6021765E-19    # Elementary Electron Charge [C]
    me = 9.10938E-31     # Mass of an Electron
    ##### EEDF (eV^-1)
    #####
    EEDF = e * (np.sqrt((8 * me)) / ((e**3)*np.pi*D*L))* smsd_real_I * np.sqrt(e*E)
    
    NaN = np.argwhere(np.isnan(EEDF))[0][0]
    N = np.argwhere(EEDF[:NaN] < 0)[::-1][0][0]
    E_range = E[:NaN][N:][::-1]+1E-200
    EEDF_range = EEDF[:NaN][N:][::-1]
    
    ####### EEPF (eV^-1.5)
    EEPF = EEDF_range / np.sqrt(E_range)
    ####### EEDF * E
    E_EEDF = EEDF_range * E_range
    ######
    # Electron Density (1/m3)
    ne = np.trapz(EEDF_range, E_range)
    # Electron Temperature (eV)
    Te = (0.66666666666) * (1/ ne) *  np.trapz(E_EEDF, E_range)
    ######
    def func(x, a, b):
        return a + b * (x ** 0.5)  
    for i in np.arange(start=1E4, stop=1E6, step=1E2):
        edge = (np.argwhere(real_I < 0)[::-1][0][0] - int(i))
        popt, pcov = curve_fit(func, E[:edge], real_I[:edge])
        fit = popt[0] + (popt[1]) * (E[:edge] ** 0.5) 
        edge_I = real_I[:edge][::-1][0]
        edge_fit = fit[-1]
        if abs((edge_I - edge_fit)/ edge_I) > 0.001:
            continue
        else:
            edge = edge
            popt, pcov = curve_fit(func, E[:edge], real_I[:edge])
            b = popt[1]
            #print(b)
            #plt.plot(E[:edge], real_I[:edge])
            #plt.plot(E[:edge], fit)
            break
        break
    
    b = abs(popt[1])
    mi = 6.6335209E-26      # Mass of Ar [kg]
    #mi = 3*1.6735575E-27   # Mass of H3+ [kg]
    ni = (b * np.sqrt(mi)) / (2 * e * (D/2) * L * np.sqrt(2*e))
    print('Electron Density (1/m3)' , ne)
    print('Ion Density (1/m3)' , ni)
    print('Electron Temperature (eV)', Te)
    print('Ion/Electron Density Ration', ni/ne)
    plt.plot(E_range, EEPF, label='%s' % f[32:])
    #plt.plot(E_range, np.fft.fft(EEPF), label='%s' % f[32:])
    #plt.plot(real_V, smsd_real_I)
    plt.yscale('log')
    plt.ylim(1E13, 1E14)
    plt.xlim(-1,17)
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.ylabel('$EEPF (eV^{-1.5}$)', fontsize=18)
    fig = plt.gcf()
    plt.grid(True)
    fig.set_size_inches(10, 10)
    plt.rcParams.update({'font.size': 9})
    plt.legend()
    n_i.append(ni)
    n_e.append(ne)
    T_e.append(Te)
    EEP_F.append(EEPF)
    E_f.append(E_range)
#print(ni)
#print(ne)
#print(n_i)
#print(T_e)
#np.savetxt('ne_ni_Te.txt', np.column_stack([n_e,n_i,T_e]), delimiter = ',')
#np.savetxt('EEPF.txt', np.column_stack([E_f[1], EEP_F[1]]), delimiter = ',')

    
    

    
    







