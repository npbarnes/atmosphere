import numpy as np
import csv
import xlrd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def read_april2016(filename='pluto_atm_april2016.xls', label_row_number=2):
    """Read Darrell Strobel's published atmosphere model from april 2016.
    Return it as a numpy array.
    """
    wb = xlrd.open_workbook(filename=filename)
    sheet = wb.sheet_by_index(0)

    rows = sheet.get_rows()

    # Skip down to where the columns are labels
    for i in range(label_row_number):
        next(rows)
    
    col_labels = next(rows)
    for col_label_cell in col_labels:
        assert col_label_cell.ctype == 1

    first_col = sheet.col(0)[label_row_number+1:]
    for data_cell in first_col:
        assert data_cell.ctype == 2

    # Skip the last two columns
    ret = np.empty((len(first_col), len(col_labels)-2), dtype=np.float64)

    for i,row in enumerate(rows):
        # Skip the last two columns
        for j,cell in enumerate(row[:-2]):
            try:
                ret[i,j] = cell.value
            except ValueError:
                print i,j, row
                raise

    return ret

def read_unknowndate(filename='pluto_atm_unknowndate.csv'):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)

        rows = []
        for row in reader:
            rows.append(row)

    return np.array(rows, dtype=np.float64)

def read_post(filename='pluto_atm_post_NH.csv'):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)

        rows = []
        for row in reader:
            # Cut empty cells at the end of each row
            rows.append(row[:6])

    # Cut the last line that has all empty cells
    rows = rows[:-1]

    return np.array(rows, dtype=np.float64)

@np.vectorize
def analytic_profile(r):
    if r > 30:
        return 0
    elif r < 1.1:
        r = 1.1

    return 1e15*r**-25 + 5e9*r**-8

def ch4_analytic_profile(r):
    return 5e9*r**-8

def exp_fit(data):
    (a,b), pcov = curve_fit(lambda r,a,b: a*np.exp(b*r), data[:,0]-1, data[:,1], p0=(1e13,-30))

    return a,b,lambda r: a*np.exp(b*r)

def power_law(r, a, b):
    return a*r**-b

def dpower_law(r,a,b):
    return np.array([r**-b, -a*np.log(r)*r**-b]).T

def power_fit(data):

    (a,b), pcov = curve_fit(power_law, data[:,0], data[:,1], p0=(5e9,8), jac=dpower_law)

    return a,b, lambda r: power_law(r, a=a, b=b)


if __name__ == '__pain__':
    a = 5
    b = 8

    x = np.linspace(1,5,1000)
    y = power_law(x, a, b) + np.random.randn(1000)/10

    plt.plot(x,y)

    (aa,bb), pcov = curve_fit(power_law, x, y, jac=dpower_law, p0=(4,9))
    #(aa,bb), pcov = curve_fit(power_law, x, y, p0=(4,9))

    plt.plot(x, power_law(x, aa, bb))
    print( aa, bb )

    plt.show()

if __name__ == '__main__':
    april   = read_april2016()
    post    = read_post()
    #unknown = read_unknowndate()

    april_ch4   = april[:,(0,4)]
    post_ch4    = post[:,(0,3)]
    #unknown_ch4 = unknown[:,(0,6)]

    #april_fit = exp_fit(april_ch4)
    #a, b, post_fit  = exp_fit(post_ch4)
    #a,b,post_fit = power_fit(post_ch4)
    #print( a,b )

    plt.semilogx(april_ch4[:,1], april_ch4[:,0]/1187., label='April')
    plt.semilogx(post_ch4[:,1], post_ch4[:,0]/1187., label='Post')
    #plt.plot(april_ch4[:,1], april_ch4[:,0]/1187., label='April')
    #plt.plot(post_ch4[:,1], post_ch4[:,0]/1187., label='Post')
    #plt.semilogy(post_ch4[:,0]/1187. - 1, post_ch4[:,1], label='Post')
    #plt.plot(post_ch4[:,0]/1187., post_ch4[:,1], label='Post')


    #altitudes = np.linspace(1, np.max(post_ch4[:,0]/1187.), 1000)

    #plt.semilogx(analytic_profile(altitudes), altitudes, label='Delamere Profile')
    #plt.semilogx(ch4_analytic_profile(altitudes), altitudes, label='Delamere CH4 Profile')
    #plt.semilogx(april_fit(altitudes), altitudes, label='April exp fit')
    #plt.semilogx(post_fit(altitudes), altitudes, label='Post exp fit')
    #plt.semilogy(altitudes, post_fit(altitudes), label='Post exp fit')
    #plt.plot(altitudes, post_fit(altitudes), label='Post exp fit')

    plt.xlabel('Density ($\mathrm{cm}^{-3}$)')
    plt.ylabel('Radial Distance ($\mathrm{R_p}$)')
    #plt.ylabel('Density ($\mathrm{cm}^{-3}$)')
    #plt.xlabel('Radial Distance ($\mathrm{R_p}$)')

    plt.legend()

    plt.figure()

    post_A = post_ch4[0,1]
    april_A = april_ch4[0,1]

    post_H = (post_ch4[1:,0] - 1187)/(np.log(post_A) - np.log(post_ch4[1:,1]))
    april_H = (april_ch4[1:,0] - 1187)/(np.log(april_A) - np.log(april_ch4[1:,1]))

    plt.plot(post_ch4[1:,0]/1187, post_H)
    plt.plot(april_ch4[1:,0]/1187, april_H)

    plt.show()




