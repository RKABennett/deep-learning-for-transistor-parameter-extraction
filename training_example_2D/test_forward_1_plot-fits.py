###############################################################################
#
# Import functions
#
###############################################################################
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
dir_path = os.path.dirname(os.path.abspath(__file__))

###############################################################################
#
# Plot settings
#
###############################################################################

fontsize = 16
mpl.rcParams.update({'font.size': fontsize})

###############################################################################
#
# Plot data
#
###############################################################################

quantile = 0.05

subfolder_name = 'NN_forward_pred'
filenames = glob.glob(dir_path + '/fits_forward/' + subfolder_name + '/*.dat')
errors = []


for filename in filenames:
    error = filename.replace(dir_path + '/fits_forward/' + subfolder_name + '/', '')
    error = error.replace('error=', '')
    error = error.replace('_pred.dat', '')
    errors.append(float(error))

num_entries = len(filenames)
target = int(num_entries*quantile)


errors = sorted(errors)
target_error = errors[target]

data_pred = np.loadtxt(dir_path + '/fits_forward/NN_forward_pred/error={:.12f}_pred.dat'.format(target_error))
data_actual = np.loadtxt(dir_path + '/fits_forward/NN_forward_actual/error={:.12f}_actual.dat'.format(target_error))

V = np.linspace(-5.9, 49.9, 32)

Id_100_pred = data_pred[0]
Id_100_log_pred = data_pred[1]
Id_1000_pred = data_pred[2]
Id_1000_log_pred = data_pred[3]

Id_100_actual = data_actual[0]
Id_100_log_actual = data_actual[1]
Id_1000_actual = data_actual[2]
Id_1000_log_actual = data_actual[3]

scale = 10**6

fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()

start, stop, skip = 0, 32, 3

zorder_pred = 10001
zorder_actual = 10000

actual_OLcolor = 'k'

actual_Fcolor_1 = '#4dadd6'
pred_color_1 = '#19546d'

actual_Fcolor_01 = '#d64d69'
pred_color_01 = '#6d192a'

# Linear scale, Vds = 0.1
ax2.plot(
    V[start:stop:skip], 
    scale*Id_100_actual[start:stop:skip],
    marker='o',
    color=actual_OLcolor,
    markerfacecolor = actual_Fcolor_01,
    ls='None',
    label='Vds=0.1, Actual',
    zorder = zorder_actual
    )
ax2.plot(
    V, 
    scale*Id_100_pred,
    marker='None',
    color=pred_color_01, 
    ls='-',
    label='Vds=0.1, Pred',
    zorder = zorder_pred
   )

# Linear scale, Vds = 1.0
ax2.plot(
    V[start:stop:skip], 
    scale*Id_1000_actual[start:stop:skip],
    marker='s',
    color=actual_OLcolor,
    markerfacecolor = actual_Fcolor_1,
    ls='None',
    label='Vds=1.0, Actual',
    zorder = zorder_actual,
    )

ax2.plot(
    V, 
    scale*Id_1000_pred,
    marker='None',
    color=pred_color_1, 
    ls='-',
    label='Vds=1.0, Pred',
    zorder = zorder_pred
    )

# Log scale, Vds = 0.1
ax1.semilogy(
    V[start:stop:skip], 
    scale*np.power(10, Id_100_log_actual)[start:stop:skip],
    marker='o',
    color=actual_OLcolor,
    markerfacecolor = actual_Fcolor_01,
    ls='None',
    label='Log Vds=0.1, Actual',
    zorder = zorder_actual
    )

ax1.semilogy(
    V, 
    scale*np.power(10, Id_100_log_pred),
    marker='None',
    color=pred_color_01, 
    ls='-',
    label='Log Vds=0.1, Pred',
    zorder = zorder_pred
    )

# Log scale, Vds = 1.0
ax1.semilogy(
    V[start:stop:skip], 
    scale*np.power(10, Id_1000_log_actual)[start:stop:skip],
    marker='s',
    color=actual_OLcolor,
    markerfacecolor = actual_Fcolor_1,
    ls='None',
    label='Log Vds=0.1, Actual',
    zorder = zorder_actual
    )

ax1.semilogy(
    V, 
    scale*np.power(10, Id_1000_log_pred),
    marker='None',
    color=pred_color_1, 
    ls='-',
    label='Log Vds=0.1, Pred',
    zorder = zorder_pred
    )

plt.tight_layout()
plt.savefig(dir_path + '/NN_forward_test_quantile={}_R2=.png'.format(quantile, float(target_error)), transparent = False)
plt.close()

