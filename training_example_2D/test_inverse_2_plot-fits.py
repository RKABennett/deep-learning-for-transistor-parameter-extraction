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
import NN_variables as NNv
dir_path = os.path.dirname(os.path.abspath(__file__))

###############################################################################
#
# Plot settings
#
###############################################################################

fontsize = 16
mpl.rcParams.update({'font.size': fontsize})

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 0.4
mpl.rcParams['ytick.major.width'] = 0.4
mpl.rcParams['xtick.minor.width'] = 0.2
mpl.rcParams['ytick.minor.width'] = 0.2
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.markersize'] = 10

###############################################################################
#
# Load data
#
###############################################################################

quantile = 0.05

trainset = 500
modelver = 0

subfolder_name = 'NN_inverse_pred'.format(trainset, modelver)
filenames = glob.glob(dir_path + '/fits_inverse/' + subfolder_name + '/*.dat')
errors = []

num_entries = len(filenames)
target = int(num_entries*quantile)
for filename in filenames:
    error = filename.replace(dir_path + '/fits_inverse/' + subfolder_name + '/', '')
    error = error.replace('error=', '')
    error = error.replace('_pred.dat', '')
    errors.append(float(error))

errors = sorted(errors)
target_error = errors[target]

data_pred = np.loadtxt(dir_path + '/fits_inverse/NN_inverse_pred/error={:.12f}_pred.dat'.format(target_error))
data_actual = np.loadtxt(dir_path + '/fits_inverse/NN_inverse_actual/error={:.12f}_actual.dat'.format(target_error))

Id_100_pred = data_pred[0]
Id_100_log_pred = data_pred[1]
Id_1000_pred = data_pred[2]
Id_1000_log_pred = data_pred[3]

Id_100_actual = data_actual[0]
Id_100_log_actual = data_actual[1]
Id_1000_actual = data_actual[2]
Id_1000_log_actual = data_actual[3]

###############################################################################
#
# Plot data
#
###############################################################################

fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()

start, stop, skip = 0, 32, 3
zorder_pred = 100001
zorder_actual = 10000
actual_OLcolor = 'k'
actual_Fcolor_1 = '#4dadd6'
pred_color_1 = '#19546d'
actual_Fcolor_01 = '#d64d69'
pred_color_01 = '#6d192a'


scale = 10**6 # A/um to uA/um conversion factor
# Linear scale, Vds = 0.1
ax2.plot(
        NNv.V[start:stop:skip], 
        scale*Id_100_actual[start:stop:skip],
        marker='o',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_01,
        ls='None',
        label='Vds=0.1, Actual',
        zorder = zorder_actual
        )
ax2.plot(
        NNv.V, 
        scale*Id_100_pred,
        marker='None',
        color=pred_color_01, 
        ls='-',
        label='Vds=0.1, Pred',
        zorder = zorder_pred
       )

# Linear scale, Vds = 1.0
ax2.plot(
        NNv.V[start:stop:skip], 
        scale*Id_1000_actual[start:stop:skip],
        marker='s',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_1,
        ls='None',
        label='Vds=1.0, Actual',
        zorder = zorder_actual,
        )

ax2.plot(
        NNv.V, 
        scale*Id_1000_pred,
        marker='None',
        color=pred_color_1, 
        ls='-',
        label='Vds=1.0, Pred',
        zorder = zorder_pred
        )

# Log scale, Vds = 0.1
ax1.semilogy(
        NNv.V[start:stop:skip], 
        scale*np.power(10, Id_100_log_actual)[start:stop:skip],
        marker='o',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_01,
        ls='None',
        label='Log Vds=0.1, Actual',
        zorder = zorder_actual
        )

ax1.semilogy(
        NNv.V, 
        scale*np.power(10, Id_100_log_pred),
        marker='None',
        color=pred_color_01, 
        ls='-',
        label='Log Vds=0.1, Pred',
        zorder = zorder_pred
        )

# Log scale, Vds = 1.0
ax1.semilogy(
        NNv.V[start:stop:skip], 
        scale*np.power(10, Id_1000_log_actual)[start:stop:skip],
        marker='s',
        color=actual_OLcolor,
        markerfacecolor = actual_Fcolor_1,
        ls='None',
        label='Log Vds=0.1, Actual',
        zorder = zorder_actual
        )

ax1.semilogy(
        NNv.V, 
        scale*np.power(10, Id_1000_log_pred),
        marker='None',
        color=pred_color_1, 
        ls='-',
        label='Log Vds=0.1, Pred',
        zorder = zorder_pred
        )

plt.tight_layout()
plt.savefig(dir_path + '/reverse_engineered_fit_quantile={}_R2={}.png'.format(quantile, float(target_error)))
plt.close()

