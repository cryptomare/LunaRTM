from IEM_Exp import volume, subsurface, subsurface_volume, total
from IEM_Exp import coeffs, mixture, reg_dielec
from IEM_Exp import surface
from IEM_Exp import rayleigh
from IEM_Exp import transmission
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

matplotlib.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# from matplotlib import pyplot as plt

# df = pd.read_csv("Dielectric_Constant.csv")
# eps_rvals = df["D_r"].values
# eps_ivals = df["Loss_Tangent"].values
# eps_vals = eps_rvals + (1j * eps_ivals)

# theta = np.deg2rad(57.5)
# inc_vals = np.deg2rad(np.linspace(0, 80, 80))
# eps_rvals = np.linspace(1, 15, 1400)
# eps_ivals = 0.003
# eps_vals = eps_rvals + (1j * eps_ivals)
# length = 12.6
# # sigma = 4.982841
# sigma_lst = np.linspace(0.001, 7, 700)
# lambda_wave = 12.6
#
# sigma_hh_vals = list()
# sigma_vv_vals = list()
# eps_dat = list()
# inc_dat = list()
# sigma_dat = list()
#
# for sigma in sigma_lst:
#     for inc in inc_vals:
#         for eps in eps_vals:
#             coeff_dict = coeffs(eps=eps, theta=inc)
#
#             f_hh = coeff_dict['f_hh']
#             f_vv = coeff_dict['f_vv']
#             F_hh = coeff_dict['F_hh']
#             F_vv = coeff_dict['F_vv']
#
#             sigma_dict = surface(
#                 theta=inc,
#                 length=length,
#                 lambda_wave=lambda_wave,
#                 sigma=sigma,
#                 f_hh=f_hh,
#                 f_vv=f_vv,
#                 F_hh=F_hh,
#                 F_vv=F_vv
#             )
#             sigma_hh_vals.append(sigma_dict['sigma_hh'])
#             sigma_vv_vals.append(sigma_dict['sigma_vv'])
#             eps_dat.append(eps)
#             inc_dat.append(inc)
#             sigma_dat.append(sigma)
#
# dat = np.stack(
#     [
#         np.array(sigma_hh_vals),
#         np.array(sigma_vv_vals),
#         np.array(inc_dat),
#         np.array(eps_dat),
#         np.array(sigma_dat)
#
#     ]
# )
#
# dat = np.transpose(dat)
# df = pd.DataFrame(
#                   dat,
#                   columns=[
#                               'Sigma_HH',
#                               'Sigma_VV',
#                               'Inc_Angle',
#                               'Epsilon',
#                               'Sigma'
#                           ]
#                   )
# df.to_csv('Data_Cube.csv')
################################################################################
# x_lst = list()
# y1_lst = list()
# y2_lst = list()
# y3_lst = list()
#
# for sig in sigma_lst:
#
#     coeff_dict = coeffs(eps=2.7, theta=theta)
#     f_hh = coeff_dict['f_hh']
#     f_vv = coeff_dict['f_vv']
#     F_hh = coeff_dict['F_hh']
#     F_vv = coeff_dict['F_vv']
#
#     sigma_dict = surface(
#         theta=theta,
#         length=length,
#         lambda_wave=lambda_wave,
#         sigma=sig,
#         f_hh=f_hh,
#         f_vv=f_vv,
#         F_hh=F_hh,
#         F_vv=F_vv
#     )
#
#     x_lst.append(sig)
#     y1_lst.append(sigma_dict['sigma_hh'])
#     y2_lst.append(sigma_dict['sigma_vv'])
#     y3_lst.append(sigma_dict['ratio'])
#
# x_lst = np.array(x_lst)
# y1_lst = np.array(y1_lst)
# y2_lst = np.array(y2_lst)
# y3_lst = np.array(y3_lst)
#
# sigma_hh_vals = np.array(sigma_hh_vals)
# sigma_vv_vals = np.array(sigma_vv_vals)
# eps_dat = np.array(eps_dat)
# inc_dat = np.array(inc_dat)

# sigma_hh_vals = list()
# sigma_vv_vals = list()
# eps_dat = list()
# inc_dat = list()
#
# for inc in inc_vals:
#     for eps in eps_vals:
#         coeff_dict = coeffs(eps=eps, theta=inc)
#
#         f_hh = coeff_dict['f_hh']
#         f_vv = coeff_dict['f_vv']
#         F_hh = coeff_dict['F_hh']
#         F_vv = coeff_dict['F_vv']
#
#         sigma_dict = surface(
#             theta=inc,
#             length=length,
#             lambda_wave=lambda_wave,
#             sigma=sigma,
#             f_hh=f_hh,
#             f_vv=f_vv,
#             F_hh=F_hh,
#             F_vv=F_vv
#         )
#
#         sigma_hh_vals.append(sigma_dict['sigma_hh'])
#         sigma_vv_vals.append(sigma_dict['sigma_vv'])
#         eps_dat.append(eps)
#         inc_dat.append(inc)
#
# x_dat = list()
# y_dat = list()
# z_dat = list()
#
# for inc, x, y in zip(inc_dat, eps_dat, sigma_hh_vals):
#     # if np.isclose(inc, np.deg2rad(49)):
#     if np.isclose(x.real, 2.7) and np.isclose(inc, np.deg2rad(49)):
#         print(inc, x, y)
#         x_dat.append(x.real)
#         y_dat.append(y.real)
#         z_dat.append(np.rad2deg(inc))

# plt.scatter(x_dat, y_dat)
# plt.show()

###############################################################################

v1 = 0
v2 = 1
ft = 7
bd = 1.525
vf = 0.1
k = (2.0 * np.pi) / 12.6
r_s = 1
d = 5
sigma_2 = 0.2
sigma = 1
length = 12.6
lambda_wave = 12.6
eps_sub = 6.0 + 0.05j
eps_ice = 3.15 + 0.001j
eps_rock = 8.0 + 0.07j


xpts = np.arange(0.2, 90, 0.1)
theta_all = np.deg2rad(np.array(xpts))


def compile(theta):

    eps, eps_real, eps_imag, d_r = reg_dielec(
            ft=ft, bd=bd, lambda_wave=lambda_wave
        )
    
    eps_s = mixture(v1=v1, v2=v2, eps_ice=eps_ice, eps_rock=eps_rock)
    
    a, tau, k_r = rayleigh(
            eps_s=eps_s, eps=eps, eps_real=eps_real,
            eps_imag=eps_imag, vf=vf, k=k, r_s=r_s, d=d
        )

    tt_hh, tt_vv, sub_hh, sub_vv, theta_t = transmission(theta=theta, eps=eps)

    _, _, f_hh, f_vv, ff_hh, ff_vv = coeffs(eps=eps_sub, theta=theta_t)

    sigma_hh_1, sigma_vv_1, _, _ = surface(
            theta=theta_t, length=length,
            lambda_wave=lambda_wave, sigma=sigma_2,
            f_hh=f_hh, f_vv=f_vv, ff_hh=ff_hh,
            ff_vv=ff_vv, cutoff=1e-16
        )

    _, _, f_hh, f_vv, F_hh, F_vv = coeffs(eps=eps, theta=theta)

    sigma_hh, sigma_vv, sigma_sur_hh, sigma_sur_vv = surface(
            theta=theta,
            length=length,
            lambda_wave=lambda_wave,
            sigma=sigma, cutoff=1e-16,
            f_hh=f_hh, f_vv=f_vv,
            ff_hh=F_hh, ff_vv=F_vv
        )

    sigma_vol_hh, sigma_vol_vv, volume_hh, volume_vv = volume(
            a=a, theta=theta, tt_hh=tt_hh,
            tt_vv=tt_vv, sub_hh=sub_hh,
            sub_vv=sub_vv, tau=tau, theta_t=theta_t
        )

    sigma_subsur_hh, sigma_subsur_vv, subsur_hh, subsur_vv = subsurface(
            theta=theta, tt_hh=tt_hh,
            sub_hh=sub_hh, tt_vv=tt_vv,
            sub_vv=sub_vv, tau=tau,
            theta_t=theta_t,
            sigma_hh=sigma_hh_1,
            sigma_vv=sigma_vv_1
        )

    r_h, r_v, _, _, _, _ = coeffs(eps=eps_sub, theta=theta_t)

    sub_out = subsurface_volume(
            a=a,
            theta=theta,
            eps_s=eps_s,
            eps_sub=eps_sub,
            tt_hh=tt_hh,
            tt_vv=tt_vv,
            tau=tau,
            theta_t=theta_t,
            sigma_2=sigma_2,
            k=k
        )
    sigma_sub_vol_hh, sigma_sub_vol_vv, sub_vol_hh, sub_vol_vv = sub_out
    sigma_total_hh, sigma_total_vv = total(
            sigma_hh=sigma_hh,
            sigma_vv=sigma_vv,
            sigma_vol_hh=volume_hh,
            sigma_vol_vv=volume_vv,
            sigma_subsur_hh=subsur_hh,
            sigma_subsur_vv=subsur_vv,
            sigma_sub_vol_hh=sub_vol_hh,
            sigma_sub_vol_vv=sub_vol_vv)
    return (
                sigma_sub_vol_hh,
                sigma_subsur_hh,
                sigma_vol_hh,
                sigma_sur_hh,
                sigma_total_hh
            )


y_vals = np.array(list(map(compile, theta_all.tolist())))
theta_m = np.deg2rad(np.arange(5, 90, 5))
markers = np.array(list(map(compile, theta_m.tolist())))
dat = np.append(
    np.reshape(np.rad2deg(theta_all), (theta_all.size, 1)), y_vals, axis=1)
mrk = np.append(
    np.reshape(np.rad2deg(theta_m), (theta_m.size, 1)), markers, axis=1)
cols = ['x', 'y1', 'y2', 'y3', 'y4', 'y5']
df_data = pd.DataFrame(dat.real, columns=cols)
df_marker = pd.DataFrame(mrk.real, columns=cols)
df_data.to_csv('data.csv', index=False)
df_marker.to_csv('marker.csv', index=False)
fig = plt.figure(num='Sensitivity Plot', tight_layout=True)
ax = fig.add_subplot(111)

ax.set_ylim([-90, 10])
l1, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 0])
l2, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 1])
l3, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 2])
l4, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 3])
l5, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 4])
l1.set_label('Subsurface - Volume')
l2.set_label('Subsurface')
l3.set_label('Volume')
l4.set_label('Surface')
l5.set_label('Total')

ax.scatter(np.rad2deg(theta_m), markers[:, 0].real)
ax.scatter(np.rad2deg(theta_m), markers[:, 1].real)
ax.scatter(np.rad2deg(theta_m), markers[:, 2].real)
ax.scatter(np.rad2deg(theta_m), markers[:, 3].real)
ax.scatter(np.rad2deg(theta_m), markers[:, 4].real)
ax.axvline(x=10, c='grey', ls='-')
ax.axvline(x=20, c='grey', ls='-')
ax.axvline(x=30, c='grey', ls='--')
ax.axvline(x=35, c='grey', ls='-')
ax.axvline(x=40, c='grey', ls='--')
ax.axvline(x=44, c='grey', ls='--')
ax.axvline(x=49, c='grey', ls='-')
ax.axvline(x=54, c='grey', ls='--')
ax.axvspan(10, 40, alpha=0.2, color='grey')
ax.axvspan(44, 54, alpha=0.2, color='grey')
xticks = np.arange(0, 91, 10)
xstr = list(np.array(xticks).astype(str))
xstr = ['$' + xtick + '^\\circ$' for xtick in xstr]
ax.set_xticks(xticks)
ax.set_yticks(np.arange(-90, 11, 20))
ax.text(
        15, -70, r'Chandrayaan-2 \n(Hybrid Pol)\n$10^\circ-20^\circ$',
        rotation=90, horizontalalignment='center', verticalalignment='center'
    )

ax.text(
        25, -70, r'Chandrayaan-2 \n(Full Pol)\n$20^\circ-30^\circ$',
        rotation=90, horizontalalignment='center', verticalalignment='center'
    )


ax.text(
        38, -70, r'Chandrayaan-1\n$35^\circ$', rotation=90,
        horizontalalignment='center', verticalalignment='center'
    )

ax.text(
        52, -70, r'LRO\n$49^\circ$', rotation=90,
        horizontalalignment='center', verticalalignment='center'
    )

ax.set_xlabel(r'Incidence Angle ($^\circ$)')
ax.set_ylabel(r'Radar Backscatter : $\sigma^0$ (dB)')

ax.legend()
ax.set_xlim([0, 88.5])

plt.show()
