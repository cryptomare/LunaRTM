from IEM_Exp import coeffs, volume, subsurface, subsurface_volume
from IEM_Exp import surface
from IEM_Exp import rayleigh
from IEM_Exp import transmission
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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
# df = pd.DataFrame(dat, columns=['Sigma_HH', 'Sigma_VV', 'Inc_Angle', 'Epsilon', 'Sigma'])
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
eps_s = 3.15 + 0.001j
eps = 2.7 + 0.003j
vf = 0.1
k = (2.0 * np.pi) / 12.6
r_s = 2
d = 4
theta_all = np.deg2rad(np.arange(5, 86, 5))


# def compile(theta):
#     a, tau, k_r = rayleigh(eps_s=eps_s, eps=eps, vf=vf, k=k, r_s=r_s, d=d)
#     T_hh, T_vv, sub_hh, sub_vv, vol_hh, vol_vv, theta_t = transmission(
#             theta=theta,
#             eps=eps.real
#         )
#     sigma_vol_hh, _ = volume(
#             a, theta, T_hh, T_vv, vol_hh, vol_vv, tau, theta_t
#         )
#     return sigma_vol_hh

# def compile(theta):
#     a, tau, k_r = rayleigh(eps_s=eps_s, eps=eps, vf=vf, k=k, r_s=r_s, d=d)
#     T_hh, T_vv, sub_hh, sub_vv, vol_hh, vol_vv, theta_t = transmission(
#             theta=theta,
#             eps=eps.real
#         )
#     f_hh, f_vv, F_hh, F_vv = coeffs(eps=6.0, theta=theta_t)
#
#     sigma_subsur_hh, _ = subsurface(
#         theta, T_hh, sub_hh, T_vv, sub_vv, tau, theta_t, length=12.6,
#         lambda_wave=12.6, sigma=0.5, f_hh=f_hh, f_vv=f_vv, F_hh=F_hh,
#         F_vv=F_vv, cutoff=1e-16)
#
#     return sigma_subsur_hh

# def compile(theta):
#     a, tau = rayleigh(eps_s=eps_s, eps=eps, vf=vf, k=k, r_s=r_s, d=d)
#     T_hh, T_vv, sub_hh, sub_vv, vol_hh, vol_vv, theta_t = transmission(
#             theta=theta,
#             eps=eps.real
#         )
#     sigma_sub_vol_hh, _ = subsurface_volume(a, theta, eps_s=eps_s,
#                                          eps_sub=8.0+0.05j, T_hh=T_hh,
#                                          sub_hh=sub_hh, T_vv=T_vv,
#                                          sub_vv=sub_vv, tau=tau,
#                                          theta_t=theta_t)
#     return sigma_sub_vol_hh


# vcompile = np.vectorize(compile)
# vcompile = list(map(compile, theta_all.tolist()))
y_vals = np.array(list(map(compile, theta_all.tolist())))
print(y_vals.shape)
plt.plot(np.rad2deg(theta_all), y_vals)
plt.show()
