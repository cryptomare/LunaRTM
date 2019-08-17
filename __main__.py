from IEM_Exp import volume, subsurface, subsurface_volume, total
from IEM_Exp import coeffs, mixture, reg_dielec
from IEM_Exp import surface
from IEM_Exp import rayleigh
from IEM_Exp import transmission
from pathlib import Path
from DataPreparation import read_feature
import numpy as np
# import matplotlib
# from matplotlib import pyplot as plt
# import pandas as pd
# matplotlib.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# v1 = 0
# v2 = 1
# ft = 7
# bd = 1.525
# vf = 0.1
# r_s = 1
# d = 5
# sigma = 1
# sigma_2 = sigma / 5.0
# lambda_wave = 12.6
# k = (2.0 * np.pi) / lambda_wave
# length = lambda_wave
# eps_sub = 6.0 + 0.05j
# eps_ice = 3.15 + 0.001j
# eps_rock = 8.0 + 0.07j

# xpts = np.arange(0.2, 90, 0.1)
# theta_all = np.deg2rad(np.array(xpts))
from Model import Rover

ft_arr = np.arange(0, 31, 2).astype(float)
ft_arr[0] = 0.005

bd_arr = np.arange(0.93, 3.35, 0.1).astype(float)

vf_arr = np.arange(0, 0.11, 0.01).astype(float)
vf_arr[0] = 0.0001

d_arr = np.arange(4, 16, 1.0).astype(float)

sigma_arr = np.arange(0, 5, 1).astype(float)
sigma_arr[0] = 0.1

theta_arr = np.arange(0, 90, 5).astype(float)
theta_arr[0] = 0.5
theta_arr_rad = np.deg2rad(theta_arr)

in_arrays = (ft_arr, bd_arr, vf_arr, d_arr, sigma_arr, theta_arr_rad)
out_arrays = np.meshgrid(*in_arrays)
ft_array, bd_array, vf_array, d_array, sigma_array, theta_array = out_arrays


data_path = Path('data/data_dump.npy')
if not data_path.is_file():
    def compiled(theta, v1, v2, ft, bd, vf, r_s, d, sigma,
                 lambda_wave, eps_sub, eps_ice, eps_rock):
        sigma_2 = sigma / 5.0
        k = (2.0 * np.pi) / lambda_wave
        length = lambda_wave
        eps, d_r = reg_dielec(
                ft=ft, bd=bd, lambda_wave=lambda_wave
            )
        eps_s = mixture(v1=v1, v2=v2, eps_ice=eps_ice, eps_rock=eps_rock)

        a, tau, k_r = rayleigh(
                eps_s=eps_s, eps=eps,
                vf=vf, k=k, r_s=r_s, d=d
            )

        tt_hh, tt_vv, sub_hh, sub_vv, theta_t = transmission(
                theta=theta, eps=eps
            )

        _, _, f_hh, f_vv, ff_hh, ff_vv = coeffs(eps=eps_sub, theta=theta_t)

        sigma_hh_1, sigma_vv_1, _, _ = surface(
                theta=theta_t, length=length,
                lambda_wave=lambda_wave, sigma=sigma_2,
                f_hh=f_hh, f_vv=f_vv, ff_hh=ff_hh,
                ff_vv=ff_vv, cutoff=1e-16
            )

        _, _, f_hh, f_vv, ff_hh, ff_vv = coeffs(eps=eps, theta=theta)

        sigma_hh, sigma_vv, sigma_sur_hh, sigma_sur_vv = surface(
                theta=theta,
                length=length,
                lambda_wave=lambda_wave,
                sigma=sigma, cutoff=1e-16,
                f_hh=f_hh, f_vv=f_vv,
                ff_hh=ff_hh, ff_vv=ff_vv
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
                    theta,
                    eps,
                    d_r,
                    sigma,
                    sigma_sub_vol_hh,
                    sigma_subsur_hh,
                    sigma_vol_hh,
                    sigma_sur_hh,
                    sigma_total_hh,
                    sigma_sub_vol_vv,
                    sigma_subsur_vv,
                    sigma_vol_vv,
                    sigma_sur_vv,
                    sigma_total_vv
                )


    orbit = np.vectorize(compiled, excluded=[
            'v1', 'v2', 'r_s', 'lambda_wave', 'eps_sub', 'eps_ice', 'eps_rock'
        ], otypes=[object])
    print('Started')
    data = orbit(
            theta=theta_array,
            v1=0, v2=1, ft=ft_array, bd=bd_array, vf=vf_array, r_s=1, d=d_array,
            sigma=sigma_array, lambda_wave=12.6, eps_sub=(6.0 + 0.05j),
            eps_ice=(3.15 + 0.001j), eps_rock=(8.0 + 0.07j),
        )
    print('Finished')
    np.save(file=data_path, arr=data)

data = np.load(data_path, allow_pickle=True)
flat_data = data.ravel()
eps_ = read_feature(flat_data, 1)
eps_real = eps_.real
eps_imag = eps_.imag
d_r_ = read_feature(flat_data, 2)
sigma_ = read_feature(flat_data, 3)
sigma_real = sigma_.real
# sigma_sub_vol_hh_ = read_feature(flat_data, 4)
# sigma_subsur_hh_ = read_feature(flat_data, 5)
# sigma_vol_hh_ = read_feature(flat_data, 6)
# sigma_sur_hh_ = read_feature(flat_data, 7)
sigma_total_hh_ = read_feature(flat_data, 8)
st_hh_real = sigma_total_hh_.real
# sigma_sub_vol_vv_ = read_feature(flat_data, 9)
# sigma_subsur_vv_ = read_feature(flat_data, 10)
# sigma_vol_vv_ = read_feature(flat_data, 11)
# sigma_sur_vv_ = read_feature(flat_data, 12)
sigma_total_vv_ = read_feature(flat_data, 13)
st_vv_real = sigma_total_vv_.real
theta_ = theta_array.ravel()
data_in = np.stack(
            (st_hh_real, st_vv_real, theta_),
            axis=-1
    )
data_out = np.stack(
        (eps_real, eps_imag, d_r_, sigma_real),
        axis=-1
    )
datx = Path('data/data_input.npy')
daty = Path('data/data_output.npy')
if not (datx.is_file() and daty.is_file()):
    np.save(file=datx, arr=data_in)
    np.save(file=daty, arr=data_out)
structure = Rover()
structure.ready()
structure.learn(x=data_in, y=data_out, epochs=100)
structure.store('model/model.h5')
# y_vals = np.array(list(map(compiled, theta_all.tolist())))
# theta_m = np.deg2rad(np.arange(5, 90, 5))
# markers = np.array(list(map(compiled, theta_m.tolist())))
# dat = np.append(
#     np.reshape(np.rad2deg(theta_all), (theta_all.size, 1)), y_vals, axis=1)
# mrk = np.append(
#     np.reshape(np.rad2deg(theta_m), (theta_m.size, 1)), markers, axis=1)
# cols = ['x', 'y1', 'y2', 'y3', 'y4', 'y5']
# df_data = pd.DataFrame(dat.real, columns=cols)
# df_marker = pd.DataFrame(mrk.real, columns=cols)
# df_data.to_csv('data.csv', index=False)
# df_marker.to_csv('marker.csv', index=False)
# fig = plt.figure(num='Sensitivity Plot', tight_layout=True)
# ax = fig.add_subplot(111)
#
# ax.set_ylim([-90, 10])
# l1, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 0])
# l2, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 1])
# l3, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 2])
# l4, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 3])
# l5, = ax.plot(np.rad2deg(theta_all), y_vals.real[:, 4])
# l1.set_label('Subsurface - Volume')
# l2.set_label('Subsurface')
# l3.set_label('Volume')
# l4.set_label('Surface')
# l5.set_label('Total')
#
# ax.scatter(np.rad2deg(theta_m), markers[:, 0].real)
# ax.scatter(np.rad2deg(theta_m), markers[:, 1].real)
# ax.scatter(np.rad2deg(theta_m), markers[:, 2].real)
# ax.scatter(np.rad2deg(theta_m), markers[:, 3].real)
# ax.scatter(np.rad2deg(theta_m), markers[:, 4].real)
# ax.axvline(x=10, c='grey', ls='-')
# ax.axvline(x=20, c='grey', ls='-')
# ax.axvline(x=30, c='grey', ls='--')
# ax.axvline(x=35, c='grey', ls='-')
# ax.axvline(x=40, c='grey', ls='--')
# ax.axvline(x=44, c='grey', ls='--')
# ax.axvline(x=49, c='grey', ls='-')
# ax.axvline(x=54, c='grey', ls='--')
# ax.axvspan(10, 40, alpha=0.2, color='grey')
# ax.axvspan(44, 54, alpha=0.2, color='grey')
# xticks = np.arange(0, 91, 10)
# xstr = list(np.array(xticks).astype(str))
# xstr = ['$' + xtick + '^\\circ$' for xtick in xstr]
# ax.set_xticks(xticks)
# ax.set_yticks(np.arange(-90, 11, 20))
# ax.text(
#         15, -70, r'Chandrayaan-2 \n(Hybrid Pol)\n$10^\circ-20^\circ$',
#         rotation=90, horizontalalignment='center', verticalalignment='center'
#     )
#
# ax.text(
#         25, -70, r'Chandrayaan-2 \n(Full Pol)\n$20^\circ-30^\circ$',
#         rotation=90, horizontalalignment='center', verticalalignment='center'
#     )
#
#
# ax.text(
#         38, -70, r'Chandrayaan-1\n$35^\circ$', rotation=90,
#         horizontalalignment='center', verticalalignment='center'
#     )
#
# ax.text(
#         52, -70, r'LRO\n$49^\circ$', rotation=90,
#         horizontalalignment='center', verticalalignment='center'
#     )
#
# ax.set_xlabel(r'Incidence Angle ($^\circ$)')
# ax.set_ylabel(r'Radar Backscatter : $\sigma^0$ (dB)')
#
# ax.legend()
# ax.set_xlim([0, 88.5])
#
# plt.show()
