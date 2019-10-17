# from IEM_Exp import volume, subsurface, subsurface_volume, total
# from IEM_Exp import coeffs, mixture, reg_dielec
# from IEM_Exp import surface
# from IEM_Exp import rayleigh
# from IEM_Exp import transmission
from pathlib import Path
from sklearn.utils.extmath import cartesian
# from DataPreparation import read_feature
import numpy as np
from numpy import memmap
# import matplotlib
# import pickle
from Model import Rover
# from matplotlib import pyplot as plt
# import pandas as pd
from joblib import Parallel, delayed
# matplotlib.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# v1_ = 0
# v2_ = 1
# ft_ = 7
# bd_ = 1.525
# vf_ = 0.1
# r_s_ = 1
# d_ = 5
# sigma_ = 1
# sigma_2_ = sigma_ / 5.0
# lambda_wave_ = 1
# k_ = (2.0 * np.pi) / lambda_wave_
# length_ = lambda_wave_
# eps_sub_ = 6.0 + 0.05j
# eps_ice_ = 3.15 + 0.001j
# eps_rock_ = 8.0 + 0.07j
#
# xpts = np.arange(0.2, 90, 0.1)
# theta_all = np.deg2rad(np.array(xpts))
# lambda_range = np.linspace(10, 100, 10).astype(np.float32)
# extra_lambda = np.array([12.6, 23.98], dtype=np.float32)
# lambda_range = np.concatenate((lambda_range, extra_lambda), axis=0)
#
#
# def inflate(ar, shape):
#     al = list()
#     for k in ar.ravel():
#         al.append(np.full(shape=shape, fill_value=k))
#     return np.stack(al, axis=-1)
#
#
# data_path = Path(
#     '/home/abhisek/Documents/RadarBackscatterModel/data_upd/new_data.npy'
#     )
# if not data_path.is_file():
#     ft_arr = np.arange(0, 33, 3).astype(np.float32)
#     ft_arr[0] = 0.005
#
#     bd_arr = np.arange(0.93, 3.33, 0.2).astype(np.float32)
#
#     # vf_arr = np.arange(0, 0.11, 0.01).astype(np.float32)
#     # vf_arr[0] = 0.0001
#
#     d_arr = np.arange(4, 16, 1).astype(np.float32)
#
#     s_count = 11
#     i = 0
#     slist = list()
#     for l in lambda_range:
#         ub = l / np.pi
#         slist.append((np.linspace(0.001, ub, s_count)).astype(np.float32))
#     scol = np.concatenate(slist, axis=0)
#     theta_arr = np.arange(0, 91, 10).astype(np.float32)
#     theta_arr[0] = 0.5
#     theta_arr[-1] = 89.5
#     theta_arr_rad = np.deg2rad(theta_arr)
#     dummy_sigma = np.zeros(shape=(s_count,), dtype=np.float32)
#     # print(ft_arr.shape, bd_arr.shape, vf_arr.shape, d_arr.shape,
#     #      theta_arr_rad.shape, dummy_sigma.shape, lambda_range.shape)
#     in_arrays = (
#             ft_arr,
#             bd_arr,
#             d_arr,
#             theta_arr_rad,
#             lambda_range,
#             dummy_sigma
#         )
#     ad = cartesian(in_arrays)
#     ad[:, 5] = np.tile(scol, ad[:, 5].size // scol.size)
#     ft_array = ad[:, 0]
#     bd_array = ad[:, 1]
#     # vf_array = ad[:, 2]
#     d_array = ad[:, 2]
#     theta_array = ad[:, 3]
#     lw_array = ad[:, 4]
#     sigma_array = ad[:, 5]
#
#     @delayed
#     def compiled(theta, v1, v2, ft, bd, vf, r_s, d, sigma,
#                  lambda_wave, eps_sub, eps_ice, eps_rock):
#         sigma_2 = sigma / 5.0
#         k = (2.0 * np.pi) / lambda_wave
#         length = lambda_wave
#         eps, d_r = reg_dielec(
#                 ft=ft, bd=bd, lambda_wave=lambda_wave
#             )
#         eps_s = mixture(v1=v1, v2=v2, eps_ice=eps_ice, eps_rock=eps_rock)
#
#         a, tau, k_r = rayleigh(
#                 eps_s=eps_s, eps=eps,
#                 vf=vf, k=k, r_s=r_s, d=d
#             )
#
#         tt_hh, tt_vv, sub_hh, sub_vv, theta_t = transmission(
#                 theta=theta, eps=eps
#             )
#
#         _, _, f_hh, f_vv, ff_hh, ff_vv = coeffs(eps=eps_sub, theta=theta_t)
#
#         sigma_hh_1, sigma_vv_1, _, _ = surface(
#                 theta=theta_t, length=length,
#                 lambda_wave=lambda_wave, sigma=sigma_2,
#                 f_hh=f_hh, f_vv=f_vv, ff_hh=ff_hh,
#                 ff_vv=ff_vv, cutoff=1e-16
#             )
#
#         _, _, f_hh, f_vv, ff_hh, ff_vv = coeffs(eps=eps, theta=theta)
#
#         sigma_hh, sigma_vv, sigma_sur_hh, sigma_sur_vv = surface(
#                 theta=theta,
#                 length=length,
#                 lambda_wave=lambda_wave,
#                 sigma=sigma, cutoff=1e-16,
#                 f_hh=f_hh, f_vv=f_vv,
#                 ff_hh=ff_hh, ff_vv=ff_vv
#             )
#
#         sigma_vol_hh, sigma_vol_vv, volume_hh, volume_vv = volume(
#                 a=a, theta=theta, tt_hh=tt_hh,
#                 tt_vv=tt_vv, sub_hh=sub_hh,
#                 sub_vv=sub_vv, tau=tau, theta_t=theta_t
#             )
#
#         sigma_subsur_hh, sigma_subsur_vv, subsur_hh, subsur_vv = subsurface(
#                 theta=theta, tt_hh=tt_hh,
#                 sub_hh=sub_hh, tt_vv=tt_vv,
#                 sub_vv=sub_vv, tau=tau,
#                 theta_t=theta_t,
#                 sigma_hh=sigma_hh_1,
#                 sigma_vv=sigma_vv_1
#             )
#
#         r_h, r_v, _, _, _, _ = coeffs(eps=eps_sub, theta=theta_t)
#
#         sub_out = subsurface_volume(
#             a=a,
#             theta=theta,
#             eps_s=eps_s,
#             eps_sub=eps_sub,
#             tt_hh=tt_hh,
#             tt_vv=tt_vv,
#             tau=tau,
#             theta_t=theta_t,
#             sigma_2=sigma_2,
#             k=k
#             )
#         sigma_sub_vol_hh, sigma_sub_vol_vv, sub_vol_hh, sub_vol_vv = sub_out
#         sigma_total_hh, sigma_total_vv = total(
#             sigma_hh=sigma_hh,
#             sigma_vv=sigma_vv,
#             sigma_vol_hh=volume_hh,
#             sigma_vol_vv=volume_vv,
#             sigma_subsur_hh=subsur_hh,
#             sigma_subsur_vv=subsur_vv,
#             sigma_sub_vol_hh=sub_vol_hh,
#             sigma_sub_vol_vv=sub_vol_vv)
#         return [
#                     lambda_wave,
#                     theta,
#                     sigma_total_hh.real,
#                     sigma_total_vv.real,
#                     eps.real,
#                     eps.imag,
#                     sigma,
#                     d
#                 ]
#
#     print('Started\n')
#     data = Parallel(n_jobs=10, verbose=60)(compiled(
#             v1=v1_,
#             v2=v2_,
#             r_s=r_s_,
#             lambda_wave=lw,
#             eps_sub=eps_sub_,
#             eps_ice=eps_ice_,
#             eps_rock=eps_rock_,
#             bd=bd,
#             vf=vf_,
#             d=d,
#             sigma=sigma,
#             ft=ft,
#             theta=theta
#         )
#         for bd, d, sigma, ft, theta, lw in zip(
#             bd_array,
#             d_array,
#             sigma_array,
#             ft_array,
#             theta_array,
#             lw_array
#         )
#         )
#     print('\nProcessing Finished')
#     dat = np.array(data, dtype=np.float32)
#     np.save(data_path, dat)
#     print('\nWriting Complete')

################################################################################
data_dir = Path('/home/abhisek/Documents/RadarBackscatterModel/data_upd/')
# inmap = memmap(
#         str(((data_dir / 'all_input.dat').absolute())),
#         dtype=np.float32,
#         mode='r',
#         shape=(11499840, 4)
#     )
# outmap = memmap(
#         str(((data_dir / 'all_output.dat').absolute())),
#         dtype=np.float32,
#         mode='r',
#         shape=(11499840, 5)
#     )
main_data = np.load(data_dir / 'new_data.npy')

#####
# main_data[:, 0] = main_data[:, 0] / 100.0
# main_data[:, 1] = main_data[:, 1] / np.deg2rad(90)
# main_data[:, 2] = (main_data[:, 2] + 125) / 100.0
# main_data[:, 3] = (main_data[:, 3] + 125) / 100.0
# main_data[:, 4] = main_data[:, 4] / 10.0
# main_data[:, 5] = main_data[:, 5] * 10
# main_data[:, 6] = main_data[:, 6] / 32.0
# main_data[:, 7] = main_data[:, 7] / 16.0
#####

inmap = main_data[:, :4]
outmap = main_data[:, 4:-2]
structure = Rover(input_features=4, output_features=2)
structure.ready()
structure.learn(x=inmap, y=outmap, batch_size=1000, epochs=10)
structure.store(
        str((data_dir / 'Model.h5').absolute())
    )

################################################################################
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
