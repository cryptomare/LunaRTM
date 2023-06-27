import numpy as np


def calc(n, theta, length, k, sigma, f, F):
    W = 2 * np.pi * n * (length ** 2) / (
        ((n * 2) + ((2 * k * length * np.sin(theta)) ** 2)) ** 1.5
    )
    I_1 = ((2 * k * np.cos(theta) * sigma) ** n) * f
    I_3 = (((k * np.cos(theta) * sigma) ** n) * F)
    I_2 = (np.exp(-1 * (k ** 2) * ((np.cos(theta)) ** 2) * (sigma ** 2)))

    I = (I_1 * I_2) + I_3

    return W, I


def surface(k, theta, length, sigma, f_hh, f_vv, F_hh, F_vv, cutoff=1e-16):
    i = 1
    indicator = 1 / np.math.factorial(i)
    invariant = ((k ** 2) / (4 * np.pi)) * np.exp(
        -2 * (k ** 2) * ((np.cos(theta)) ** 2) * (sigma ** 2))
    register_hh = 0.0
    register_vv = 0.0

    while not np.isclose(indicator, 0.0, atol=cutoff):

        W_hh, I_hh = calc(n=i, theta=theta, length=length, k=k, sigma=sigma,
                          f=f_hh, F=F_hh)
        W_vv, I_vv = calc(n=i, theta=theta, length=length, k=k, sigma=sigma,
                          f=f_vv, F=F_vv)

        mod_I_hh = I_hh * np.conj(I_hh)
        mod_I_vv = I_vv * np.conj(I_vv)

        if W_hh != 0 and W_vv != 0:
            w_var_hh = W_hh * indicator
            w_var_vv = W_vv * indicator

            sigma_hh_iter = mod_I_hh * w_var_hh
            sigma_vv_iter = mod_I_vv * w_var_vv

            register_hh += sigma_hh_iter
            register_vv += sigma_vv_iter

        i += 1
        indicator = 1 / np.math.factorial(i)

    sigma_hh = (invariant * register_hh)
    sigma_vv = (invariant * register_vv)

    sigma_sur_hh = 10 * np.log10(sigma_hh)
    sigma_sur_vv = 10 * np.log10(sigma_vv)

    return sigma_hh, sigma_vv, sigma_sur_hh, sigma_sur_vv


def coeffs(eps, theta):
    r_h_num = (np.cos(theta) - ((eps - ((np.sin(theta)) ** 2)) ** 0.5))
    r_h_denom = (np.cos(theta) + ((eps - ((np.sin(theta)) ** 2)) ** 0.5))
    r_h = r_h_num / r_h_denom

    r_v_num = ((eps * (np.cos(theta))) - (
                (eps - ((np.sin(theta)) ** 2)) ** 0.5))
    r_v_denom = ((eps * (np.cos(theta))) + (
                (eps - ((np.sin(theta)) ** 2)) ** 0.5))
    r_v = r_v_num / r_v_denom

    T_h = 1 + r_h
    T_v = 1 + r_v
    T_hm = 1 - r_h
    T_vm = 1 - r_v
    sq = ((eps - ((np.sin(theta)) ** 2)) ** 0.5)

    F_hh_1 = (((((np.sin(theta)) ** 2) / np.cos(theta)) - (sq)) * (T_h ** 2))
    F_hh_2 = (2 * ((np.sin(theta)) ** 2) * (
                (1 / np.cos(theta)) + (1 / sq)) * T_h * T_hm)
    F_hh_3 = (((((np.sin(theta)) ** 2) / np.cos(theta)) + (
                (1 + ((np.sin(theta)) ** 2)) / sq)) * (T_hm ** 2))

    F_hh = -1 * (F_hh_1 - F_hh_2 + F_hh_3)

    F_vv_1 = (((((np.sin(theta)) ** 2) / np.cos(theta)) - (sq / eps)) * (
                T_v ** 2))
    F_vv_2 = (2 * ((np.sin(theta)) ** 2) * (
                (1 / np.cos(theta)) + (1 / sq)) * T_v * T_vm)
    F_vv_3 = (((((np.sin(theta)) ** 2) / np.cos(theta)) + (
                (eps * (1 + ((np.sin(theta)) ** 2))) / sq)) * (T_vm ** 2))

    F_vv = F_vv_1 - F_vv_2 + F_vv_3

    f_hh = (-2 * r_h) / (np.cos(theta))
    f_vv = (2 * r_v) / (np.cos(theta))

    return r_h, r_v, f_hh, f_vv, F_hh, F_vv


def mixture(v1, v2, v3, c_0, d, eps_ice, eps_rock, eps_real):
    eps_helium_real = eps_real
    eps_helium_imag = (0.0013 * c_0 + 0.00119) * eps_real
    eps_helium = complex(eps_helium_real, eps_helium_imag)
    eps_s = np.exp((v1 * np.log(eps_ice)) + (v2 * np.log(eps_rock)) + (
                v3 * np.log(eps_helium)))
    return eps_s


def reg_dielec(ft, bd, lambda_wave):
    eps_real = 1.919 ** bd
    loss_tan = 10 ** ((0.038 * ft) + (0.312 * bd) - 3.260)
    eps_imag = loss_tan * eps_real
    eps = complex(eps_real, eps_imag)
    d_r = (np.sqrt(eps_real) * lambda_wave) / (2 * np.pi * eps_imag)
    return eps, eps_real, eps_imag, d_r


def rayleigh(eps_s, eps, eps_real, eps_imag, vf, k, r_s, d):
    eps_si = eps_s.imag
    k_i = k * np.sqrt(eps_imag)
    k_r = k * np.sqrt(eps_real)
    ka_1 = (3 * eps) / (eps_s + (2 * eps))
    ks_1 = (eps_s - eps) / (eps_s + (2 * eps))
    Nd = (3 * vf) / (4 * np.pi * (r_s ** 3))
    ka = (2 * k_i * (1 - vf)) + (
        vf * k_r * (eps_si / eps_real) * (ka_1 * np.conj(ka_1))
    )
    ks = (8 / 3) * np.pi * Nd * (k_r ** 4) * (r_s ** 6) * (ks_1 * np.conj(ks_1))
    ke = ka + ks
    a = ks / ke
    tau = ke * d

    return a, tau


def transmission(theta, eps):
    theta_t = np.arcsin(np.sin(theta) / (np.sqrt(eps)))
    T_hh = (2 * np.cos(theta)) / (
                np.cos(theta) + (np.sqrt(eps) * np.cos(theta_t)))
    T_vv = (2 * np.cos(theta)) / (
                np.cos(theta_t) + (np.sqrt(eps) * np.cos(theta)))
    sub_hh = (2 * np.sqrt(eps) * np.cos(theta_t)) / (
                np.cos(theta) + (np.sqrt(eps) * np.cos(theta_t)))
    sub_vv = (2 * np.sqrt(eps) * np.cos(theta_t)) / (
                np.cos(theta_t) + (np.sqrt(eps) * np.cos(theta)))
    return T_hh, T_vv, sub_hh, sub_vv, theta_t


def volume(a, theta, T_hh, T_vv, sub_hh, sub_vv, tau, theta_t):
    volume_hh = 0.5 * a * np.cos(theta) * T_hh * sub_hh * (
                1 - np.exp(((-1) * 2 * tau) / np.cos(theta_t))) * 1.5
    volume_vv = 0.5 * a * np.cos(theta) * T_vv * sub_vv * (
                1 - np.exp(((-1) * 2 * tau) / np.cos(theta_t))) * 1.5

    sigma_vol_hh = 10 * np.log10(volume_hh)
    sigma_vol_vv = 10 * np.log10(volume_vv)

    return sigma_vol_hh, sigma_vol_vv, volume_hh, volume_vv


def subsurface(theta, T_hh, sub_hh, T_vv, sub_vv, tau, theta_t, sigma_hh,
               sigma_vv):
    subsur_hh = (np.cos(theta) / np.cos(theta_t)) * T_hh * sub_hh * np.exp(
        ((-1) * 2 * tau) / np.cos(theta_t)) * sigma_hh
    subsur_vv = (np.cos(theta) / np.cos(theta_t)) * T_vv * sub_vv * np.exp(
        ((-1) * 2 * tau) / np.cos(theta_t)) * sigma_vv

    sigma_subsur_hh = 10 * np.log10(subsur_hh)
    sigma_subsur_vv = 10 * np.log10(subsur_vv)

    return sigma_subsur_hh, sigma_subsur_vv, subsur_hh, subsur_vv


def subsurface_volume(
        a, theta, eps_s, eps_sub, T_hh, T_vv, tau, theta_t, sigma_2, k
):
    R = (((np.sqrt(eps_sub) - np.sqrt(eps_s)) / (
                np.sqrt(eps_sub) + np.sqrt(eps_s))) ** 2)
    k_r = k * np.sqrt(eps_sub)
    l_r = np.exp(
        (-1) * (sigma_2 ** 2) * (k_r ** 2) * ((np.cos(theta_t)) ** 2))

    sub_vol_hh = a * np.cos(theta) * T_hh * T_hh * R * l_r * (
                tau / np.cos(theta_t)) * np.exp(
        ((-1) * 2 * tau) / np.cos(theta_t)) * 3.0
    sub_vol_vv = a * np.cos(theta) * T_vv * T_vv * R * l_r * (
                tau / np.cos(theta_t)) * np.exp(
        ((-1) * 2 * tau) / np.cos(theta_t)) * 3.0

    sigma_sub_vol_hh = 10 * np.log10(sub_vol_hh)
    sigma_sub_vol_vv = 10 * np.log10(sub_vol_vv)

    return sigma_sub_vol_hh, sigma_sub_vol_vv, sub_vol_hh, sub_vol_vv


def total(sigma_hh, sigma_vv, sigma_vol_hh, sigma_vol_vv, sigma_subsur_hh,
          sigma_subsur_vv, sigma_sub_vol_hh, sigma_sub_vol_vv):
    total_hh = (
        2 * sigma_sub_vol_hh) + sigma_subsur_hh + sigma_vol_hh + sigma_hh
    total_vv = (
        2 * sigma_sub_vol_vv
    ) + sigma_subsur_vv + sigma_vol_vv + sigma_vv
    sigma_total_hh = 10 * np.log10(total_hh)
    sigma_total_vv = 10 * np.log10(total_vv)

    return sigma_total_hh, sigma_total_vv


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    return arr.reshape(-1, la)


def sim_func(sigma, inc, eps, length, lambda_wave):
    _, _, f_hh, f_vv, F_hh, F_vv = coeffs(eps=eps, theta=inc)

    sigma_list = surface(
        theta=inc,
        length=length,
        sigma=sigma,
        f_hh=f_hh,
        f_vv=f_vv,
        F_hh=F_hh,
        F_vv=F_vv
    )

    return sigma_list[0], sigma_list[1]
