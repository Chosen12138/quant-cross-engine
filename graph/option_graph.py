from quant_xpress import *
from quant_xpress.ops import *
from quant_xpress.ops.basic_ops import *
from math import pi
from quant_xpress.ops.ts_ops import *
from quant_xpress.ops.tech_ops import *

__all__ = ['BSM', 'B76']


class BSM(GraphDef):

    [P, S, SIGMA, T, R, X, Q, SIDE, MODE] = features('P S SIGMA T R X Q SIDE MODE')

    #   BSM欧式期权解析解
    _d1 = (log(S/X) + (R - Q + 0.5 * (SIGMA ** 2))*T)/(SIGMA * sqrt(T))
    _d2 = _d1 - SIGMA * sqrt(T)
    _N_d1 = norm_cdf(_d1)
    _N_d2 = norm_cdf(_d2)
    _N_d1_ = norm_cdf(-_d1)
    _N_d2_ = norm_cdf(-_d2)
    _D_N_d1 = (1/(sqrt(2 * pi))) * exp(-_d1 ** 2 / 2)
    _D_N_d2 = (1/(sqrt(2 * pi))) * exp(-_d2 ** 2 / 2)

    _DELTA_CALL = exp(-Q * T) * _N_d1
    _GAMMA_CALL = (exp(-Q * T) * _D_N_d1)/(S * SIGMA * sqrt(T))
    _VEGA_CALL = S * exp(-Q * T) * _D_N_d1 * sqrt(T) * 0.01
    _THETA_CALL = ((-S * exp(-Q * T) * _D_N_d1 * SIGMA)/(2 * sqrt(T)) +
                    Q * S * exp(-Q * T) * _N_d1 - R * X * exp(-R * T) * _N_d2) / 365
    _RHO_CALL = T * X * exp(-R * T) * _N_d2 * 0.01

    _DELTA_PUT = exp(-Q * T) * (_N_d1 - 1)
    _GAMMA_PUT = (exp(-Q * T) * _D_N_d1)/(S * SIGMA * sqrt(T))
    _VEGA_PUT = S * exp(-Q * T) * _D_N_d1 * sqrt(T) * 0.01
    _THETA_PUT = ((-S * exp(-Q * T) * _D_N_d1 * SIGMA)/(2 * sqrt(T)) -
                 Q * S * exp(-Q * T) * _N_d1_ + R * X * exp(-R * T) * _N_d2_) / 365
    _RHO_PUT = -T * X * exp(-R * T) * _N_d2_ * 0.01

    DELTA = where(SIDE == 1, _DELTA_CALL, _DELTA_PUT)
    GAMMA = where(SIDE == 1, _GAMMA_CALL, _GAMMA_PUT)
    VEGA = where(SIDE == 1, _VEGA_CALL, _VEGA_PUT)
    RHO = where(SIDE == 1, _RHO_CALL, _RHO_PUT)
    THETA = where(SIDE == 1, _THETA_CALL, _THETA_PUT)


class B76(GraphDef):
    [P, S, SIGMA, T, R, X, Q, SIDE, MODE] = features('P S SIGMA T R X Q SIDE MODE')
    #   BL76欧式期权解析解
    _d1 = (log(S/X) + 0.5 * T * (SIGMA ** 2))/(SIGMA * sqrt(T))
    _d2 = _d1 - SIGMA * sqrt(T)

    _N_d1 = norm_cdf(_d1)
    _N_d2 = norm_cdf(_d2)
    _PN_d1 = norm_pdf(_d1)
    _PN_d2 = norm_pdf(_d2)

    _PREMIUM_CALL = S * exp(-R * T) * _N_d1 - X * exp(-R * T) * _N_d2
    _DELTA_CALL = exp(-R * T) * _N_d1
    _GAMMA_CALL = (exp(-R * T) * _PN_d1)/(S * SIGMA * sqrt(T))
    _VEGA_CALL = S * exp(-Q * T) * _PN_d1 * sqrt(T) * 0.01
    _THETA_CALL = (-R * _PREMIUM_CALL + S * _PN_d1 * SIGMA * exp(-R * T) / (2 * sqrt(T)))/365
    _RHO_CALL = T * X * exp(-R * T) * _N_d2 * 0.01

    _PREMIUM_PUT = -S * exp(-R * T) * (1-_N_d1) + X * exp(-R * T) * (1-_N_d2)
    _DELTA_PUT = -exp(-R * T) * (1-_N_d1)
    _GAMMA_PUT = (exp(-R * T) * _PN_d1)/(S * SIGMA * sqrt(T))
    _VEGA_PUT = S * exp(-R * T) * _PN_d1 * sqrt(T) * 0.01
    _THETA_PUT = (-R * _PREMIUM_PUT + S * _PN_d1 * SIGMA * exp(-R * T) / (2 * sqrt(T)))/365
    _RHO_PUT = -T * X * exp(-R * T) * (1-_N_d2) * 0.01

    PREMIUM = where(SIDE == 1, _PREMIUM_CALL, _PREMIUM_PUT)
    DELTA = where(SIDE == 1, _DELTA_CALL, _DELTA_PUT)
    GAMMA = where(SIDE == 1, _GAMMA_CALL, _GAMMA_PUT)
    VEGA = where(SIDE == 1, _VEGA_CALL, _VEGA_PUT)
    RHO = where(SIDE == 1, _RHO_CALL, _RHO_PUT)
    THETA = where(SIDE == 1, _THETA_CALL, _THETA_PUT)
