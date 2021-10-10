from quant_xpress import *
from quant_xpress.ops import *
from quant_xpress.ops.ts_ops import *
from quant_xpress.ops.tech_ops import *

__all__ = ['VISUAL_A']


class VISUAL_A(GraphDef):

    [OPEN, HIGH, LOW, CLOSE, VOLUME] = features('OPEN HIGH LOW CLOSE VOLUME')

    _LAST_CLOSE = ts_delay(CLOSE, d=1)
    _LAST_OPEN = ts_delay(OPEN, d=1)
    _MA60 = ts_mean(CLOSE, d=60)
    _MA90 = ts_mean(CLOSE, d=90)
    SPEED60 = (_MA60 - ts_delay(_MA60, d=7))/_MA60 * 1000
    SPEED90 = (_MA90 - ts_delay(_MA90, d=7))/_MA90 * 1000

    _TYP = (HIGH + LOW + CLOSE) / 3
    MA55 = ts_mean(CLOSE, d=55)

    # BOLL_UP = ts_mean(_TYP, d=21) + 2 * ts_std(_TYP, d=21)
    # BOLL_DOWN = ts_mean(_TYP, d=21) - 2 * ts_std(_TYP, d=21)

    S_IND = ts_sum(where(CLOSE > OPEN, CLOSE - OPEN, 0), d=15) / ts_sum(abs(CLOSE - OPEN), d=15)
    ATR21 = ATR(CLOSE, HIGH, LOW, d=21)

    # where((CLOSE > OPEN) & (LAST_CLOSE < LAST_OPEN), )
