__all__ = ["wind_config"]

wind_future_listed_symbols = [
    "AG.SHF",
    "AL.SHF",
    "AU.SHF",
    "BU.SHF",
    "CU.SHF",
    "FU.SHF",
    "HC.SHF",
    "NI.SHF",
    "PB.SHF",
    "RB.SHF",
    "RU.SHF",
    "SN.SHF",
    "SP.SHF",
    "SS.SHF",
    "ZN.SHF",
    "WR.SHF",
    "BC.INE",
    "LU.INE",
    "NR.INE",
    "SC.INE",
    "A.DCE",
    "B.DCE",
    "BB.DCE",
    "C.DCE",
    "CS.DCE",
    "EB.DCE",
    "EG.DCE",
    "FB.DCE",
    "I.DCE",
    "J.DCE",
    "JD.DCE",
    "JM.DCE",
    "L.DCE",
    "LH.DCE",
    "M.DCE",
    "P.DCE",
    "PG.DCE",
    "PP.DCE",
    "RR.DCE",
    "V.DCE",
    "Y.DCE",
    "AP.CZC",
    "CF.CZC",
    "CJ.CZC",
    "CY.CZC",
    "FG.CZC",
    "JR.CZC",
    "LR.CZC",
    "MA.CZC",
    "OI.CZC",
    "PF.CZC",
    "PK.CZC",
    "PM.CZC",
    "RI.CZC",
    "RM.CZC",
    "RS.CZC",
    "SA.CZC",
    "SF.CZC",
    "SM.CZC",
    "SR.CZC",
    "TA.CZC",
    "UR.CZC",
    "WH.CZC",
    "ZC.CZC",
    "T.CFE",
    "TF.CFE",
    "TS.CFE",
    "IF.CFE",
    "IC.CFE",
    "IH.CFE",
]

wind_option_listed_symbols = [
    ("SH", "000300.SH"),
    ("SH", "510050.SH"),
    ("SH", "510300.SH"),
    ("DCE", "M"),
    ("DCE", "I"),
    ("DCE", "P"),
    ("DCE", "PP"),
    ("DCE", "PG"),
    ("DCE", "C"),
    ("DCE", "L"),
    ("DCE", "V"),
    ("INE", "SC"),
    ("SHFE", "ZN"),
    ("SHFE", "RU"),
    ("SHFE", "AU"),
    ("SHFE", "CU"),
    ("SHFE", "AL"),
    ("CZCE", "ZC"),
    ("CZCE", "TA"),
    ("CZCE", "SR"),
    ("CZCE", "RM"),
    ("CZCE", "MA"),
    ("CZCE", "CF"),
]


class WindConfig(object):
    def __init__(self):

        self.wind_future_listed_symbols = wind_future_listed_symbols
        self.wind_option_listed_symbols = wind_option_listed_symbols
        self.wind_option_listed_symbols_only = [
            i[1] for i in self.wind_option_listed_symbols
        ]
        self.wind_option_code_exchange_mapper = dict(
            [i[1], i[0]] for i in self.wind_option_listed_symbols
        )
        self.wind_future_listed_codes = []
        for symbol in self.wind_future_listed_symbols:
            if symbol.split(".")[-1] == "CFE":
                code = symbol.split(".")[0] + ".CFFEX"
            elif symbol.split(".")[-1] == "CZC":
                code = symbol.split(".")[0] + ".CZCE"
            elif symbol.split(".")[-1] == "SHF":
                code = symbol.split(".")[0] + ".SHFEX"
            else:
                code = symbol
            self.wind_future_listed_codes.append(code)
        self.WIND_VIRTUAL_CODE_MAPPER = dict(
            zip(self.wind_future_listed_codes, self.wind_future_listed_symbols)
        )
        self.WIND_EXCHANGE_MAPPER = {
            "CFFEX": "CFFEX",
            "SHFEX": "SHFE",
            "CZCE": "CZCE",
            "DCE": "DCE",
            "INE": "INE",
        }


wind_config = WindConfig()
