import os
from datetime import datetime

__all__ = ["SUPPORT", "LOG", "OPTIONS", "FUTURES", "SPECIAL"]


class SUPPORT:
    """引擎支持项"""

    #   支持的期权合约
    option_symbols = []
    #   支持的期货合约
    future_symbols = ["IF", "IH", "IC"]
    #   交易日历/合约映射/费用信息等更新日
    config_version_date = "2021-09-24"
    #   数据最早开始日期
    data_start_date = "2016-01-01"
    #   数据最新结束日期
    data_end_date = "2021-09-24"
    #   项目路径
    db_path = "/home/jay/data/DataBase"
    project_path = "/home/jay/桌面/quant-pro-engine"


class LOG:
    log_path = "/home/jay/桌面/yaf_log"
    #   log_path = 'D:/yaf_engine_log'
    today_log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d"))


class OPTIONS:
    """
    期权数据库的路径配置
    """

    db_path = os.path.join(SUPPORT.db_path, "期权数据")
    compress_db_path = os.path.join(db_path, "compress_data")
    source_db_path = os.path.join(db_path, "source_data")
    config_db_path = os.path.join(db_path, "config_data", SUPPORT.config_version_date)
    factor_db_path = os.path.join(db_path, "factor_data")


class FUTURES:
    """
    期货数据库的路径配置
    """

    db_path = os.path.join(SUPPORT.db_path, "期货数据")
    compress_db_path = os.path.join(db_path, "compress_data")
    source_db_path = os.path.join(db_path, "source_data")
    config_db_path = os.path.join(db_path, "config_data", SUPPORT.config_version_date)
    factor_db_path = os.path.join(db_path, "factor_data")


class SPECIAL:
    db_path = os.path.join(SUPPORT.db_path, "特殊数据")
