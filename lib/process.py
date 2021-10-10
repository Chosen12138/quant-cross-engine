
import logging
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from datetime import datetime

from lib.utils import *
from conf.path_config import *
from conf.wind_config import *

__all__ = ['OptionConfigProcess']


class OptionConfigProcess(object):

    def __init__(self, option_symbol: str = '000300.SH.CFFEX', file_type: str = 'bin'):
        option_symbol = '.'.join(option_symbol.split('.')[:-1])
        self.option_symbol = option_symbol
        if option_symbol not in wind_config.wind_option_listed_symbols_only:
            raise NotImplementedError('option symbol is not found in wind config')
        self.his_option_codes = pd.read_csv(os.path.join(OPTIONS.config_db_path, f"历史期权代码{self.option_symbol}.csv"), parse_dates=['listed_date', 'expire_date'])
        if len(self.his_option_codes) < 20:
            raise NotImplementedError('his option codes not enough')
        self.log_init()
        self.ZL_underlying_codes = self.getZLContracts()
        self.future_underlying_minute_data = {}
        self.finished_gears_management_codes = []
        self.file_type = file_type

    @property
    def exchange_mapper(self):
        return {'SHFE': 'SHFEX',
                'CFFEX': 'CFFEX',
                'CFFE': 'CFFEX',
                'CZCE': 'CZCE',
                'CZC': 'CZCE',
                'DCE': 'DCE',
                'INE': 'INE'}

    def log_init(self):
        log_fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        logging.basicConfig(format=log_fmt, level='INFO')

    def getTradeDates(self, exchange: str = 'CFFEX'):
        df_dates = pd.read_csv(os.path.join(FUTURES.config_db_path, f'{exchange}交易日历.csv'))
        return df_dates[f'{exchange}_trade_date'].tolist()

    def getZLContracts(self):
        #   期货期权合约的主力处理问题
        if self.option_symbol in ['000300.SH', '510050.SH', '510300.SH']:
            return None
        underlying_symbol = self.option_symbol
        exchange = wind_config.wind_option_code_exchange_mapper[underlying_symbol]
        exchange = self.exchange_mapper[exchange]
        file_path = os.path.join(FUTURES.config_db_path,
                                 f'历史信息{underlying_symbol}.{exchange}.csv')
        df = pd.read_csv(file_path)
        return df.TRADE_HISCODE.unique().tolist()

    def proTodayTradeOptions(self, trade_date):
        """预处理日志文件，序列化当日可交易期权信息"""
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, '%Y-%m-%d')

        all_trade_options_dict = {}
        zl_trade_options_dict = {}
        min_start_date = self.his_option_codes.listed_date.min()
        if trade_date < min_start_date:
            return None
        else:
            for index, row in self.his_option_codes.iterrows():
                if row.listed_date <= trade_date <= row.expire_date:

                    call_or_put = 'C' if row.call_or_put == '认购' else 'P'
                    time_to_expire = (row.expire_date - trade_date).days / 365
                    days_to_expire = (row.expire_date - trade_date).days

                    all_trade_options_dict[row.wind_code] = {'call_or_put': call_or_put,
                                                             'option_mark_code': row.option_mark_code,
                                                             'exercise_price': row.exercise_price,
                                                             'listed_date': row.listed_date.strftime('%Y-%m-%d'),
                                                             'expire_date': row.expire_date.strftime('%Y-%m-%d'),
                                                             'time_to_expire': time_to_expire,
                                                             'days_to_expire': days_to_expire,
                                                             'multiplier': row.contract_unit}

                    if self.ZL_underlying_codes is not None:
                        if row.option_mark_code in self.ZL_underlying_codes:
                            zl_trade_options_dict[row.wind_code] = {'call_or_put': call_or_put,
                                                                    'option_mark_code': row.option_mark_code,
                                                                    'exercise_price': row.exercise_price,
                                                                    'listed_date': row.listed_date.strftime('%Y-%m-%d'),
                                                                    'expire_date': row.expire_date.strftime('%Y-%m-%d'),
                                                                    'time_to_expire': time_to_expire,
                                                                    'days_to_expire': days_to_expire,
                                                                    'multiplier': row.contract_unit}

        return all_trade_options_dict, zl_trade_options_dict

    def main(self):
        trade_dates = self.getTradeDates()
        if not os.path.exists(os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权合约信息')):
            os.mkdir(os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权合约信息'))

        for trade_date in trade_dates:
            if trade_date > SUPPORT.config_version_date:
                break
            returns = self.proTodayTradeOptions(trade_date=trade_date)
            if returns is None:
                continue
            else:
                all_trade_options, zl_trade_options = returns
            out_json_file = os.path.join(OPTIONS.config_db_path,
                                         f'每日可交易{self.option_symbol}期权合约信息', f'{trade_date}.json')
            if os.path.exists(out_json_file):
                os.remove(out_json_file)
            if all_trade_options:
                if self.option_symbol in ['510050.SH', '000300.SH', '513000.SH']:
                    self.manage_today_minute_gears(trade_date=trade_date, all_trade_options=all_trade_options)
                else:
                    self.manage_today_minute_gears(trade_date=trade_date, all_trade_options=zl_trade_options)

            out = {trade_date: {'all_trade_options': all_trade_options,
                                'zl_trade_options': zl_trade_options}}
            with open(out_json_file, "w") as f:
                f.write(json.dumps(out, ensure_ascii=True, indent=4, separators=(',', ':')))
            logging.info(f'期权品种{self.option_symbol}于{trade_date}的可交易合约信息已序列化')
        logging.info(f'期权品种{self.option_symbol}所有可交易合约信息已存储')

    def queryTodayTradeOptions(self, trade_date):
        """索引指定交易日的可交易期权信息"""

        out_json_file = os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权合约信息', f'{trade_date}.json')
        if not os.path.exists(out_json_file):
            raise FileNotFoundError('当日可以交易期权合约不存在')
        with open(out_json_file, "r") as f:
            result = json.load(f)
        return result[trade_date]

    def manage_today_minute_gears(self, trade_date, all_trade_options):
        #
        # if self.option_symbol in ['000300.SH', '510050.SH', '510300.SH']:
        #     return

        underlying_symbol = self.option_symbol
        exchange = wind_config.wind_option_code_exchange_mapper[underlying_symbol]
        target_underlying_codes = [info['option_mark_code'] for code, info in all_trade_options.items()]
        target_underlying_codes = list(set(target_underlying_codes))
        target_underlying_codes.sort()

        for mark_code in target_underlying_codes:
            part_code = mark_code.split('.')[0]
            full_code = '.'.join([part_code, exchange])
            if full_code not in self.future_underlying_minute_data.keys():
                if full_code in ['510050.SH', '000300.SH', '513000.SH']:
                    parent_path = os.path.join(SPECIAL.db_path,
                                               full_code)
                else:
                    parent_path = os.path.join(FUTURES.source_db_path,
                                               underlying_symbol,
                                               full_code)
                actor = DataFrameActor(parent_path=parent_path,
                                       df=pd.DataFrame(),
                                       filename='MIN_01')
                try:
                    if self.file_type == 'bin':
                        df = actor.from_bin()
                    elif self.file_type == 'csv':
                        df = actor.from_csv(parse_date='DATETIME')
                    else:
                        df = actor.from_hdf(key='Bar')
                    df = df.drop_duplicates(subset='DATETIME')
                    df.DATETIME = pd.to_datetime(df.DATETIME)
                    df['DATE'] = df.DATETIME.apply(lambda r: r.strftime('%Y-%m-%d'))
                except Exception:
                    continue
                self.future_underlying_minute_data[mark_code] = df

        df_gears_all_mark_codes = []
        for mark_code in target_underlying_codes:
            trade_all_gears_config = []
            if mark_code not in self.future_underlying_minute_data.keys():
                continue
            spot_df = self.future_underlying_minute_data[mark_code]
            spot_df = spot_df[spot_df.DATE == trade_date]
            exercise_price_list = [float(i['exercise_price']) for i in all_trade_options.values()
                                   if i['option_mark_code'] == mark_code]
            exercise_price_list = list(set(exercise_price_list))
            exercise_price_list.sort()

            for code, info in all_trade_options.items():
                if info['option_mark_code'] == mark_code:
                    exercise_price = float(info['exercise_price'])
                    idx = exercise_price_list.index(exercise_price)
                    gears = []

                    for row_ in spot_df.itertuples():
                        spot_price = row_.CLOSE
                        at_money_price_idx = search_closest(exercise_price_list, spot_price)
                        if info['call_or_put'] == 'C':
                            if at_money_price_idx > idx:
                                gears.append(f'实{abs(at_money_price_idx - idx)}')
                            elif at_money_price_idx < idx:
                                gears.append(f'虚{abs(at_money_price_idx - idx)}')
                            else:
                                gears.append(f'平值')

                        else:
                            if at_money_price_idx > idx:
                                gears.append(f'虚{abs(at_money_price_idx - idx)}')
                            elif at_money_price_idx < idx:
                                gears.append(f'实{abs(at_money_price_idx - idx)}')
                            else:
                                gears.append(f'平值')
                    df_gears = pd.DataFrame(gears)
                    if len(df_gears) == 0:
                        continue
                    df_gears.columns = [code]
                    trade_all_gears_config.append(df_gears)
            if trade_all_gears_config:
                df_all_gears = pd.concat(trade_all_gears_config, axis=1)
                df_all_gears.index = spot_df.DATETIME
                df_gears_all_mark_codes.append(df_all_gears)
        if df_gears_all_mark_codes:
            df_all_out = pd.concat(df_gears_all_mark_codes, axis=1, join='outer')
            df_all_out = df_all_out.reset_index()
            if not os.path.exists(os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权gears信息')):
                os.mkdir(os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权gears信息'))
            parent_path = os.path.join(OPTIONS.config_db_path, f'每日可交易{self.option_symbol}期权gears信息')
            actor = DataFrameActor(parent_path=parent_path,
                                   df=df_all_out,
                                   filename=trade_date)
            actor.to_bin()


if __name__ == '__main__':

    target_option_symbols = (
        'M.DCE', 'I.DCE', 'P.DCE', 'PP.DCE', 'PG.DCE',
        'C.DCE', 'L.DCE', 'V.DCE', 'SC.INE', 'RU.SHF',
        'CU.SHF', 'AL.SHF', 'ZC.CZC', 'TA.CZC',
        'SR.CZC', 'RM.CZC', 'MA.CZC', 'CF.CZC',
        '510050.SH.SSE', '000300.SH.CFFEX'
    )

    def task(option_symbol):
        pro = OptionConfigProcess(option_symbol=option_symbol)
        pro.main()

    # task("510050.SH.SSE")
    task("000300.SH.CFFEX")
    # with ProcessPoolExecutor(max_workers=6) as executor:
    #     executor.map(task, target_option_symbols)
