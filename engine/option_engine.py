import json
import os.path
from typing import *
from datetime import datetime, time, timedelta

import numpy as np
import pandas.errors
from matplotlib import pyplot as plt, ticker
from openpyxl.utils.dataframe import dataframe_to_rows

from conf.fee_config import OptionTradeConfig
from conf.path_config import SUPPORT, FUTURES, OPTIONS, LOG

import logging
import pandas as pd


from graph import *
from lib import *


class CrossTraderActor(object):

    def __init__(self,
                 trade_dates,
                 logger,
                 option_order_tasks: Dict[str, type],
                 fixed_order_money=50_000,
                 initial_cash=1_000_000,
                 trading_time=time(21, 0, 0),
                 trading_minute_range=5,
                 trade_side_rule=('long_call', 'long_put')):
        self.option_order_tasks = option_order_tasks
        if len(self.option_order_tasks) < 20:
            raise NotImplementedError('tasks not enough')
        self.option_minute_bar_data = {}
        self.option_day_bar_data = {}
        self.trade_dates = trade_dates
        self.trading_time = trading_time
        self.trading_minute_range = trading_minute_range
        self.portfolio_change_dates = [dt for dt in self.option_order_tasks.keys()]
        self.portfolio_change_dates = list(set(self.portfolio_change_dates))
        self.portfolio_change_dates.sort()
        self.option_code_symbol_mapper = {}
        self.holding = {}
        self.profit_record = {}
        self.long_call_profit_record = {}
        self.long_put_profit_record = {}
        self.option_code_settle_record = {}
        self.cash = initial_cash
        self.init_equity = initial_cash
        self.equity = initial_cash
        self.fixed_order_money = fixed_order_money
        self.logger = logger
        self.finished_order_task_record = {}
        self.holding_record = {}
        self.trade_side_rule = trade_side_rule
        self.symbol_profit = {}
        for dt, tasks in self.option_order_tasks.items():
            for task in tasks:
                if task.symbol not in self.finished_order_task_record.keys():
                    if task.symbol == 'IH':
                        symbol = '510050.SH'
                    elif task.symbol == 'IF':
                        symbol = '000300.SH'
                    else:
                        symbol = task.symbol
                    self.finished_order_task_record[symbol] = []

    def query_option_bar_data_by_code(self, symbol, option_code, frequency='MIN_01', file_type='bin'):
        exchange_mapper = {
            'CU': 'SHFE',
            'RU': 'SHFE',
            'AL': 'SHFE',
            'I': 'DCE',
            'V': 'DCE',
            'C': 'DCE',
            'M': 'DCE',
            'PP': 'DCE',
            'SR': 'CZCE',
            'CF': 'CZCE',
            'TA': 'CZCE',
            'MA': 'CZCE',
            'ZC': 'CZCE',
            'SC': 'INE',
            '510050.SH': 'SH',
            '000300.SH': 'CFFEX'}

        if symbol == 'IH':
            symbol = '510050.SH'
        elif symbol == 'IF':
            symbol = '000300.SH'
        exchange = exchange_mapper[symbol]
        parent_path = os.path.join(OPTIONS.source_db_path,
                                   symbol,
                                   f'{option_code}.{exchange}')
        actor = DataFrameActor(parent_path=parent_path,
                               df=pd.DataFrame(),
                               filename=frequency)
        if file_type == 'bin':
            df = actor.from_bin()
        else:
            df = actor.from_csv(parse_date='DATETIME')
        df.DATETIME = pd.to_datetime(df.DATETIME)
        df['YESTERDAY_CLOSE'] = df.CLOSE.shift()
        return df

    def run(self):
        #   启动回测
        for trade_date in self.trade_dates:
            #   先15:00结算, 后进入换仓日夜盘或者后一交易日盘
            trade_timestamp = pd.Timestamp(trade_date)
            dt = trade_date.strftime('%Y-%m-%d')
            self.daily_settle(trade_date=trade_date)
            self.profit_record[dt] = round(self.equity - self.init_equity, 2)
            self.holding_record[dt] = self.holding.__str__()
            self.logger.info(f'{dt} 结算完成, 最新累计盈亏{round(self.equity - self.init_equity, 2)}, 最新持仓{self.holding}')

            #   索引到真实换仓日
            if trade_date in self.portfolio_change_dates:
                #   如果有持仓, 先进行平仓
                if self.holding:
                    for option_code, amount in self.holding.copy().items():
                        df_minute_bar = self.option_minute_bar_data[option_code]
                        price_list = []
                        deal_price = np.nan
                        deal_time = np.nan
                        for row_ in df_minute_bar.itertuples():
                            if row_.DATETIME > trade_timestamp + timedelta(hours=self.trading_time.hour,
                                                                           minutes=self.trading_time.minute):
                                price_list.append(row_.CLOSE)
                            if len(price_list) >= self.trading_minute_range and row_.DATETIME > trade_timestamp + timedelta(hours=self.trading_time.hour,
                                                                           minutes=self.trading_time.minute) + timedelta(
                                    minutes=self.trading_minute_range):
                                deal_price = np.nanmean(price_list)
                                deal_time = row_.DATETIME
                                break
                        if np.isnan(deal_price):
                            df_day_bar = self.option_day_bar_data[option_code]
                            try:
                                deal_price = df_day_bar[df_day_bar.DATETIME == trade_timestamp].CLOSE.iloc[0]
                            except Exception:
                                deal_price = self.option_code_settle_record[option_code]
                            deal_time = trade_timestamp + timedelta(hours=15)
                        if amount > 0:
                            self.record_order(option_code=option_code,
                                              order_amount=abs(amount),
                                              order_side='cover_long',
                                              deal_price=deal_price,
                                              deal_time=deal_time)
                        else:
                            self.record_order(option_code=option_code,
                                              order_amount=abs(amount),
                                              order_side='cover_short',
                                              deal_price=deal_price,
                                              deal_time=deal_time)
                        self.option_minute_bar_data.pop(option_code)
                        self.option_day_bar_data.pop(option_code)

                order_tasks = self.option_order_tasks[trade_timestamp]
                #   如果存在订单任务
                if order_tasks:
                    #   索引数据
                    for order_task in order_tasks:
                        if order_task.order_side not in self.trade_side_rule:
                            continue
                        self.option_code_symbol_mapper[order_task.option_code] = order_task.symbol
                        if order_task.option_code not in self.option_day_bar_data.keys():
                            self.option_day_bar_data[order_task.option_code] = self.query_option_bar_data_by_code(
                                symbol=order_task.symbol,
                                option_code=order_task.option_code,
                                frequency='DAY'
                            )
                        if order_task.option_code not in self.option_minute_bar_data.keys():
                            self.option_minute_bar_data[order_task.option_code] = self.query_option_bar_data_by_code(
                                symbol=order_task.symbol,
                                option_code=order_task.option_code,
                                frequency='MIN_01'
                            )
                        df_minute_bar = self.option_minute_bar_data[order_task.option_code]
                        price_list = []
                        deal_price = np.nan
                        deal_time = np.nan
                        for row_ in df_minute_bar.itertuples():
                            if row_.DATETIME > trade_timestamp + timedelta(hours=self.trading_time.hour,
                                                                           minutes=self.trading_time.minute):
                                price_list.append(row_.CLOSE)

                            if len(price_list) >= self.trading_minute_range and row_.DATETIME > trade_timestamp + timedelta(hours=self.trading_time.hour,
                                                                           minutes=self.trading_time.minute) + timedelta(minutes=self.trading_minute_range):
                                deal_price = np.nanmean(price_list)
                                deal_time = row_.DATETIME
                                break

                        if np.isnan(deal_price):
                            continue
                            # df_day_bar = self.option_day_bar_data[order_task.option_code]
                            # deal_price = df_day_bar[df_day_bar.DATETIME == trade_timestamp].CLOSE.iloc[0]
                        if order_task.order_side in ['long_call', 'long_put']:
                            self.record_order(option_code=order_task.option_code,
                                              order_money=self.fixed_order_money,
                                              order_amount=None,
                                              order_side='open_long',
                                              deal_price=deal_price,
                                              deal_time=deal_time)
                        else:
                            self.record_order(option_code=order_task.option_code,
                                              order_money=self.fixed_order_money,
                                              order_amount=None,
                                              order_side='open_short',
                                              deal_price=deal_price,
                                              deal_time=deal_time)

    def record_order(self, option_code, order_amount, order_side, deal_price, deal_time, order_money: Optional[float] = None):
        symbol = self.option_code_symbol_mapper[option_code]
        if symbol == 'IH':
            symbol = '510050.SH'
        elif symbol == 'IF':
            symbol = '000300.SH'
        if order_money is not None and order_amount is None:
            order_amount = round(order_money/(deal_price * OptionTradeConfig.multiplier_table[symbol]))
        if order_amount <= 0:
            return
        fee = OptionTradeConfig.fee_fixed_table[symbol] * order_amount
        if order_side == 'open_long':
            if option_code not in self.holding.keys():
                self.holding[option_code] = order_amount
            else:
                self.holding[option_code] = self.holding[option_code] + order_amount
            cash_change = order_amount * deal_price * OptionTradeConfig.multiplier_table[symbol]
            self.cash = self.cash - cash_change
            self.finished_order_task_record[symbol].append(FinishedOptionOrderTask(option_code=option_code,
                                                                                   symbol=symbol,
                                                                                   open_time=deal_time,
                                                                                   open_price=deal_price,
                                                                                   deal_amount=order_amount,
                                                                                   ))
            # self.logger.info(f'{deal_time} 开仓买方合约{option_code}, 价格{deal_price}, 可用现金减少{cash_change}')
        elif order_side == 'open_short':
            if option_code not in self.holding.keys():
                self.holding[option_code] = -order_amount
            else:
                self.holding[option_code] = self.holding[option_code] - order_amount
            self.cash = self.cash + order_amount * deal_price * OptionTradeConfig.multiplier_table[symbol]
        elif order_side == 'cover_long':
            if option_code in self.holding.keys():
                self.holding[option_code] = self.holding[option_code] - order_amount
                if self.holding[option_code] <= 0:
                    self.holding.pop(option_code)
            cash_change = order_amount * deal_price * OptionTradeConfig.multiplier_table[symbol]
            self.cash = self.cash + cash_change
            for finished_order_task in self.finished_order_task_record[symbol]:
                if finished_order_task.option_code == option_code and finished_order_task.cover_time is None:
                    finished_order_task.cover_time = deal_time
                    finished_order_task.cover_price = deal_price
                    finished_order_task.order_side = order_side
                    finished_order_task.fee = fee * 2
                    self.logger.info(f'{deal_time} 平仓买方合约{option_code}, 完成订单{finished_order_task}')
            # self.logger.info(f'{deal_time} 平仓买方合约{option_code}, 价格{deal_price}, 可用现金增加{cash_change}')
        else:
            if option_code in self.holding.keys():
                self.holding[option_code] = self.holding[option_code] + order_amount
                if self.holding[option_code] >= 0:
                    self.holding.pop(option_code)
            self.cash = self.cash - order_amount * deal_price * OptionTradeConfig.multiplier_table[symbol]
        self.cash = self.cash - fee * 2
        self.option_code_settle_record[option_code] = deal_price

    def daily_settle(self, trade_date):
        if self.holding:
            market_value = 0
            holding_recent_price = {}
            for option_code, amount in self.holding.items():
                symbol = self.option_code_symbol_mapper[option_code]
                if symbol == 'IH':
                    symbol = '510050.SH'
                elif symbol == 'IF':
                    symbol = '000300.SH'

                df_day_bar = self.option_day_bar_data[option_code]
                #   部分合约数据缺失
                try:
                    price = df_day_bar[df_day_bar.DATETIME == pd.Timestamp(trade_date)].CLOSE.iloc[0]
                except Exception:
                    price = self.option_code_settle_record[option_code]
                self.option_code_settle_record[option_code] = price
                holding_recent_price[option_code] = price
                market_value = market_value + amount * price * OptionTradeConfig.multiplier_table[symbol]
            self.equity = self.cash + market_value
            self.logger.info(f'当日结算, 持仓合约的结算价格 {holding_recent_price}')

    def plot_performance(self, show=True, save_path: Optional[str] = None):
        import seaborn as sns
        sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 1.7})
        fig, ax_arr = plt.subplots(2, 1, figsize=(12, 9))
        df_holding = pd.DataFrame(data=self.holding_record.values(),
                                  index=self.holding_record.keys(),
                                  columns=['Holding'])
        df = pd.DataFrame(data=self.symbol_profit.values(),
                          index=self.symbol_profit.keys())
        df = df.reset_index()
        df.columns = ['Symbol', 'CumProfit']
        df = df.sort_values(by='CumProfit', ascending=False)
        ax_arr[0].set_title("Equity Performance")
        sns.barplot(x=df['Symbol'],
                    y=df['CumProfit'],
                    ax=ax_arr[0])
        df = pd.DataFrame(data=self.profit_record.values(),
                          index=self.profit_record.keys(),
                          columns=['CumProfit'])
        sns.lineplot(data=df["CumProfit"], color="orange", ax=ax_arr[1])
        ax_arr[1].xaxis.set_major_locator(ticker.MultipleLocator(base=20))
        plt.xticks(rotation=90)
        sns.set_style('darkgrid')
        if save_path is not None:
            plt.savefig(save_path)
            df_all = pd.concat([df, df_holding], axis=1)
            df_all.to_excel(f'{save_path}权益与持仓记录.xlsx')
        if show:
            plt.show()
        return

    def analysis(self, save_path: Optional[str] = None):
        from openpyxl import Workbook
        # 创建一个工作簿对象
        wb = Workbook()
        for symbol, finished_order_tasks in self.finished_order_task_record.items():
            profit_list = []
            index_list = []
            details_list = []
            if finished_order_tasks:
                for finished_order_task in finished_order_tasks:
                    if finished_order_task.cover_time is not None:
                        index_list.append(finished_order_task.cover_time)
                        profit_list.append(finished_order_task.profit)
                        details_list.append(finished_order_task.__str__())
            df_symbol_profit = pd.DataFrame(data=profit_list,
                                            index=index_list,
                                            columns=['逐笔收益'])
            df_symbol_profit.index.name = '平仓时间'
            df_symbol_profit['累积收益'] = df_symbol_profit['逐笔收益'].cumsum()
            df_symbol_profit['交易详情'] = details_list
            ws_symbol = wb.create_sheet(symbol, 0)
            for row in dataframe_to_rows(df_symbol_profit,
                                         index=True,
                                         header=True):
                ws_symbol.append(row)
            self.symbol_profit[symbol] = sum(profit_list)
            self.logger.info(f'收益分析: 品总{symbol} 累计盈利{sum(profit_list)}')
        wb.save(f'{save_path}收益明细.xlsx')
        self.plot_performance(save_path=save_path)


class OptionCrossBackTestEngine(object):

    option_symbols: Tuple[str] = ('CU', 'AL', 'RU',
                                  'I', 'V', 'C', 'M', 'PP',
                                  'MA', 'CF', 'SR', 'TA', 'ZC',
                                  '000300.SH', '510050.SH')

    minimum_days_to_expire = 7

    def __init__(self,
                 strategy_name: Union[str] = 'CrossHedge',
                 frequency: Union[str] = 'DAY',
                 adjust_rule: Union[Tuple[str, time]] = ('Friday', time(21, 5, 0)),
                 trading_time_range: Union[int] = 2,
                 start_date: Union[str, datetime] = '2020-01-01',
                 end_date: Union[str, datetime] = SUPPORT.data_end_date,
                 option_symbols: Optional[Tuple[str]] = None,
                 ):
        if option_symbols is not None:
            self.option_symbols = option_symbols
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.adjust_rule = adjust_rule
        self.trading_time_range = trading_time_range
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        if frequency not in ['HOU_01', 'DAY']:
            raise ValueError('frequency is not supported')
        self.future_index_data = pd.DataFrame()
        self.future_index_factors = pd.DataFrame()
        self.long_call_target = pd.DataFrame()
        self.long_put_target = pd.DataFrame()
        self.short_put_target = pd.DataFrame()
        self.short_call_target = pd.DataFrame()
        self.trade_dates = pd.DataFrame()
        self.option_order_tasks = {}

        self.log_time_version = None
        self.log_file_path = None
        self.make_log_path()

    @property
    def option_future_exchange_mapper(self):
        return {'CU': ('CU', 'SHFE'),
                'RU': ('RU', 'SHFE'),
                'AL': ('AL', 'SHFE'),
                'I': ('I', 'DCE'),
                'V': ('V', 'DCE'),
                'C': ('C', 'DCE'),
                'M': ('M', 'DCE'),
                'PP': ('PP', 'DCE'),
                'SR': ('SR', 'CZCE'),
                'CF': ('CF', 'CZCE'),
                'TA': ('TA', 'CZCE'),
                'MA': ('MA', 'CZCE'),
                'ZC': ('ZC', 'CZCE'),
                'SC': ('SC', 'INE'),
                '510050.SH': ('IH', 'CFFEX'),
                '000300.SH': ('IF', 'CFFEX')}

    @property
    def trade_side_mapper(self):
        return {'long_call': 'C',
                'short_call': 'C',
                'long_put': 'P',
                'short_put': 'P'}

    def make_log_path(self):
        """创建日志记录文件"""
        if not os.path.exists(LOG.today_log_path):
            os.mkdir(LOG.today_log_path)
        if not os.path.exists(os.path.join(LOG.today_log_path, self.strategy_name)):
            os.mkdir(os.path.join(LOG.today_log_path, self.strategy_name))
        self.log_time_version = datetime.now().time().strftime('%H:%M:%S')
        self.log_file_path = os.path.join(LOG.today_log_path, self.strategy_name, self.log_time_version)
        os.mkdir(self.log_file_path)

        file_name = os.path.join(LOG.today_log_path, self.strategy_name, self.log_time_version, 'CrossEngineLog.txt')
        logging.basicConfig(
            filename=file_name,
            filemode="w",
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level="INFO",
        )
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(
            fmt=logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        self.console_logger = logging.getLogger()
        self.console_logger.addHandler(self.ch)

    def load_future_index_data(self, index_type: str = '指数'):
        df_future_list = []
        for symbol in self.option_symbols:
            future_symbol = self.option_future_exchange_mapper[symbol][0]
            future_exchange = self.option_future_exchange_mapper[symbol][1]
            parent_path = os.path.join(FUTURES.source_db_path,
                                       future_symbol,
                                       f'{future_symbol}{index_type}.{future_exchange}')
            actor = DataFrameActor(parent_path=parent_path,
                                   df=pd.DataFrame(),
                                   filename=self.frequency)
            df = actor.from_bin()
            df = df[['DATETIME', 'CODE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            df_future_list.append(df)
        self.future_index_data = pd.concat(df_future_list, axis=0, join='inner')
        self.future_index_data.DATETIME = pd.to_datetime(self.future_index_data.DATETIME)
        self.future_index_data = self.future_index_data.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.console_logger.info(f'载入各品种期货{index_type}{self.frequency}频率数据')

    def calculate_future_index_factor(self):
        self.future_index_factors = calculate_factors_using_graph(df_bar=self.future_index_data,
                                                                  factor_graph=FACTOR_CROSS,
                                                                  merge_org=True)
        self.future_index_factors = self.future_index_factors[(self.future_index_factors.DATETIME >= self.start_date) &
                                                              (self.future_index_factors.DATETIME <= self.end_date)]
        self.future_index_factors = self.future_index_factors.reset_index(drop=True)

    def cross_filter(self,
                     factor_data,
                     factor_name,
                     num=2,
                     top_or_bottom='top'):
        data = factor_data.copy()
        if data.index.names != [None]:
            data = data.reset_index()

        if factor_name not in data.columns:
            self.console_logger.info('因子不存在')
            exit(0)
        data = data.sort_values(by=['DATETIME', factor_name],
                                ascending=False)
        if top_or_bottom == 'bottom':
            index = data.groupby(by='DATETIME')[factor_name].tail(num).index
        else:
            index = data.groupby(by='DATETIME')[factor_name].head(num).index
        data = data.loc[index]
        data = data.set_index(['DATETIME', 'CODE'])
        data = data[~data.index.duplicated()]
        data = data.reset_index()
        return data

    def auto_filter_futures_model2(self):
        """SPEED30顺势"""
        long_call_target = self.cross_filter(factor_data=self.future_index_factors,
                                             factor_name='SPEED30',
                                             num=3,
                                             top_or_bottom='top')
        long_put_target = self.cross_filter(factor_data=self.future_index_factors,
                                            factor_name='SPEED30',
                                            num=3,
                                            top_or_bottom='bottom')
        self.long_call_target = long_call_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_put_target = long_put_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_call_target = self.long_call_target.reset_index(drop=True)
        self.long_put_target = self.long_put_target.reset_index(drop=True)
        self.long_call_target['FutureSymbol'] = self.long_call_target.CODE.apply(lambda r: get_symbol_prefix(r))
        self.long_put_target['FutureSymbol'] = self.long_put_target.CODE.apply(lambda r: get_symbol_prefix(r))
        del long_call_target, long_put_target

    def auto_filter_futures_model3(self):
        """SPEED30逆势"""
        long_call_target = self.cross_filter(factor_data=self.future_index_factors,
                                             factor_name='SPEED30',
                                             num=4,
                                             top_or_bottom='bottom')
        long_put_target = self.cross_filter(factor_data=self.future_index_factors,
                                            factor_name='SPEED30',
                                            num=4,
                                            top_or_bottom='top')
        self.long_call_target = long_call_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_put_target = long_put_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_call_target = self.long_call_target.reset_index(drop=True)
        self.long_put_target = self.long_put_target.reset_index(drop=True)
        self.long_call_target['FutureSymbol'] = self.long_call_target.CODE.apply(lambda r: get_symbol_prefix(r))
        self.long_put_target['FutureSymbol'] = self.long_put_target.CODE.apply(lambda r: get_symbol_prefix(r))
        del long_call_target, long_put_target

    def auto_filter_futures_score(self, num=3):
        """逆势"""
        long_call_target = self.cross_filter(factor_data=self.future_index_factors,
                                             factor_name='SCORE',
                                             num=3,
                                             top_or_bottom='bottom')
        long_put_target = self.cross_filter(factor_data=self.future_index_factors,
                                            factor_name='SCORE',
                                            num=3,
                                            top_or_bottom='top')
        self.long_call_target = long_call_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_put_target = long_put_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_call_target = self.long_call_target.reset_index(drop=True)
        self.long_put_target = self.long_put_target.reset_index(drop=True)
        self.long_call_target['FutureSymbol'] = self.long_call_target.CODE.apply(lambda r: get_symbol_prefix(r))
        self.long_put_target['FutureSymbol'] = self.long_put_target.CODE.apply(lambda r: get_symbol_prefix(r))
        del long_call_target, long_put_target

    def auto_filter_futures(self):
        """8品种多空"""
        factor_data_up = self.future_index_factors[self.future_index_factors.SPEED30 > 0].copy()
        factor_data_down = self.future_index_factors[self.future_index_factors.SPEED30 < 0].copy()
        long_call_target = pd.concat([self.cross_filter(factor_data=factor_data_up,
                                                        factor_name='SPEED30',
                                                        num=2,
                                                        top_or_bottom='bottom'),
                                      self.cross_filter(factor_data=factor_data_down,
                                                        factor_name='SPEED30',
                                                        num=2,
                                                        top_or_bottom='bottom')], axis=0)
        long_put_target = pd.concat([self.cross_filter(factor_data=factor_data_up,
                                                       factor_name='SPEED30',
                                                       num=2,
                                                       top_or_bottom='top'),
                                     self.cross_filter(factor_data=factor_data_down,
                                                       factor_name='SPEED30',
                                                       num=2,
                                                       top_or_bottom='top')], axis=0)
        self.long_call_target = long_call_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_put_target = long_put_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_call_target = self.long_call_target.reset_index(drop=True)
        self.long_put_target = self.long_put_target.reset_index(drop=True)
        self.long_call_target['FutureSymbol'] = self.long_call_target.CODE.apply(lambda r: get_symbol_prefix(r))
        self.long_put_target['FutureSymbol'] = self.long_put_target.CODE.apply(lambda r: get_symbol_prefix(r))
        del long_call_target, long_put_target, factor_data_up, factor_data_down

    def select_trade_dates(self):
        """取出符合频率的交易日期"""
        self.trade_dates = self.future_index_factors.DATETIME.apply(lambda r: r.date()).unique().tolist()
        df = pd.DataFrame(self.trade_dates, columns=['DATE'])
        df['WEEKDAY'] = df.DATE.apply(lambda r: r.weekday())
        weekday_mapper = {'Friday': 4,
                          'Thursday': 3,
                          'Wednesday': 2,
                          'Tuesday': 1,
                          'Monday': 0}
        df = df[df.WEEKDAY == weekday_mapper[self.adjust_rule[0]]]
        df.DATE = pd.to_datetime(df.DATE)
        return df.DATE.tolist()

    def back_test(self):
        trade_dates = self.select_trade_dates()
        if len(self.long_call_target) > 0:
            self.long_call_target = self.long_call_target[self.long_call_target.DATETIME.isin(trade_dates)].copy()
            self.long_call_target.to_csv('long_call.csv', index=False)
        if len(self.long_put_target) > 0:
            self.long_put_target = self.long_put_target[self.long_put_target.DATETIME.isin(trade_dates)].copy()
            self.long_put_target.to_csv('long_put.csv', index=False)
        if len(self.short_call_target) > 0:
            self.short_call_target = self.short_call_target[self.short_call_target.DATETIME.isin(trade_dates)].copy()
        if len(self.short_put_target) > 0:
            self.short_put_target = self.short_put_target[self.short_put_target.DATETIME.isin(trade_dates)].copy()

        target_symbol_mapper = {}
        for key, df_target in zip(['long_call',
                                   'long_put',
                                   'short_call',
                                   'short_put'],
                                  [self.long_call_target,
                                   self.long_put_target,
                                   self.short_call_target,
                                   self.short_put_target]):
            if len(df_target) == 0 or 'DATETIME' not in df_target.columns:
                continue
            target_symbol_mapper.update({key: df_target})

        if not target_symbol_mapper:
            self.console_logger.info('没有交易的品种, 退出回测')
            exit(0)
        #   循环产生持仓query
        for dt in trade_dates:
            for trade_side, df_target in target_symbol_mapper.items():
                today_target_symbols = df_target[df_target.DATETIME == dt].FutureSymbol.tolist()
                if today_target_symbols:
                    for symbol in today_target_symbols:
                        option_codes_info = self.query_target_option_code_with_info(
                            trade_date=dt,
                            symbol=symbol,
                            call_or_put=self.trade_side_mapper[trade_side]
                        )
                        if option_codes_info:
                            #   注意trade_date的顺延问题
                            task = OptionOrderTask(trade_date=option_codes_info['TRADE_DATE'],
                                                   order_side=trade_side,
                                                   symbol=symbol,
                                                   option_code=option_codes_info['CODE'],
                                                   info=option_codes_info['INFO'])
                            if dt not in self.option_order_tasks.keys():
                                self.option_order_tasks[dt] = [task]
                            else:
                                self.option_order_tasks[dt].append(task)
                            self.console_logger.info(f'{dt.strftime("%Y-%m-%d")}, {trade_side}, {symbol},{option_codes_info}')

    def load_option_gears(self, symbol, trade_date, file_type='bin'):
        if not isinstance(trade_date, str):
            trade_date = trade_date.strftime('%Y-%m-%d')
        parent_path = os.path.join(OPTIONS.config_db_path,
                                   f'每日可交易{symbol}期权gears信息')

        actor = DataFrameActor(parent_path=parent_path,
                               df=pd.DataFrame(),
                               filename=trade_date)
        try:
            if file_type == 'bin':
                df_gears = actor.from_bin()
            else:
                df_gears = actor.from_csv(parse_date='DATETIME')
            df_gears.DATETIME = pd.to_datetime(df_gears.DATETIME)
        except (FileNotFoundError, AttributeError, pandas.errors.EmptyDataError):
            df_gears = None
        return df_gears

    def query_target_option_code_with_info(self,
                                           trade_date,
                                           symbol,
                                           call_or_put,
                                           price_gears: str = '平值',
                                           file_type: str = 'bin',
                                           option_rule: str = 'zl'):
        target_option_codes = {}
        if symbol == 'IF':
            symbol = '000300.SH'
            option_rule = 'all'
        if symbol == 'IH':
            symbol = '510050.SH'
            option_rule = 'all'
        #   选取当日和后一日的gears信息, 避免当日无夜盘和无夜盘品种无法索引
        trade_date_idx = self.trade_dates.index(trade_date.date())
        if trade_date_idx + 1 >= len(self.trade_dates):
            return target_option_codes
        next_trade_date = self.trade_dates[trade_date_idx+1]
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        next_trade_date_str = next_trade_date.strftime('%Y-%m-%d')
        #   调仓时间, 依据adjust_rule设置, 支持当日夜盘或者次日
        query_start_time = datetime(year=trade_date.year,
                                    month=trade_date.month,
                                    day=trade_date.day,
                                    hour=self.adjust_rule[1].hour,
                                    minute=self.adjust_rule[1].minute)
        #   如果当天成功未能索引, 取次日, 对应获取期权合约信息
        df_gears_today = self.load_option_gears(symbol=symbol,
                                                trade_date=trade_date_str,
                                                file_type=file_type)
        if df_gears_today is None:
            return target_option_codes
        df_gears = df_gears_today[(df_gears_today.DATETIME >= query_start_time)].head(1)
        json_file = os.path.join(OPTIONS.config_db_path,
                                 f'每日可交易{symbol}期权合约信息',
                                 f'{trade_date_str}.json')
        if len(df_gears) == 0:
            df_gears_next_day = self.load_option_gears(symbol=symbol,
                                                       trade_date=next_trade_date_str,
                                                       file_type=file_type)
            if df_gears_next_day is None:
                return target_option_codes
            df_gears = df_gears_next_day.head(1)
            trade_date_str = next_trade_date_str
            json_file = os.path.join(OPTIONS.config_db_path,
                                     f'每日可交易{symbol}期权合约信息',
                                     f'{trade_date_str}.json')
        with open(json_file, "r") as f:
            result = json.load(f)
        if option_rule == "zl":
            info = result[trade_date_str]["zl_trade_options"]
        else:
            info = result[trade_date_str]["all_trade_options"]

        for key, val in df_gears.to_dict().items():
            #   510050.SH special, keep int when process, sorry for that
            key = str(key)
            if list(val.values()) == [price_gears]:
                if info[key]['call_or_put'] == call_or_put:
                    if info[key]['days_to_expire'] > self.minimum_days_to_expire:
                        if not target_option_codes:
                            target_option_codes['CODE'] = key
                            target_option_codes['TRADE_DATE'] = trade_date_str
                            target_option_codes['INFO'] = info[key]
                        else:
                            #   更新为满足条件下到期日更早的合约
                            if target_option_codes['INFO']['days_to_expire'] > info[key]['days_to_expire']:
                                target_option_codes.update({'CODE': key,
                                                            'TRADE_DATE': trade_date_str,
                                                            'INFO': info[key]})
        return target_option_codes

    def query_option_data_by_code(self, symbol, option_exchange, option_code, frequency='MIN_01'):

        if symbol == 'IH':
            symbol = '510050.SH'
            option_exchange = 'SH'
        if symbol == 'IF':
            symbol = '000300.SH'
            option_exchange = 'CFFEX'
        parent_path = os.path.join(
            OPTIONS.source_db_path,
            symbol,
            f"{option_code}.{option_exchange}")
        actor = DataFrameActor(parent_path=parent_path,
                               df=pd.DataFrame(),
                               filename=frequency)
        df = actor.from_bin()
        df.DATETIME = pd.to_datetime(df.DATETIME)
        return df


if __name__ == '__main__':
    opt_cross_engine = OptionCrossBackTestEngine(strategy_name='Score_Friday_Factor2',
                                                 adjust_rule=('Friday', time(21, 5, 0)),
                                                 trading_time_range=3,
                                                 option_symbols=('CU', 'AL', 'RU',
                                                                 'I', 'V', 'C', 'M', 'PP',
                                                                 'MA', 'CF', 'SR', 'TA', 'ZC',
                                                                 '000300.SH', '510050.SH')
                                                 )
    opt_cross_engine.load_future_index_data()
    opt_cross_engine.calculate_future_index_factor()
    opt_cross_engine.auto_filter_futures_score(num=3)
    opt_cross_engine.back_test()
    actor = CrossTraderActor(trade_dates=opt_cross_engine.trade_dates,
                             option_order_tasks=opt_cross_engine.option_order_tasks,
                             logger=opt_cross_engine.console_logger,
                             trade_side_rule=('long_call', 'long_put' ),
                             trading_time=opt_cross_engine.adjust_rule[1],
                             trading_minute_range=opt_cross_engine.trading_time_range)
    actor.run()
    actor.analysis(save_path=os.path.join(opt_cross_engine.log_file_path,
                                          opt_cross_engine.strategy_name))
    # actor.plot_performance(show=True,
    #                        save_path=os.path.join(opt_cross_engine.log_file_path,
    #                                               opt_cross_engine.strategy_name))



