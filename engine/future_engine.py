import os
import json
import logging

import pandas
import pandas as pd
import numpy as np
from typing import *

from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt, ticker
from openpyxl.utils.dataframe import dataframe_to_rows

from conf import *
from graph import *
from lib import *


class FutureCrossTraderActor(object):

    def __init__(self,
                 trade_dates,
                 logger,
                 future_order_tasks: Dict[str, type],
                 fixed_order_money=200_000,
                 initial_cash=5_000_000,
                 trading_time=time(21, 0, 0),
                 trading_minute_range=5,
                 trade_side_rule=('long', 'short')):
        self.future_order_tasks = future_order_tasks
        if len(self.future_order_tasks) < 20:
            raise NotImplementedError('tasks not enough')
        self.future_minute_bar_data = {}
        self.future_day_bar_data = {}
        self.trade_dates = trade_dates
        self.trading_time = trading_time
        self.trading_minute_range = trading_minute_range
        self.portfolio_change_dates = [dt for dt in self.future_order_tasks.keys()]
        self.portfolio_change_dates = list(set(self.portfolio_change_dates))
        self.portfolio_change_dates.sort()
        self.future_code_symbol_mapper = {}
        self.holding = {}
        self.profit_record = {}
        self.long_profit_record = {}
        self.short_profit_record = {}
        self.future_code_settle_record = {}
        self.cash = initial_cash
        self.init_equity = initial_cash
        self.equity = initial_cash
        self.fixed_order_money = fixed_order_money
        self.logger = logger
        self.finished_order_task_record = {}
        self.holding_record = {}
        self.trade_side_rule = trade_side_rule
        self.symbol_profit = {}
        for dt, tasks in self.future_order_tasks.items():
            for task in tasks:
                if task.symbol not in self.finished_order_task_record.keys():
                    self.finished_order_task_record[task.symbol] = []

    def query_future_bar_data_by_code(self, symbol, future_code, frequency='MIN_01', file_type='bin'):
        exchange_mapper = {
            'CU': 'SHFE',
            'RU': 'SHFE',
            'AL': 'SHFE',
            'RB': 'SHFE',
            'HC': 'SHFE',
            'I': 'DCE',
            'J': 'DCE',
            'V': 'DCE',
            'C': 'DCE',
            'M': 'DCE',
            'P': 'DCE',
            'Y': 'DCE',
            'PP': 'DCE',
            'SR': 'CZCE',
            'CF': 'CZCE',
            'TA': 'CZCE',
            'MA': 'CZCE',
            'ZC': 'CZCE',
            'SC': 'INE',
            'IF': 'CFFEX',
            'IH': 'CFFEX',
            'T': 'CFFEX'}
        exchange = exchange_mapper[symbol]
        parent_path = os.path.join(FUTURES.source_db_path,
                                   symbol,
                                   f'{future_code}.{exchange}')
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
                    for future_code, amount in self.holding.copy().items():
                        df_minute_bar = self.future_minute_bar_data[future_code].copy()
                        price_list = []
                        deal_price = np.nan
                        deal_time = np.nan
                        for row_ in df_minute_bar.itertuples():
                            if row_.DATETIME > trade_timestamp + timedelta(hours=self.trading_time.hour,
                                                                           minutes=self.trading_time.minute):
                                price_list.append(row_.CLOSE)
                            if len(price_list) >= self.trading_minute_range and (
                                    row_.DATETIME > trade_timestamp + timedelta(
                                    hours=self.trading_time.hour, minutes=self.trading_time.minute) + 
                                    timedelta(minutes=self.trading_minute_range)):
                                deal_price = np.nanmean(price_list)
                                deal_time = row_.DATETIME
                                self.future_minute_bar_data[future_code] = df_minute_bar.iloc[row_.Index:]
                                break
                        if np.isnan(deal_price):
                            df_day_bar = self.future_day_bar_data[future_code]
                            try:
                                deal_price = df_day_bar[df_day_bar.DATETIME == trade_timestamp].CLOSE.iloc[0]
                            except Exception:
                                deal_price = self.future_code_settle_record[future_code]
                            deal_time = trade_timestamp + timedelta(hours=15)
                        if amount > 0:
                            self.record_order(future_code=future_code,
                                              order_amount=abs(amount),
                                              order_side='cover_long',
                                              deal_price=deal_price,
                                              deal_time=deal_time)
                        else:
                            self.record_order(future_code=future_code,
                                              order_amount=abs(amount),
                                              order_side='cover_short',
                                              deal_price=deal_price,
                                              deal_time=deal_time)
                        self.future_minute_bar_data.pop(future_code)
                        self.future_day_bar_data.pop(future_code)

                order_tasks = self.future_order_tasks[trade_timestamp]
                #   如果存在订单任务
                if order_tasks:
                    #   索引数据
                    for order_task in order_tasks:
                        if order_task.order_side not in self.trade_side_rule:
                            continue
                        self.future_code_symbol_mapper[order_task.future_code] = order_task.symbol
                        if order_task.future_code not in self.future_day_bar_data.keys():
                            self.future_day_bar_data[order_task.future_code] = self.query_future_bar_data_by_code(
                                symbol=order_task.symbol,
                                future_code=order_task.future_code,
                                frequency='DAY'
                            )
                        if order_task.future_code not in self.future_minute_bar_data.keys():
                            self.future_minute_bar_data[order_task.future_code] = self.query_future_bar_data_by_code(
                                symbol=order_task.symbol,
                                future_code=order_task.future_code,
                                frequency='MIN_01'
                            )
                        df_minute_bar = self.future_minute_bar_data[order_task.future_code].copy()
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
                                self.future_minute_bar_data[order_task.future_code] = df_minute_bar.iloc[row_.Index:]
                                break

                        if np.isnan(deal_price):
                            continue
                        self.record_order(future_code=order_task.future_code,
                                          order_money=self.fixed_order_money,
                                          order_amount=None,
                                          order_side=f'open_{order_task.order_side}',
                                          deal_price=deal_price,
                                          deal_time=deal_time)

    def record_order(self, future_code, order_amount, order_side, deal_price, deal_time, order_money: Optional[float] = None):
        symbol = self.future_code_symbol_mapper[future_code]
        if order_money is not None and order_amount is None:
            order_amount = round(order_money/(deal_price *
                                              FutureTradeConfig.multiplier_table[symbol] *
                                              FutureTradeConfig.commitment_table[symbol]))
        if order_amount <= 0:
            return

        if order_side == 'open_long':
            if future_code not in self.holding.keys():
                self.holding[future_code] = order_amount
            else:
                self.holding[future_code] = self.holding[future_code] + order_amount
            self.finished_order_task_record[symbol].append(FinishedFutureOrderTask(future_code=future_code,
                                                                                   symbol=symbol,
                                                                                   open_time=deal_time,
                                                                                   open_price=deal_price,
                                                                                   order_side=order_side,
                                                                                   deal_amount=order_amount,
                                                                                   ))
            self.cash = self.cash - order_amount * deal_price * FutureTradeConfig.multiplier_table[symbol]

        elif order_side == 'open_short':
            if future_code not in self.holding.keys():
                self.holding[future_code] = -order_amount
            else:
                self.holding[future_code] = self.holding[future_code] - order_amount
            self.finished_order_task_record[symbol].append(FinishedFutureOrderTask(future_code=future_code,
                                                                                   symbol=symbol,
                                                                                   open_time=deal_time,
                                                                                   open_price=deal_price,
                                                                                   order_side=order_side,
                                                                                   deal_amount=order_amount,
                                                                                   ))
            self.cash = self.cash + order_amount * deal_price * FutureTradeConfig.multiplier_table[symbol]
        elif order_side == 'cover_long':
            if future_code in self.holding.keys():
                self.holding[future_code] = self.holding[future_code] - order_amount
                if self.holding[future_code] <= 0:
                    self.holding.pop(future_code)

            for finished_order_task in self.finished_order_task_record[symbol]:
                if finished_order_task.future_code == future_code and finished_order_task.cover_time is None:
                    finished_order_task.cover_time = deal_time
                    finished_order_task.cover_price = deal_price
                    finished_order_task.order_side = order_side
                    self.logger.info(f'{deal_time} 平仓多头合约{future_code}, 盈亏{round(finished_order_task.profit, 2)}, '
                                     f'完成订单{finished_order_task}')

        else:
            if future_code in self.holding.keys():
                self.holding[future_code] = self.holding[future_code] + order_amount
                if self.holding[future_code] >= 0:
                    self.holding.pop(future_code)
            for finished_order_task in self.finished_order_task_record[symbol]:
                if finished_order_task.future_code == future_code and finished_order_task.cover_time is None:
                    finished_order_task.cover_time = deal_time
                    finished_order_task.cover_price = deal_price
                    finished_order_task.order_side = order_side
                    self.logger.info(f'{deal_time} 平仓空头合约{future_code}, 盈亏{round(finished_order_task.profit, 2)}, '
                                     f'完成订单{finished_order_task}')

    def daily_settle(self, trade_date):
        if self.holding:
            holding_recent_price = {}
            for future_code, amount in self.holding.items():
                symbol = self.future_code_symbol_mapper[future_code]
                df_day_bar = self.future_day_bar_data[future_code]

                try:
                    price = df_day_bar[df_day_bar.DATETIME == pd.Timestamp(trade_date)].CLOSE.iloc[0]
                except Exception:
                    if future_code in self.future_code_settle_record.keys():
                        price = self.future_code_settle_record[future_code]
                    else:
                        price = np.nan
                self.future_code_settle_record[future_code] = price
                holding_recent_price[future_code] = price

                for finished_order_task in self.finished_order_task_record[symbol]:
                    if finished_order_task.future_code == future_code and finished_order_task.cover_time is None:
                        finished_order_task.settle_price = price
                        self.equity = self.equity + finished_order_task.settle_profit
            # self.logger.info(f'持仓合约的结算价格 {holding_recent_price}')

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
        sns.lineplot(data=df["CumProfit"],
                     color="orange",
                     ax=ax_arr[1])
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


class FutureCrossBackTestEngine(object):

    future_symbols: Tuple[str] = ('CU', 'AL', 'RU', 'RB', 'HC',
                                  'I', 'J', 'V', 'C', 'M', 'Y', 'P',
                                  'PP', 'SC', 'MA', 'CF', 'SR', 'TA', 'ZC',
                                  'IF', 'IH', 'T')

    def __init__(self,
                 strategy_name: Union[str] = 'FutureCrossHedge',
                 frequency: Union[str] = 'DAY',
                 adjust_rule: Union[Tuple[str, time]] = ('Friday', time(21, 5, 0)),
                 trading_time_range: Union[int] = 3,
                 start_date: Union[str, datetime] = '2019-01-01',
                 end_date: Union[str, datetime] = SUPPORT.data_end_date,
                 future_symbols: Optional[Tuple[str]] = None,
                 ):
        if future_symbols is not None:
            self.future_symbols = future_symbols
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
        self.all_future_info = pd.DataFrame()
        self.long_target = pd.DataFrame()
        self.short_target = pd.DataFrame()
        self.trade_dates = pd.DataFrame()
        self.future_order_tasks = {}

        self.log_time_version = None
        self.log_file_path = None
        self.make_log_path()
        self.get_all_future_info()

    @property
    def future_exchange_mapper(self):
        return {
            'CU': 'SHFE',
            'RU': 'SHFE',
            'AL': 'SHFE',
            'RB': 'SHFE',
            'HC': 'SHFE',
            'I': 'DCE',
            'J': 'DCE',
            'V': 'DCE',
            'C': 'DCE',
            'M': 'DCE',
            'P': 'DCE',
            'Y': 'DCE',
            'PP': 'DCE',
            'SR': 'CZCE',
            'CF': 'CZCE',
            'TA': 'CZCE',
            'MA': 'CZCE',
            'ZC': 'CZCE',
            'SC': 'INE',
            'IF': 'CFFEX',
            'IH': 'CFFEX',
            'T': 'CFFEX'}

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
        for future_symbol in self.future_symbols:
            future_exchange = self.future_exchange_mapper[future_symbol]
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

    def auto_filter_futures_score(self, num=3):
        """逆势"""
        long_target = self.cross_filter(factor_data=self.future_index_factors,
                                        factor_name='SCORE',
                                        num=num,
                                        top_or_bottom='bottom')
        short_target = self.cross_filter(factor_data=self.future_index_factors,
                                         factor_name='SCORE',
                                         num=num,
                                         top_or_bottom='top')
        self.long_target = long_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.short_target = short_target.sort_values(by=['DATETIME', 'CODE'], ascending=True)
        self.long_target = self.long_target.reset_index(drop=True)
        self.short_target = self.short_target.reset_index(drop=True)
        self.long_target['FutureSymbol'] = self.long_target.CODE.apply(lambda r: get_symbol_prefix(r))
        self.short_target['FutureSymbol'] = self.short_target.CODE.apply(lambda r: get_symbol_prefix(r))
        del long_target, short_target

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
        if len(self.long_target) > 0:
            self.long_target = self.long_target[self.long_target.DATETIME.isin(trade_dates)].copy()
        if len(self.short_target) > 0:
            self.short_target = self.short_target[self.short_target.DATETIME.isin(trade_dates)].copy()

        self.long_target = self.long_target.merge(self.all_future_info,
                                                  left_on=['DATETIME', 'FutureSymbol'],
                                                  right_on=['DATETIME', 'FutureSymbol'],
                                                  how='inner')
        self.short_target = self.short_target.merge(self.all_future_info,
                                                    left_on=['DATETIME', 'FutureSymbol'],
                                                    right_on=['DATETIME', 'FutureSymbol'],
                                                    how='inner')

        target_symbol_mapper = {}
        for key, df_target in zip(['long', 'short'],
                                  [self.long_target,
                                   self.short_target]):
            if len(df_target) == 0 or 'DATETIME' not in df_target.columns:
                continue
            target_symbol_mapper.update({key: df_target})

        if not target_symbol_mapper:
            self.console_logger.info('没有交易的品种, 退出回测')
            exit(0)

        #   循环产生持仓query
        for dt in trade_dates:
            for trade_side, df_target in target_symbol_mapper.items():
                today_target_query = df_target[df_target.DATETIME == dt]
                if len(today_target_query) > 0:
                    today_target_symbols = today_target_query.FutureSymbol.tolist()
                    today_target_codes = today_target_query.TARGET_TRADE_HISCODE.tolist()

                    for symbol, code in zip(today_target_symbols, today_target_codes):
                        #   注意trade_date的顺延问题
                        task = FutureOrderTask(trade_date=dt,
                                               order_side=trade_side,
                                               symbol=symbol,
                                               future_code=code,
                                               info="")
                        if dt not in self.future_order_tasks.keys():
                            self.future_order_tasks[dt] = [task]
                        else:
                            self.future_order_tasks[dt].append(task)
                        self.console_logger.info(f'{dt.strftime("%Y-%m-%d")}, {task.order_side}, {task.symbol}, '
                                                 f'{task.future_code}')

    def get_all_future_info(self, symbols: Optional[List] = None):
        if symbols is None:
            symbols = self.future_symbols
        df_contract_info_list = []
        for symbol in symbols:
            #   查询历史合约表
            exchange = self.future_exchange_mapper[symbol]
            if exchange == 'SHFE':
                exchange = 'SHFEX'

            df_contract_info = pd.read_csv(os.path.join(FUTURES.config_db_path,
                                                        f'历史信息{symbol}.{exchange}.csv'),
                                           parse_dates=['trade_date'])
            df_contract_info['TARGET_TRADE_HISCODE'] = df_contract_info.TRADE_HISCODE.shift(-5)
            df_contract_info['FutureSymbol'] = symbol
            df_contract_info = df_contract_info.fillna(method='ffill')
            df_contract_info_list.append(df_contract_info[['FutureSymbol',
                                                           'TRADE_HISCODE',
                                                           'TARGET_TRADE_HISCODE',
                                                           'trade_date']])
        df_contract_info = pd.concat(df_contract_info_list, axis=0)
        df_contract_info = df_contract_info.rename(columns={'trade_date': 'DATETIME'})
        df_contract_info = df_contract_info.sort_values(by=['FutureSymbol', 'DATETIME'],
                                                        ascending=True)
        df_contract_info.TRADE_HISCODE = df_contract_info.TRADE_HISCODE.apply(lambda r: r.split('.')[0])
        df_contract_info.TARGET_TRADE_HISCODE = df_contract_info.TARGET_TRADE_HISCODE.apply(lambda r: r.split('.')[0])
        self.all_future_info = df_contract_info.reset_index(drop=True)

    def query_future_data_by_code(self, symbol, future_exchange, future_code, frequency='MIN_01'):
        parent_path = os.path.join(
            FUTURES.source_db_path,
            symbol,
            f"{future_code}.{future_exchange}")
        actor = DataFrameActor(parent_path=parent_path,
                               df=pd.DataFrame(),
                               filename=frequency)
        df = actor.from_bin()
        df.DATETIME = pd.to_datetime(df.DATETIME)
        return df


if __name__ == '__main__':
    fut_cross_engine = FutureCrossBackTestEngine(strategy_name='FutureTest',
                                                 adjust_rule=('Friday', time(21, 5, 0)),
                                                 trading_time_range=3)
    fut_cross_engine.load_future_index_data()
    fut_cross_engine.calculate_future_index_factor()
    fut_cross_engine.auto_filter_futures_score(num=5)
    fut_cross_engine.back_test()

    actor = FutureCrossTraderActor(trade_dates=fut_cross_engine.trade_dates,
                                   future_order_tasks=fut_cross_engine.future_order_tasks,
                                   logger=fut_cross_engine.console_logger,
                                   trade_side_rule=('long', 'short'),
                                   trading_time=fut_cross_engine.adjust_rule[1],
                                   trading_minute_range=fut_cross_engine.trading_time_range)
    actor.run()
    actor.analysis(save_path=os.path.join(fut_cross_engine.log_file_path,
                                          fut_cross_engine.strategy_name)
                   )