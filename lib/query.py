import requests
import pandas as pd

from lib import datetime_to_unix


class OnlineQuery(object):

    def __init__(self):
        pass

    @property
    def frequency_mapper(self):
        return {"MIN_01": "futu_min1",
                "MIN_03": "futu_min3",
                "MIN_05": "futu_min5",
                "MIN_10": "futu_min10",
                "MIN_15": "futu_min15",
                "MIN_30": "futu_min30",
                "HOU_01": "futu_hour",
                "DAY": "futu_day",
                }

    @property
    def frequency(self):
        return "MIN_01", "MIN_05", "MIN_15", "MIN_30", "HOU_01", "DAY"

    @property
    def index_type(self):
        return "指数", "主连"

    def get_contract_bar_yield(self,
                               full_code,
                               start_time,
                               end_time,
                               frequency='MIN_01'):
        start_unix = datetime_to_unix(start_time)
        end_unix = datetime_to_unix(end_time)

        url = (
            "http://yd.yafco.com/marketcenter/api/quota/data/"
            f"getFutuQuotaData?code={full_code}&type={self.frequency_mapper[frequency]}&begTime={start_unix}&endTime={end_unix}"
        )
        bar_list = []
        response = requests.get(url)
        json_ = response.json()
        for key, val in json_["data"].items():
            for record in val:
                row_dict = record["values"]
                row_dict.update(record["infos"])
                for key, val in record.items():
                    if key == "values" or key == "infos":
                        continue
                    row_dict.update({key: val})
                bar_list.append(row_dict)
        return bar_list

    def get_index_bar_yield(self,
                            full_code,
                            start_time,
                            end_time,
                            frequency='MIN_01'):
        return

    def get_contract_bar_point(self,
                               full_code,
                               end_time):
        return

    def get_index_bar_point(self,
                            full_code,
                            end_time):
        return


if __name__ == "__main__":
    import time
    query = OnlineQuery()
    start = time.time()
    a = query.get_contract_bar_yield(full_code='CU2110.SHFE',
                                     start_time=pd.to_datetime('2021-09-30 09:00:00'),
                                     end_time=pd.to_datetime('2021-09-30 15:00:00'))
    end = time.time()
    print(a)
    print(end - start)