import io
import json
import logging

import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime, timedelta

from typing import *

__all__ = ['EventLogger', 'FileEventLogger', 'JsonFileEventLogger']


def _json_encode(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S.%f')
    if isinstance(obj, timedelta):
        return str(obj)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, np.int64):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_encode(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _json_encode(v) for k, v in obj.items()}
    if is_dataclass(obj.__class__):
        return _json_encode(asdict(obj))
    if isinstance(obj, float) and np.isnan(obj):
        return 'nan'
    return obj


class EventLogger(object):

    def write_event(self, event_type, event_time, obj: Any):
        raise NotImplementedError()

    def write_order_accepted(self, event_time, strategy_name, order):
        self.write_event('OrderAccepted', event_time, {'strategy': strategy_name, 'order': order})

    def write_order_canceled(self, event_time, strategy_name, order):
        self.write_event('OrderCanceled', event_time, {'strategy': strategy_name, 'order': order})

    def write_holding_check(self, event_time, strategy_name, order):
        self.write_event('HoldingCheck', event_time, {'strategy': strategy_name, 'order': order})

    def write_order_finished(self, event_time, strategy_name, order):
        self.write_event('OrderFinished', event_time, {'strategy': strategy_name, 'order': order})


class FileEventLogger(EventLogger):

    mode: str

    def __init__(self, path: str, mode: str = 'w'):
        self.path = path
        self.mode = mode
        self.file_obj = None

    def __enter__(self):
        self.file_obj = open(self.path, self.mode, encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_obj is not None:
            self.file_obj.close()
            self.file_obj = None


class JsonFileEventLogger(FileEventLogger):

    def write_event(self, event_type, event_time, obj: Any):
        event_obj = {'event_type': event_type, 'event_time': event_time, 'obj': obj}
        event_str = json.dumps(_json_encode(event_obj))
        #   logging.info('%s', event_str)
        self.file_obj.write(f'{event_str}\n')
