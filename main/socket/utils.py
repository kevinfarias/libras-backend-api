import datetime
import time

class DateUtils:
    def getNowInStr():
        return str('{date:%Y%m%d_%H%M}').format(date=datetime.datetime.now())

    def getTimeDiffInMin(start, end):
        return (end - start)/60
