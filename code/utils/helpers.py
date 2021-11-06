import datetime

def get_time():
    return datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")