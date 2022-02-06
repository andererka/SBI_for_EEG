#!/usr/bin/env python
# coding: utf-8
import torch
import os
import sys
import datetime

from utils.helpers import get_time


print(os.getcwd())





finish = datetime.datetime(year= 2022, month= 1, day= 24, hour= 18, minute=9, second=10)

start= datetime.datetime(year=2022, month=1, day=23, hour=13, minute=59, second=43)

print(finish-start)

