#!/usr/bin/env python
# coding: utf-8
import torch
import os
import sys
import datetime

from utils.helpers import get_time


print(os.getcwd())



start1 = datetime.datetime.now()
start = get_time()

print('hello')

finish = get_time()
finish1 = datetime.datetime.now()

diff = finish1 - start1

print(diff)


