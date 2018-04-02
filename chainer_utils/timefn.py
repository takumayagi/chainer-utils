#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import time
import functools

def timefn(fn):
    @functools.wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("Elapsed time ({}): {} (s)".format(fn.__name__, end - start))
        return result
    return measure_time

@timefn
def main():
    time.sleep(1)

if __name__ == "__main__":
    main()
