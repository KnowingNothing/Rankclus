# -*- coding: utf-8 -*-

"""
Created on May 14, 2018
@author: zhengsz@pku.edu.cn
Last modify: May 24, 2018
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # 改变标准输出的默认编码

"""
    we use this procedure to produce a much smaller database
"""
if __name__ == "__main__":
    total_counts = 1000000
    f = open("small_dblp.xml", "w")
    print("Start coping some xml data of DBLP at path: E:/DATASETS/DBLP/dblp.xml...")
    beg_time = time.time()
    ff = open("E:/DATASETS/DBLP/dblp.xml", "r")
    for i in range(total_counts):
        line = ff.readline()
        f.write(line)
    while True:
        line = ff.readline()
        if "</article>" in line and line.index("</article>") == 0:
            f.write("</article>")
            break
        f.write(line)
    f.write("</dblp>")
    ff.close()
    end_time = time.time()
    print("Copy done! Use {} seconds in total.".format(end_time - beg_time))
    f.close()
