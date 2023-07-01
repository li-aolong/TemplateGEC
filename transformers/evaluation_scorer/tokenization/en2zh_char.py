# -*- coding: UTF-8 -*-
import sys
table = {ord(f):ord(t) for f,t in zip(
     u',!?()',
     u'，！？（）')}


inputfile = sys.argv[1]
outputfile = sys.argv[2]

f_out = open(outputfile, "w")

for line in open(inputfile, 'r'):
    line = line.strip()
    line = line.translate(table)
    f_out.write(line+"\n")

f_out.close()
