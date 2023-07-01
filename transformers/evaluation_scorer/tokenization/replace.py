# replace ' - ' for CoNLL14 test src.
import sys

inputfile = sys.argv[1]
outfile = sys.argv[2]

out = open(outfile, 'w')

for line in open(inputfile):
    line = line.replace(' - ', '-')
    out.write(line)
out.close()
