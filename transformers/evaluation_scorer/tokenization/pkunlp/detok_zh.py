# linux: sed 's/\ //g' file1 > file2  
import sys

def detok_zh(input, output):
    detok = open(output, "w")
    for line in open(input):
        line = line.strip()
        line = line.split()
        for tok in line:
            detok.write(tok)
        detok.write("\n")

if __name__ == "__main__":
    detok_zh(sys.argv[1], sys.argv[2])
