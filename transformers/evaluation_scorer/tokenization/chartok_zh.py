# linux: sed 's/\ //g' file1 > file2  
import sys

def chartok_zh(input):
    char_list = []
    
    for line in open(input):
        line = line.strip()

        for char in line:
            
            char_list.append(char)
            char_list.append(' ')

        char_list.pop()
        char_list.append("\n")

    return char_list

def deblank(input_file, output):

    chartok = open(output, "w")

    for i in chartok_zh(input_file):
        chartok.write(i)
    
    chartok.close()

if __name__ == "__main__":
    deblank(sys.argv[1], sys.argv[2])
