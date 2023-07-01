# 指定
# input_file="/home/yhli/164/codes/fairseq-gec/nlpcc18-char/transformer_vanilla/results_train/hyp.txt"
# output_file="/home/yhli/164/codes/fairseq-gec/nlpcc18-char/transformer_vanilla/results_train/hyp.word.tok"
# python /home/yhli/codes/MuCGEC/tools/segment/segment_pkunlp.py $input_file $output_file
import threading
import os
 
def writetoTxt(txtFile, str):
    id = threading.currentThread().getName()
    mutex = threading.Lock()
    mutex.acquire()
    print(str)
    with open(txtFile, 'a') as f:
        f.write("write from thread {0} \r\n".format(id))

        mutex.release()
        mutex = threading.Lock()

for i in range(10):
    myThread = threading.Thread(target=writetoTxt, args=("tmp.txt", 'test'))
    myThread.start()