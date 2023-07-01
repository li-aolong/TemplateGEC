import sys
import spacy
# python -m spacy download de_core_news_sm

nlp = spacy.load("de_core_news_sm")

def token(inputfile, outfile):
    des = open(outfile,"w")
    for line in open(inputfile, 'r'):
        line = line.strip()
        tokens = nlp(line)
        sent_tok = ' '.join([tok.text for tok in tokens])
        des.write(sent_tok+"\n")
    des.close()

if __name__ == "__main__":
    token(sys.argv[1], sys.argv[2])

