import numpy as np

path1 = "../Output/t1.csv"
path2 = "../Output/t2.csv"
path3 = "../Output/t3.csv"

def vote(filepath_list, outpath="../Output/tvote.csv"):
    fo = open(outpath, 'w')
    fo.write('Id,Prediction\n')
    Line_List = []
    for path in filepath_list:
        fstream = open(path, 'r')
        afile = fstream.readlines()
        Line_List.append(afile)
    for idx in range(200):
        count_0 = 0
        count_1 = 0
        for l in Line_List:
            if l[idx+1].strip()[-1] == '0':
                count_0 += 1
            else:
                count_1 += 1
        if count_0>0 and count_1>0:
            print "NONO"
        if count_0>count_1:
            fo.write(str(idx)+',0\n')
        else:
            fo.write(str(idx)+',1\n')

def diffcount(anchor_file, test_file):
    fstream_a = open(anchor_file, 'r')
    fstream_t = open(test_file, 'r')
    afile = fstream_a.readlines()
    tfile = fstream_t.readlines()
    afile = [a.strip() for a in afile]
    tfile = [t.strip() for t in tfile]
    count = 0
    for idx in range(200):
        if afile[idx+1][-1] !=  tfile[idx+1][-1]:
            count+=1
    print count
