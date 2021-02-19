'''
从训练集/测试集生成列表存入train_label.txt/test_label.txt 
'''


import os
import pandas as pd
import numpy as np
import re



data = pd.read_excel('D:/CK/CK表情数据集/CK表情数据集/Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.xls')
    # print(type(data))
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
#print(excel_list)

#Folder = "/home/parrot/train"
Folder = "D:/CK/CK表情数据集/CK表情数据集/cohn-kanade/cohn-kanade/cohn-kanade/"

#Out = "/home/parrot/py/train_label.txt"
Out = "C:/Users/李笑嫣/Desktop/AU代码/1.txt"

f = open(Out, "w")

for dir in os.listdir(Folder):
    for file in os.listdir(os.path.join(Folder,dir)):
        d=int(dir[1:])
        fir=int(file)
        ini='0 '*64
        
        #for i in range(0,len(excel_list)):
        for i in range(0,len(excel_list)):
            if excel_list[i][0]==d and excel_list[i][1]== fir:
                str_l=str(excel_list[i][2])
        label=str_l.split('+')
        lab=[]
        for l in label:
            la=list([val for val in l if val.isnumeric()])
            strr="".join(la)
            lab.append(strr)
        label=lab

        
        for l in label:
            ini=list(ini)
            ini[int(l)*2-2]='1'
            ini=''.join(ini)

        last=list(ini)
        last=last[:-1]
        last=''.join(last)
        ini=last

              
        for img in os.listdir(os.path.join(Folder,dir,file)):
            path=Folder+dir+'/'+file+'/'+img+" "+ini+'\n'    
            f.writelines(path)
        

f.close()
