import json

count=1
with open("./cyberleninka_ds.txt", encoding='utf-8') as fp:
    datafiles = json.load(fp)
    for datafile in datafiles:
        with open('./cyberleninka/'+str(count)+'.txt','w', encoding='utf-8') as output_file:
            output_file.write(" ".join(datafile['fulltext']))
        count+=1
            