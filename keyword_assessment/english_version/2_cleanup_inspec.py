import re
import os
import json
extracted = []
for folder_id in range(0,2201):
    try:
        with open("Inspec/docsutf8/"+str(folder_id)+".txt") as f:
            paragraphs = []
            content = f.readlines()
            title=str(folder_id)
            content = [c.strip().lower() for c in content]
            paragraph = ""
            leftover = ""
            for line in content:
                fline = leftover+line
                fline = re.sub(r'^https?:\/\/.*[\r\n]*', '', fline, flags=re.MULTILINE)
                fline = re.sub(r'[//0-9\[\]\t]','',fline)
                leftover = ""
                while(fline.find('.')!=-1):
                    index = fline.find('.')
                    paragraph+=fline[:index+1]
                    fline=fline[index+1:]
                    if(len(paragraph)>500):
                        paragraphs.append(paragraph.strip())
                        paragraph=""
                paragraph+=fline.strip()+" "
            paragraphs.append(paragraph.strip())
            keywords = []
            try:
                with open("Inspec/keys/"+str(folder_id)+".key") as f2:
                    lines = f2.readlines()
                    keywords = keywords+[k.strip().replace('\t','') for k in lines]
            except IOError:
                pass
            keywords_unique = set()
            for kw in keywords:
                keywords_unique.add(kw.lower().strip())
            # print(len(keywords_unique))
            extracted.append({
                'title': title,
                'keywords': list(keywords_unique),
                'fulltext': paragraphs
            })
    except IOError:
        pass
with open('dataset_inspec.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted, ensure_ascii=False))
    output_file.write(data_json)