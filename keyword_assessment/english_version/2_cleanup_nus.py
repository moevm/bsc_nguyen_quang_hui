import re
import os
import json
extracted = []
for folder_id in range(1,216):
    try:
        with open("dataset/"+str(folder_id)+"/"+str(folder_id)+".txt") as f:
            paragraphs = []
            content = f.readlines()
            title=content[0].strip()
            content=content[1:]
            content = [c.strip().lower() for c in content]
            paragraph = ""
            leftover = ""
            for line in content:
                fline = leftover+line
                
                leftover = ""
                while(fline.find('.')!=-1):
                    index = fline.find('.')
                    paragraph+=fline[:index+1]
                    fline=fline[index+1:]
                    if(len(paragraph)>1000):
                        paragraph = re.sub(r'^https?:\/\/.*[\r\n]*', '', paragraph, flags=re.MULTILINE)
                        paragraph = re.sub(r'[//0-9\[\]\!\@\#\$\%\^\&\*\-]','',paragraph)
                        # paragraph = re.sub(r'\(.+\)','', paragraph)
                        paragraph = re.sub(r'\<.+href.+\>','', paragraph)
                        paragraph = re.sub(r'\s\s+',' ', paragraph)
                        paragraphs.append(paragraph.strip())
                        paragraph=""
                paragraph+=fline.strip()+" "
            if(len(paragraph.strip())>300):
                paragraphs.append(paragraph.strip())
            elif len(paragraphs)>0:
                paragraphs[len(paragraphs)-1]+=paragraph.strip()
            else:
                paragraphs.append(paragraph.strip())
            keywords = []
            try:
                with open("dataset/"+str(folder_id)+"/"+str(folder_id)+".kwd") as f2:
                    lines = f2.readlines()
                    keywords = keywords+[k.strip() for k in lines]
            except IOError:
                pass
            for r, d, f in os.walk("dataset/"+str(folder_id)+"/KEY"):
                for file in f:
                    if '.key' in file:
                        with open("dataset/"+str(folder_id)+"/KEY/"+file) as f2:
                            lines = f2.readlines()
                            keywords = keywords+ [k.strip() for k in lines]
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
with open('dataset_nus.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted, ensure_ascii=False))
    output_file.write(data_json)