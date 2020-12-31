import json
import random
import re
# The collected dataset contains all visible field on the site, so that it can be use for other purpose as well
# in this scenario we only needs the title (to distinguish the articles), keywords list and the fulltext
# Although the text was cleaned up once while being collected, for this task, we also remove the keyword list from the text
# and remove paragraphs shorter than 150. The result is a list of paragraph long enough to analyze.
with open('cyberleninka.txt', encoding="utf-8") as json_file:
    raw = json.load(json_file)
    print(len(raw))
    extracted_short = []
    extracted_medium = []
    extracted_long = []
    extracted_mixed = []
    extracted = []
            # keywords = []
            # try:
            #     with open("dataset/"+str(folder_id)+"/"+str(folder_id)+".kwd") as f2:
            #         lines = f2.readlines()
            #         keywords = keywords+[k.strip() for k in lines]
            # except IOError:
            #     pass
            # for r, d, f in os.walk("dataset/"+str(folder_id)+"/KEY"):
            #     for file in f:
            #         if '.key' in file:
            #             with open("dataset/"+str(folder_id)+"/KEY/"+file) as f2:
            #                 lines = f2.readlines()
            #                 keywords = keywords+ [k.strip() for k in lines]
            # keywords_unique = set()
            # for kw in keywords:
            #     keywords_unique.add(kw.lower().strip())
            # # print(len(keywords_unique))
            # extracted.append({
            #     'title': title,
            #     'keywords': list(keywords_unique),
            #     'fulltext': paragraphs
            # })
    for article in raw:
        paragraphs = [] 
        content = article['fulltext']
        content = [c.strip().lower() for c in content]
        paragraph = ""
        leftover = ""
        fulltext_iter = iter(content)
        article_length = 0
        for line in fulltext_iter:
            if line.lower().startswith('ключевые слова') or line.lower().startswith('key words') or line.lower().startswith('keywords'):
                if len(paragraph)<18:
                    next(fulltext_iter, None)
                continue
            article_length+=len(line)
            fline = leftover+line
            leftover = ""
            while(fline.find('.')!=-1):
                index = fline.find('.')
                paragraph+=fline[:index+1]
                fline=fline[index+1:]
                if(len(paragraph)>1000):
                    paragraph = re.sub(r'^https?:\/\/.*[\r\n]*', '', paragraph, flags=re.MULTILINE)
                    paragraph = re.sub(r'[//0-9\(\)\|\=\{\}\[\]\!\@\#\$\%\^\&\*\-]','',paragraph)
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

        article_info = {
            'title': article['title'],
            'keywords': article['keywords'],
            'fulltext': paragraphs,
            'length': article_length
        }
        extracted.append(article_info)

        # if article_length<8000:
        #     extracted_short.append(article_info)
        # elif article_length<16000: 
        #     extracted_medium.append(article_info)
        # else:
        #     extracted_long.append(article_info)

# print("Long: "+str(len(extracted_long)))
# print("Medium: "+str(len(extracted_medium)))
# print("Short: "+str(len(extracted_short)))
random.shuffle(extracted)
extracted_mixed = extracted[:900]
extracted = extracted[900:]
print(str(len(extracted)))
extracted.sort(key=lambda x: x['length'])

with open('cyberleninka_mshort.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted[:300], ensure_ascii=False))
    output_file.write(data_json)
            
with open('cyberleninka_medium.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted[400:700], ensure_ascii=False))
    output_file.write(data_json)

with open('cyberleninka_mlong.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted[800:1100], ensure_ascii=False))
    output_file.write(data_json)

with open('cyberleninka_mixed.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted_mixed, ensure_ascii=False))
    output_file.write(data_json)
