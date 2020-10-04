import json

# The collected dataset contains all visible field on the site, so that it can be use for other purpose as well
# in this scenario we only needs the title (to distinguish the articles), keywords list and the fulltext
# Although the text was cleaned up once while being collected, for this task, we also remove the keyword list from the text
# and remove paragraphs shorter than 150. The result is a list of paragraph long enough to analyze.
with open('cyberleninka.txt', encoding="utf-8") as json_file:
    raw = json.load(json_file)
    print(len(raw))
    extracted = []
    for article in raw:
        filtered_text = []
        fulltext_iter = iter(article['fulltext'])
        for paragraph in fulltext_iter:
            if paragraph.lower().startswith('ключевые слова') or paragraph.lower().startswith('key words') or paragraph.lower().startswith('keywords'):
                if len(paragraph)<18:
                    next(fulltext_iter, None)
                continue
            if len(paragraph)<150:
                continue
            filtered_text.append(paragraph)
        extracted.append({
            'title': article['title'],
            'keywords': article['keywords'],
            'fulltext': filtered_text
        })

with open('cyberleninka_ds.txt','w+', encoding='utf-8') as output_file:
    data_json = (json.dumps(extracted, ensure_ascii=False))
    output_file.write(data_json)
            