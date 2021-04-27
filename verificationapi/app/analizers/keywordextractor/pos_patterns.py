POS_PATTERNS = {
    'en': [
        ["NOUN"],
        ["PROPN"],
        ['ADJ'],
        ['VERB', 'NOUN'],
        ["ADJ", "NOUN"],
        ["NOUN", "NOUN"],
        ["NOUN", "PROPN"],
        ["ADJ", "ADJ", "NOUN"],
        ["NOUN", "NOUN", "NOUN"],
        ["ADJ", "NOUN", "NOUN"]
    ],
    'ru': [
        ["NOUN"],  # 789
        # ["ADJ"], # 14
        ["PROPN"],  # 589
        ["ADJ", "NOUN"],  # 944
        # ["NOUN", "PROPN"], # 10
        # ["VERB", "NOUN"], # 49
        # ["NOUN", "PUNCT", "NOUN"], # 43
        ["NOUN", "NOUN"],  # 444
        ["PROPN", "PROPN"],  # 932
        ["ADJ", "ADJ", "NOUN"],  # 85
        ["NOUN", "ADJ", "NOUN"],  # 102
        ["NOUN", "NOUN", "NOUN"],  # 83
        ["ADJ", "NOUN", "NOUN"],  # 71
        ["PROPN", "PROPN", "PROPN"],  # 254
        # ["PROPN", "X", "PROPN"], # 47
        # ["PROPN", "X", "PROPN", "PROPN"], # 64
        # ["PROPN", "PUNCT", "PROPN", "PROPN"], # 31
        # ["PROPN", "PROPN", "X", "PROPN"], # 27
        # ["ADJ", "PUNCT", "ADJ", "NOUN"], # 46
        # ["ADJ", "NOUN", "NOUN", "NOUN"], # 10
        # ["NOUN", "NOUN", "ADJ", "NOUN"], # 16
        # ["NOUN", "NOUN", "NOUN", "NOUN"], # 20
        # ["PROPN", "PROPN", "PROPN", "PROPN"], # 34
    ]
}
