import torch
from scipy.interpolate import interp1d
from numpy import arange

def calculateScore(parts_embeddings, keyword_embeddings):
    avg_score = 0
    max_score = 0
    max_score_index = 0
    index = 0
    for part_emb in parts_embeddings:
        score = torch.nn.functional.cosine_similarity(
            torch.tensor(part_emb), torch.tensor(keyword_embeddings), dim=-1)
        # print(score)
        avg_score += score.item()
        if(score.item() > max_score):
            max_score = score.item()
            max_score_index = index
        index += 1
    avg_score = (avg_score/len(parts_embeddings)+1)/2
    max_score = (max_score+1)/2
    max_score_index = max_score_index / len(parts_embeddings)
    return avg_score, max_score, max_score_index


def calculateScoreInterpolation(parts_embeddings, keyword_embeddings):
    x = []
    y = []
    avg_score = 0
    max_score = 0
    max_score_index = 0
    index = 0
    for part_emb in parts_embeddings:
        score = torch.nn.functional.cosine_similarity(
            torch.tensor(part_emb), torch.tensor(keyword_embeddings), dim=-1)
        x.append(float(index))
        y.append((score.item()+1)/2)
        # print(score)
        avg_score += score.item()
        if(score.item() > max_score):
            max_score = score.item()
            max_score_index = index
        index += 1
    avg_score = (avg_score/len(parts_embeddings)+1)/2
    max_score = (max_score+1)/2
    max_score_index = max_score_index / len(parts_embeddings)

    if(len(x)>1):
        # try:
        f = interp1d(x,y,kind='linear')
        x_new = arange(0.00001,x[len(x)-1],float(x[len(x)-1])/24.0).tolist()
        x_new.append(x[len(x)-1]-0.00001)
        # print(len(x_new))
        for i in range(len(x_new)):
            if(x_new[i]!=x_new[i]):
                x_new[i]=0
        five_points = f(x_new).tolist()
        # except:
        #     five_points = [y[0]]*25 
    elif len(x)==1:
        five_points = [y[0]]*25
    else:
        five_points = [0]*25
    # print(five_points)
    return avg_score, max_score, max_score_index, five_points