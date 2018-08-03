from scipy import spatial

def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

