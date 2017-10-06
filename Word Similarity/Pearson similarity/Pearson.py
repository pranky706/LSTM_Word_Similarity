import math

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)
	
from collections import namedtuple
from gensim.models import doc2vec
# Load data

#doc1 = ["This is a sentence", "This is another sentence"]
a=input("Enter the First sentence: ")
b=input("Enter the Second sentence: ")
doc1=[a,b]
# Transform data (you can add more data preprocessing steps) 

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)



result=	pearson_def(model.docvecs[0],model.docvecs[1])+1
#result=	pearson_def(a,b)+1
#result1 = 1 - spatial.distance.cosine(model.docvecs[0],model.docvecs[1])
print(result)