from keras.preprocessing.text import Tokenizer, text_to_word_sequence
t1=input("Enter the first text: ")
t2=input("Enter the second text: ")

texts=[t1,t2]

tknzr = Tokenizer(lower=True, split=" ")
tknzr.fit_on_texts(texts)
	
X = tknzr.texts_to_matrix(texts)


#print(X[0])



import math
from scipy import spatial
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
result=	cosine_similarity(X[0],X[1])+1
#result1 = 1 - spatial.distance.cosine(model.docvecs[0],model.docvecs[1])
print("The Similarity between is=",result)