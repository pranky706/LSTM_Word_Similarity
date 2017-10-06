from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
a=[]
text1=[]
text2=[]
texts=[]
x=open("b.txt" , 'r')
for i in range(4077):
	y=x.readline().split("\t")
	text1.append(y[0])
	text2.append(y[1])
	texts.append(y[0])
	texts.append(y[1])

tknzr = Tokenizer(lower=True, split=" ")
#tknzr2.fit_on_texts(text1)
#tknzr3.fit_on_texts(text2)
tknzr.fit_on_texts(texts)

	
X = tknzr.texts_to_matrix(text1)
X=np.array(X)
#data=data.reshape(1,-1)


Y= tknzr.texts_to_matrix(text2)
Y=np.array(Y)
Y = Y.reshape((-1,1))
Y=[[Y[i] for i in range(4077)]]
Y=np.array(Y)
Y = Y.reshape((-1,1))

vocab_size = len(tknzr.word_index) + 1
#vocab_size=vocab_size*50

model = Sequential()
model.add(Dense(1, input_dim=vocab_size))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X, Y, batch_size=64, nb_epoch=20,shuffle=True)









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
	
	
p=open("a.txt","w")
for i in range(4077):
	x1=tknzr.texts_to_matrix(text1[i])
	a.append(model.predict(x1))
	x2=tknzr.texts_to_matrix(text2[i])
	a.append(model.predict(x2))
	result=	cosine_similarity(a[0][:10],a[1][:10])
	p.write(text1[i])
	p.write("\t")
	p.write(text2[i])
	p.write("\t")
	p.write(str(result))
	p.write("\t")
	p.write(str(result*100))
	p.write("\n")






