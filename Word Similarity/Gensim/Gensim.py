x=open("input.txt" , 'r')
line1=[]
line2=[]
for i in range(100):

	y=x.readline().split("\t")
	line1.append(y[3])
	line2.append(y[4])
	#next(x)

#print(line1[5])
#print(line2[5])

from gensim.models import doc2vec
from collections import namedtuple

# Load data

#doc1 = ["This is a sentence", "This is another sentence"]
#a=input("Enter the First sentence: ")
#b=input("Enter the Second sentence: ")
j=0
out = open("output.txt", 'w')
for i in range(100):
	doc1=[line1[i],line2[i]]
# Transform data (you can add more data preprocessing steps) 

	docs = []
	analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
	for i, text in enumerate(doc1):
		words = text.lower().split()
		tags = [i]
		docs.append(analyzedDocument(words, tags))

	# Train model (set min_count = 1, if you want the model to work with the provided example data set)

	model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
	
	# Get the vectors

	model.docvecs[0]
	model.docvecs[1]

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
	result=	cosine_similarity(model.docvecs[0],model.docvecs[1])+1
	result1 = 1 - spatial.distance.cosine(model.docvecs[0],model.docvecs[1])
	#lines = open("b.txt", 'r').readlines()
	#lines[j] = result
	s=str(result)
	out.write("\n")
	out.write(s)
	#j=j+1
	
	
	
#print(result1)