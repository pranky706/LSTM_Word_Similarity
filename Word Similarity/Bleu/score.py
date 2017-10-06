import nltk

#a = ['This', 'is', 'a', 'sentence'] 
#b = ['This', 'is', 'another', 'one']

#a=input("Enter the first sentence: ")
#b=input("Enter the second sentence: ")


a1=open("input1.txt","r")
a=a1.readline()

b1=open("input2.txt","r")
b=b1.readline()


BLEUscore = nltk.translate.bleu_score.sentence_bleu(a, b, weights = [1])
#print("Similarity Score=",BLEUscore)

c1=open("output.txt","w")
c1.write(str(BLEUscore))

