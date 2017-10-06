#sent = "cats running ran cactus cactuses cacti community communities"
x=input("Enter the sentence: ")
from nltk.stem import PorterStemmer, WordNetLemmatizer
port = PorterStemmer()
a=" ".join([port.stem(i) for i in x.split()])
print(a)