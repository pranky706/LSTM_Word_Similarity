from rouge import Rouge

#hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you saw on cnn student news"

#reference = "this page includes the show transcript use the transcript to help students with reading comprehension and vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests students ' knowledge of even ts in the news"

#a=input("Enter the first sentence: ")
#b=input("Enter the second sentence: ")


a1=open("input1.txt","r")
a=a1.readline()

b1=open("input2.txt","r")
b=b1.readline()

rouge = Rouge()
scores = rouge.get_scores(a,b)


#print("Similarity Score= ",scores)

c1=open("output.txt","w")
c1.write(str(scores))