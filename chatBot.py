from turtle import shape
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer =LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle
#nltk.download('punkt')

#########INICIAR ARHIVO I FILTRAR LOS DATOS DENTRO###################
with open("cont.json") as respuestas:
    datos= json.load(respuestas)
print (datos)

palabras=[]
tags=[]
auxiliar1=[]
auxiliar2=[]
for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        aux=nltk.word_tokenize(patrones)#toma una frase la separa en palabras
        palabras.extend(aux)
        auxiliar1.append(aux)
        auxiliar2.append(contenido["tag"])
        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])
#####################################################################

palabras=[stemmer.stem(w.lower())for w in palabras if w !="?"]
palabras= sorted(list(set(palabras)))

tags=sorted(tags)
training=[]
output=[]
outputNull=[0 for  _ in range(len(tags))]

for x, document in enumerate(auxiliar1):#x guarda el indice y document la palabra
    cube=[]
    auxWord=[stemmer.stem(w.lower())for w in document]
    for w in palabras:
        if w in auxWord:
            cube.append(1)
        else:
            cube.append(0)
    exitRow=outputNull[:]
    exitRow[tags.index(auxiliar2[x])]=1
    training.append(cube)
    output.append(exitRow)
    
#print(training)
#print(output)
    
    
#CREACION DE RED NEURONAL
training=numpy.array(training)
output=numpy.array(output)

tensorflow.compat.v1.reset_default_graph()
#tensorflow.reset_default_graph()
red= tflearn.input_data(shape=[None,len(training[0])])#entrada que hace entrenamiento
red= tflearn.fully_connected(red,10)
red= tflearn.fully_connected(red,10)
red= tflearn.fully_connected(red,len(output[0]),activation="softmax")
red= tflearn.regression(red)

#CREACION DE MODELO

model= tflearn.DNN(red)
model.fit(training,output,n_epoch=2000,batch_size=10,show_metric=True)
model.save("model.tflearn")
#despues de ejecutar se puede ver la efectividad
#Training Step: 2000  | total loss: ←[1m←[32m0.01018←[0m←[0m
#| Adam | epoch: 2000 | loss: 0.01018 - acc: 1.0000 -- iter: 9/9 
#acc: 1.0000 maximo de efectividad

def botChat():
    while True:
        entrada= input("Tu:" )
        cube= [0 for _ in range(0,len(palabras))]
        inputProcessed=nltk.word_tokenize(entrada)#permite separar los signos especiales como ?
        inputProcessed=[stemmer.stem(palabra.lower()) for palabra in inputProcessed]
        for individualword in inputProcessed:
            for i,palabra in enumerate(palabras):
                if palabra==individualword:
                    cube[i]=1
        result=model.predict([numpy.array(cube)])
        resultIndex=numpy.argmax(result)#devuelv el tag con mayot probabilidad
        tag=tags[resultIndex]
        print(result)
        
        for tagAux in datos["contenido"]:
            if tagAux["tag"]==tag:
                answers=tagAux["respuestas"]
        print("Bot: ",random.choice(answers))
botChat()