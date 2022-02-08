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
    
print(training)
print(output)
    