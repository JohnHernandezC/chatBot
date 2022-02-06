import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random
import pickle
#nltk.download('punkt')
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
print(palabras)
print(tags)
print(auxiliar1)
print(auxiliar2)