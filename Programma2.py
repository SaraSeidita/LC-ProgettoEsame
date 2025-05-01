#!/usr/bin/python
# -*- coding: utf-8 -*-

# Linguistica Computazionale - Progetto d'esame 2019/2020 #
# Sara Seidita 579517 #

# Programma 2 #

import sys
import codecs 
import nltk
import re 

# Prima funzione: calcolo il numero dei token presente nel testo
def NumeroToken(phrases):
	TotalTokens = []
	for phrase in phrases: 
		tokens = nltk.word_tokenize(phrase)
		TotalTokens = TotalTokens + tokens
	return TotalTokens
	
# Punto principale del programma: Estrarre le frasi in cui compaiono solo i nomi propri

def EstraiFrase(phrases):
	Names = DieciNomiPropri(phrases) # qua richiamo la funzione DieciNomiPropri perché viene richiesto di estrarre 
	# i 10 nomi propri più frequenti 
	ListaA = []
	ListaB = []
	# La lista A è una lista vuota che contiene Nomi e frasi 
	# La lista B è la lista finale che conterrà solo le frasi che hanno i nomi propri 
	for phrase in phrases:
		tokens = nltk.word_tokenize(phrase)
		tokenPOS = nltk.pos_tag(tokens)
		analisi = nltk.ne_chunk(tokenPOS)
		for nodo in analisi:
			NE = ""
			if hasattr(nodo, "label"):
				if nodo.label() in ["PERSON"]:
					for partNE in nodo.leaves():
						NE = NE + " " + partNE[0] 
					for elem in Names:
						if elem[0]==NE:
							ListaA = ListaA + [(elem[0], phrase)] 
	for (name, phrase) in ListaA:
		ListaB = ListaB + [(phrase)]		
	return ListaB
	
# Punto 2-3: 10 nomi propri più frequenti, 10 luoghi più frequenti #

def DieciNomiPropri(phrasesN):
	listName = []
	NomiFreq = []
	for phrase in phrasesN:
		tokens = nltk.word_tokenize(phrase)
		tokenPOS = nltk.pos_tag(tokens)
		analisi = nltk.ne_chunk(tokenPOS)
		IOBformat = nltk.chunk.tree2conllstr(analisi)
		for nodo in analisi: 
			NE = "" 
			if hasattr(nodo, "label"):
				if nodo.label() in ["PERSON"]:
					for partNE in nodo.leaves():
						NE = NE + " " + partNE[0] 
					listName.append(NE)
		# ora calcolo la frequenza per estrarre i dieci nomi propri più frequenti #
		freq = nltk.FreqDist(listName)
		NomiFreq = freq.most_common(10)  # uso la distribuzione di frequenza per trovare i 10 nomi propri 
	return NomiFreq

def DieciLuoghi(phrasesL):
	# come per DieciNomiPropri, anche per DieciLuoghi, ma qua estraggo i 10 luoghi più frequenti 
	listPlaces = []
	LuoghiFreq = []
	for phrase in phrasesL:
		tokens = nltk.word_tokenize(phrase)
		tokenPOS = nltk.pos_tag(tokens)
		analisi = nltk.ne_chunk(tokenPOS)
		IOBformat = nltk.chunk.tree2conllstr(analisi)
		for nodo in analisi: 
			NE = "" 
			if hasattr(nodo, "label"):
				if nodo.label() in ["GPE"]:
					for partNE in nodo.leaves():
						NE = NE + " " + partNE[0] 
					listPlaces.append(NE)
		# ora calcolo la frequenza per estrarre i dieci nomi propri più frequenti #
		freq = nltk.FreqDist(listPlaces)
		LuoghiFreq = freq.most_common(10) 
	return LuoghiFreq

# Punto 4-5: i 10 sostantivi più frequenti e i 10 verbi più frequenti

def SostantiviVerbi(StestoPOS, VtestoPOS):
	# Inizializzo la listaSostantivi e listaVerbi e le due liste finali 
	# Poi con un ciclo for cerco i sostantivi/verbi presenti nel testo e usando FreqDist cerco i 10 più frequenti
	ListaS = []
	ListaV = []
	sostantivi = []
	verbi = []
	# sostantivi
	for(tokS, posS) in StestoPOS:
		if posS in ["NN", "NNS", "NNP", "NNPS"]:
			ListaS.append(tokS)
		FreqS = nltk.FreqDist(ListaS)
		sostantivi = FreqS.most_common(10)		
	# verbi 
	for(tokV, posV) in VtestoPOS:
		if posV in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
			ListaV.append(tokV)
		FreqV = nltk.FreqDist(ListaV)
		verbi = FreqV.most_common(10)
	 
	return "Sostantivi:", sostantivi, "Verbi:", verbi 



def main(file1, file2):
	# lettura dei file #

	fileInput1 = codecs.open(file1, 'r', 'utf-8')
	fileInput2 = codecs.open(file2, 'r', 'utf-8')
	raw1 = fileInput1.read()
	raw2 = fileInput2.read()

	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	frasi1 = sent_tokenizer.tokenize(raw1)
	frasi2 = sent_tokenizer.tokenize(raw2)
	
	F1 = EstraiFrase(frasi1) 
	F2 = EstraiFrase(frasi2)
	
	# 10 luoghi più frequenti presenti nelle frasi in cui compare il nome proprio 
	Luoghi1 = DieciLuoghi(F1)
	Luoghi2 = DieciLuoghi(F2)
	
	# 10 nomi di persona presenti nelle frasi in cui compare il nome proprio 
	Persone1 = DieciNomiPropri(F1)
	Persone2 = DieciNomiPropri(F2)
	
	# CALCOLO DEL NUMERO DEI TOKEN #
	
	# numero token e numero frasi dove ci sono i nomi proprio, invoco la prima funzione che si trova a inizio programma 
	TokenNumber1 = NumeroToken(F1)
	TokenNumber2 = NumeroToken(F2)
	
	# calcolo dei token in tutto il corpus #
	Testo1 = NumeroToken(frasi1)
	Testo2 = NumeroToken(frasi2)
	
	# Ad ogni token applichiamo #
	POS1 = nltk.pos_tag(TokenNumber1)
	POS2 = nltk.pos_tag(TokenNumber2)
	
	# I 10 sostantivi e 10 verbi più frequenti 
	SV1 = SostantiviVerbi(POS1, POS1)
	SV2 = SostantiviVerbi(POS2, POS2)


	# OUTPUT #
	file1 = 'Alice Adventures in Wonderland by Lewis Carroll'
	file2 = 'Peter Pan by J. M. Barrie'
		
	# FILE 1 #
	print (" PRIMO FILE ")
	print ("Per il file", file1)
	print ("-")
	
	print (" "*10)
	print ("-"*10)
	print ("Dieci nomi propri piu frequenti del primo file")
	for nomipropriP1 in Persone1:
		print nomipropriP1[0], nomipropriP1[1]
	print (" "*10)
	print ("-"*10)
	print ("Dieci luoghi piu frequenti")
	for luoghiP1 in Luoghi1:
		print luoghiP1[0], luoghiP1[1]
	print (" "*10)
	print ("-"*10)
	print ("Dieci sostantivi e verbi piu frequenti")
	print ("I dieci sostantivi e verbi piu frequenti del file sono:", SV1)	
	print (" "*10)
	
	# FILE 2 #
	print (" SECONDO FILE ")
	print ("-")
	print ("Per il file", file2)
	print (" "*10)
	print ("-"*10)
	print ("Dieci nomi propri piu frequenti")
	for nomipropri2 in Persone2:
		print nomipropri2[0], nomipropri2[1]
	print (" "*10)
	print ("-"*10)
	print ("Dieci luoghi piu frequenti")
	for luoghi2 in Persone2:
		print luoghi2[0], luoghi2[1]
	print (" "*10)
	print ("-"*10)
	print ("Dieci sostantivi e verbi piu frequenti")
	print ("I dieci sostantivi e verbi piu frequenti del file sono:", SV2)
	print (" "*10)	
	
main(sys.argv[1], sys.argv[2]) 




