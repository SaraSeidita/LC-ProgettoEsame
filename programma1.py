#!/usr/bin/python
# -*- coding: utf-8 -*-

# Linguistica Computazionale - Progetto d'esame 2019/2020 #
# Sara Seidita 579517 #

# Programma 1 #

import sys
import codecs 
import nltk
from nltk import bigrams
import math

# Punto N°1: il numero totale di frasi e di token #

def CalcolaNumero(phrases):
	# in questa funzione calcolo prima il numero totale delle frasi, poi il numero totale dei tokens
	# creo prima la lista delle frasi, poi divido le frasi in token e infine creo la lista dei token 
	TotalPhraseNumber = 0
	TotalTokensNumber = 0
	for phrase in phrases:
		TotalPhraseNumber = TotalPhraseNumber + len(phrases)
		tokens = nltk.word_tokenize(phrase)
		TotalTokensNumber = TotalTokensNumber + len(tokens)
	return 'Frasi:', TotalPhraseNumber, 'Token:', TotalTokensNumber

# Punto N°2. Calcolo della lunghezza media delle frasi e delle parole token in termini di caratteri #

def CalcolaLunghezzaM(phrases):
	# in questo programma calcolo la lunghezza media delle frasi e dei tokens 
	# come per la funziona CalcolaNumero, creo le liste delle frasi e dei token in un ciclo for 
	# infine calcolo la lunghezza media 
	NPhrase = 0.0
	NToken = 0.0
	for phrase in phrases:
		NPhrase = NPhrase + 1
		tokens = nltk.word_tokenize(phrase)
		NToken = NToken + len(tokens)
		
	MediumLengthPhrases = NToken / NPhrase # lunghezza media delle frasi
	MediumLengthTokens = len(tokens) / NToken # lunghezza media dei token
	return 'Frasi:', MediumLengthPhrases, 'Token:', MediumLengthTokens

# Punto N°3. la grandezza del vocabolario e la distribuzione degli hapax all'aumentare del corpus per #
# porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.); # 

def ContHapax(TotalTokens, vocab):
	# questa funzione serve per calcolare gli hapax nel vocabolario, da richiamare nella funzione incrHapax
	c = 0
	for tok in vocab:
		FreqToken = TotalTokens.count(tok)
		if FreqToken == 1: # un hapax è un token che ha occorrenza 1 in un testo, quindi la frequenza del token deve essere uguale a 1
			c = c + 1 
	return c

def incrHapax(TotalTokens):
	# In questa funzione prima inizializzo la lista: degli Hapax (hapaxList), la lista incrementata (incrList) e della lista del vocabolario (vocList)
	# il secondo passo è fare un contatore della lista che conta i token in un range da x a x+1000, creando una sottolista dei 1000 token letti 
	# infine in un ciclo for dove calcolo la distribuzione di hapax, richiamando la funzione ContHapax 
	hapaxList = [] 
	incrList = []
	vocList = []
	CList = [TotalTokens[x:x+1000] for x in range(0, len(TotalTokens),1000)] 
	for lista in CList:
		incrList = incrList + lista
		vocList = set(incrList)
		NumberHapax = ContHapax(incrList, vocList) # qua richiamo la funzione ausiliare ContHapax 
		hapaxList.append(NumberHapax) # appeno il numero trovato 
	return hapaxList 
	
# 4-5. Part-of-Speech: rapporto tra sostantivi e verbi; le 10 PoS (Part-of-Speech) più frequenti; 

def AnnotazioneLinguistica(frasi):
	tokensPOSTot = []
	tokensTOT = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPOS = nltk.pos_tag(tokens)
		tokensTOT = tokensTOT + tokens
		tokensPOSTot = tokensPOSTot + tokensPOS
	return tokensPOSTot

def EstraiSequenzaPOS(TestoAnalizzatoPOS):
	listaPOS = []
	for bigramma in TestoAnalizzatoPOS:
		listaPOS.append(bigramma[1])
	return listaPOS

#Dopo aver raccolto tutte le POS nelle due funzioni precedenti, conto quanti sono nomi e quanti sono verbi: dividendo, ottengo il rapporto Sostantivi/verbi
def RapportoSV(SequenzaPOS):
	sostantivo = 0.0 # inizializzo sostantivo e verbo 
	verbo = 0.0
	for element in SequenzaPOS:
		# conto i sostantivi e verbi
		if element == ("NN" or "NNS" or "NP" or "NPS"):
			sostantivo = sostantivo + 1
		if element == ("VB" or "VBD" or "VBG" or "VBN" or "VBN" or "VBP" or "VBZ"):
			verbo = verbo + 1

	rapporto = float(sostantivo)/float(verbo) # calcolo finale

	return rapporto

# Funzione per raccogliere le 10 PoS più frequenti 

def DieciPOS(tokensPOS):
	# Questa funzione serve per calcolare i 10 PoS più frequenti, usando freqDist.most_common 
    seqPOS = EstraiSequenzaPOS(tokensPOS)
    freqDist = nltk.FreqDist(seqPOS)
    return freqDist.most_common(10)

# 6. PoS: estraete ed ordinate i 10 bigrammi di PoS:
# 		con probabilità condizionata massima, indicando anche la relativa probabilità;
#		con forza associativa massima (calcolata in termini di Local Mutual Information), indicando anche la relativa forza associativa.

# funzioni ausiliari #

def ordina(listaDaOrdinare):
	# funzione ausiliare per ordinare i 10 bigrammi di PoS #
	return sorted(listaDaOrdinare, reverse = True)
	
def frequenzaAttesa(a,b,N):
	# funzione ausiliare per calcolare la frequenza attesa, in quanto serve per calcolare la condizionata massima e la LMI #
	totale = ((a*10)*(b*10))/(N*1.0)
	return totale

# funzioni principali #

def condizionataMassima(tokenPOS, bigramList):
	listaBigrammiPC = [] # inizializzo la lista dei bigrammi per la probabilità condizionata 
	# calcolo poi la distribuzione di frequenza (distrFreq)
	N = len(tokenPOS)
	distrFreq = nltk.FreqDist(bigramList)
	for bigramma in distrFreq:
		freqA = tokenPOS.count(bigramma[0])
		freqB = tokenPOS.count(bigramma[1])
		freqAttesa = frequenzaAttesa(freqA,freqB,N) # richiamo la funzione ausiliare frequenzaAttesa 
		PCOND = (freqAttesa/N*1.0)*100 # calcolo della Probabilità Condizionata
		listaBigrammiPC.append([PCOND,bigramma]) # appendo 
		listaBigrammiPC = ordina(listaBigrammiPC) # richiamo la funzione "ordina" per riordinare la lista dei bigrammi 
	return listaBigrammiPC

def calcoloLMI(tokenPOS, bigramList):
	bigramList = bigrams(tokenPOS)
	# inizializzo il dizionario, la lista finale e LMI #
	Dictionary = []
	finalList = []
	LMI = 0 
	N = len(tokenPOS)
	distrFreq = nltk.FreqDist(bigramList)
	for bigramma in distrFreq:
		# dentro questo ciclo for calcolo la FO (Frequenza Osservata) e la FA (Frequenza Attesa)
		# Esse servono poi per calcolare la LMI 
		A = tokenPOS.count(bigramma[0])
		B = tokenPOS.count(bigramma[1])
		FO = distrFreq[bigramma] # Frequenza osservata #
		FA = frequenzaAttesa(A,B,N) # Frequenza attesa #
		LMI = (FO*1.0)*math.log((FO*1.0)/(FA*1.0),2) # Calcolo della LMI #
		finalList.append([LMI, bigramma])
		finalList = ordina(finalList) #riordino la lista 
		
	return finalList

# FUNZIONE MAIN #
 
def main(file1, file2):
	# Lettura dei due file 
	fileInput1 = codecs.open(file1, 'r', 'utf-8')
	fileInput2 = codecs.open(file2, 'r', 'utf-8')
	raw1 = fileInput1.read()
	raw2 = fileInput2.read()
	# carico il tokenizzatore di NLTK
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	# divido il file in frasi 
	frasi1 = sent_tokenizer.tokenize(raw1)
	frasi2 = sent_tokenizer.tokenize(raw2)
	
	# numero token e numero frasi 
	NumeroToken1 = CalcolaNumero(frasi1)
	NumeroToken2 = CalcolaNumero(frasi2)
	
	# lunghezza media frasi e token  
	LunghezzaM1 = CalcolaLunghezzaM(frasi1)
	LunghezzaM2 = CalcolaLunghezzaM(frasi2)
	
	# vocabolario e hapax 
	HapaxTxt1 = incrHapax(frasi1) #Invoco la funzione che incrementa gli Hapax ogni 1000
	HapaxTxt2 = incrHapax(frasi2) 
	
	# PoS
	TestoAnalizzatoPOS1 = AnnotazioneLinguistica(frasi1)
	TestoAnalizzatoPOS2 = AnnotazioneLinguistica(frasi2)

	SequenzaPOS1 = EstraiSequenzaPOS(TestoAnalizzatoPOS1)
	SequenzaPOS2 = EstraiSequenzaPOS(TestoAnalizzatoPOS2)
	
	# RAPPORTO S/V
	rapporto1 = RapportoSV(SequenzaPOS1)
	rapporto2 = RapportoSV(SequenzaPOS2)
	
	# DIECI POS FREQUENTI 
	DieciPosFrequenti1 = DieciPOS(TestoAnalizzatoPOS1)
	DieciPosFrequenti2 = DieciPOS(TestoAnalizzatoPOS2)
	
	# LMI 
	bigrammiPOS1 = bigrams(DieciPosFrequenti1)
	bigrammiPOS2 = bigrams(DieciPosFrequenti2)
	probCond1 = condizionataMassima(DieciPosFrequenti1, bigrammiPOS1)
	probCond2 = condizionataMassima(DieciPosFrequenti2, bigrammiPOS2)
	LMI1 = calcoloLMI(DieciPosFrequenti1, bigrammiPOS1)
	LMI2 = calcoloLMI(DieciPosFrequenti2, bigrammiPOS2)
	
	
	# OUTPUT # 
	file1 = 'File 1 (Alice nel Paese delle Meraviglie)'
	file2 = 'File 2 (Peter Pan)'
	
	print ' '*20
	print ' '*20
	print '1. NUMERO TOTALE DELLE FRASI E DI TOKEN ------'
	print ' '
	print '\tNumero frasi e numero token del', file1, ':\n', NumeroToken1
	print '\t\nNumero frasi e numero token del', file2, ':\n', NumeroToken2
	print '  '
	print '-'*20
	print '  '
	print '2. LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN E LA LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERE -----'
	print ' '
	print '\tLunghezza media frasi e token del', file1, ':\n', LunghezzaM1
	print '\t\nLunghezza media frasi e token del', file2, ':\n', LunghezzaM2
	print ' '
	print '-'*20
	print ' '
	print '3. GRANDEZZA DEL VOCABOLARIO E DISTRIBUZIONE DEGLI HAPAX PER AUMENTO DI PORZIONI INCREMENTALI DI 1000 TOKEN -----'
	
	print '\t Incremento degli hapax del', file1, ':\n', HapaxTxt1
	print '\t Incremento degli hapax del', file2, ':\n', HapaxTxt2

	print ' '
	print '-'*20
	print '4. IL RAPPORTO TRA SOSTANTIVI E VERBI ------'
	print ' ' 
	print '\tIl rapporto S/V del', file1, ':\n', rapporto1
	print '\t\nIl rapporto S/V del', file2, ':\n', rapporto2
	print ' '
	print '-'*20
	print ' '
	print '5. I DIECI POS PIU FREQUENTI------'
	print ' '
	print '\tI dieci PoS piu frequenti del', file1
	for pos in DieciPosFrequenti1:
		print '\t', pos[0], '\toccorre\t:', pos[1], 'volte'
	print '\t\nI dieci PoS piu frequenti del', file2
	for pos in DieciPosFrequenti2:
		print '\t', pos[0], '\toccorre\t', pos[1], 'volte'
	print ' '
	print '\t 6. PROBABILITà CONDIZIONATA MASSIMA \n'
	print ' '
	print '\t\t\t', file1, ':'
	for elem1 in probCond1:
		print '\t', elem1[1][0], elem1[1][1], '\t\t:', elem1[0]
	print ' '
	print '\t\t\t', file2, ':'
	print ' '
	for elem2 in probCond2:	
		print '\t', elem2[1][0], elem2[1][1], '\t\t:', elem2[0]

	print ' '*10
	
	print '\t\n 7. FORZA ASSOCIATIVA MASSIMA CALCOLATA IN TERMINI DI LMI'
	print '\t\t\t', file1, ':'
	for elem1 in LMI1:
		print '\t', elem1[1][0], elem1[1][1], '\t\t:', elem1[0]
	print ' '
	print '\t\t\t', file2, ':'
	print ' '
	for elem2 in LMI2:	
		print '\t', elem2[1][0], elem2[1][1], '\t\t:', elem2[0]
	print ' '
	print

main(sys.argv[1], sys.argv[2])
	
	

