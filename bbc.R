# Installa e carica le librerie necessarie (per la latent semantic analysis e per il text mining).
install.packages( "lsa" ); library( lsa )
install.packages( "tm" ); library( tm )

# Imposta la directory di lavoro. 
setwd("/Users/billy/Documents/MAGISTRALE/2_ANNO/DM/Text Mining/PROJECT")

# Salva nella variabile "bbc" il contenuto del dataset "bbc-text.csv" e lo visualizza.
bbc <- read.csv( "bbc-text.csv" ); View( bbc )
# Salva nel vettore "categories" il topic di ogni articolo (attributo classe) e ne visualizza l'ordine.
categories = bbc$category; levels( categories )
# Salva nel vettore "texts" il contenuto testuale di ogni articolo.
texts = bbc$text
# Costruisce il corpus dai documenti degli articoli e ne visualizza la lunghezza.
corpus <- Corpus( VectorSource( texts ) ); length( corpus )

# Pre-processing dei testi.
# Converte tutte le parole in minuscolo.
corpus <- tm_map( corpus, tolower )
# Rimuove i segni di punteggiatura.
corpus <- tm_map( corpus, removePunctuation )
# Rimuove i numeri.
corpus <- tm_map( corpus, removeNumbers )
# Rimuove le stopwords scegliendole dal file “stopwords_en” fornito dalla libreria "tm".
corpus <- tm_map( corpus, removeWords, stopwords_en )

# Estrae dal corpus una matrice termini-documenti (ci sono 29723 termini, con sparsità del 100%).
tdmc <- TermDocumentMatrix( corpus ); tdmc
# Elimina i termini che compaiono in meno dell’5% nei documenti (rimangono 404 termini, con sparsità del 91%).
tdms <- removeSparseTerms( tdmc, 0.95 ); tdms
# Salva nella variabile "words" le parole selezionate (le feature).
words <- rownames( tdms )
# Converte "tdms" dall’attuale formato sparso a quello standard di R (matrice che contiene le occorrenze dei termini).
tdm <- as.matrix( tdms )
# Applicazione del term weighting con tf-idf alla matrice termini-documenti "tdm".
tdmle <- lw_logtf( tdm ) * ( 1-entropy( tdm ) )

# Funzione che calcola la norma di un vettore.
norm_vec <- function(x) sqrt( sum( x^2 ) )
# Calcola la norma di ogni termine della matrice termini-documenti.
norma_termini <- apply( tdmle, 1, norm_vec )

# Esegue la decomposizione SVD della matrice pesata "tdmle" usando la funzione lsa.
lsar <- lsa( tdmle )
# Calcola una matrice "tls" con le posizioni dei termini nello spazio latente. 
tls <- lsar$tk %*% diag( lsar$sk )
# Calcola un matrice "dls" con le posizioni dei documenti nello spazio latente.
dls <- lsar$dk %*% diag( lsar$sk )

# Calcola la norma di ogni termine dentro lo spazio lsa.
norma_termini_lsa <- apply( tls, 1, norm_vec )
# Aggiunge la colonna delle norme alla matrice tls.
tls_norma_termini = cbind( tls, norma_termini_lsa )
# Subset della matrice dei termini con norma > 1.25.
tls_ridotto <- subset( tls_norma_termini, norma_termini_lsa > 1.25 )[,-138]

# Crea un grafico delle parole con relative etichette nello spazio latente nelle dimensioni 1 e 2.
plot( tls, pch=20 ); text( tls, labels=words, cex=0.8, pos=1 )
# Rappresentazione grafica dei termini ridotti nelle dimensioni 1 e 2.
plot( tls_ridotto, pch=20 ); text( tls_ridotto, labels=words, cex=0.8, pos=1 )
# Grafico sulle dimensioni 2 (asse X) e 3 (asse Y), traccia in verde l'origine.
plot( tls[,-1], pch=20 ); text( tls[,-1], labels=words, cex=0.8, pos=1 ); points( 0, 0, pch=20, cex=3, col="green" )
# Visualizza in un grafico anche i documenti (sia sulle dimensioni 1-2 che sulle dimensioni 2-3).
plot( dls, pch=20, cex=0.8 )
plot( dls[,-1], pch=20, cex=0.8 )

# Funzione che associa a ciascuna categoria un colore predefinito.
# business->yellow; entertainment->orange; politics->red; sport->blue; tech->darkgray.
catecolor <- function(x) switch( x, "yellow", "orange", "red", "blue", "darkgray" )
# Si ottiene un vettore con i colori che deve assumere ciascun punto del grafico.
catecols <- sapply( categories, catecolor )
# Ogni punto assume la colorazione prestabilita della sua classe di appartenenza.
plot( dls[,-1], pch=20, cex=0.8, col=catecols )

# Funzione che trasforma una matrice normalizzando ciascuna riga (applica la norma 2 a ciascun vettore).
normrows <- function(x) x / apply( x, 1, norm, "2" )
# Calcola le matrici con righe normalizzate.
tlsn <- normrows( tls ); dlsn <- normrows( dls )
# Nuovo grafico con termini e documenti sovrapposti (per i termini si visualizzano solo le etichette).
# Dal grafico si può visualizzare come si sono formati cluster di punti molto chiari delle varie categorie da analizzare
# e si nota anche che le categorie "sport", "politics" e "tech" hanno anche una forma più chiara e semplice da analizzare 
# rispetto le categorie "entertainment" e "business".
plot( dlsn[,-1], pch=20, cex=0.8, col=catecols ); text( tlsn[,-1], labels=words, cex=0.8 ); points( 0, 0, pch=20, cex=3, col="green" )

####################################################
# Si analizzano gli articoli di categoria "sport"  #
####################################################

# Si nota che il termine "cup" è tra i più rappresentativi nella concentrazione dei documenti sportivi.
# Si verifica se nello spazio originale c'è una correlazione lessicale tra "cup" e la categoria "sport" (tabella di contingenza).
cup.cat <- table( tdm["cup",]>0, categories=="sport" )
# Assegna i nomi di righe e colonne alla tabella di contingenza.
dimnames( cup.cat ) <- list( cup = c( "No", "Si" ), sport = c( "No", "Si" ) )
# Effettua la verifica con test chiquadro sulla tabella di contingenza.
chisqtest = chisq.test( cup.cat, correct=FALSE )
# Forte correlazione tra termine "cup" e gli articoli di categoria "sport".
chisqtest; chisqtest$expected; chisqtest$observed

# Si cerca il termine semanticamente più vicino a "cup" nello spazio LSA (con la similarità coseno).
# Avendo norma alta, si sceglie "team".
associate( tls[,2:3], "cup", threshold=0.9 )

# Si riesegue il test sulla presenza contemporanea dei termini "cup" e "team".
tc.cat <- table( tdm["team",]>0 & tdm["cup",]>0, categories=="sport" )
dimnames( tc.cat ) <- list( team_cup = c( "No", "Si" ), sport = c( "No", "Si" ) )
# Forte dipendenza tra la presenza dei termini e la categoria "sport". 
chisq.test( tc.cat, correct=FALSE )

# Si ripete l'analisi nello spazio LSA (per cercare relazioni semantiche oggettive tra la coppia di termini e la categoria). 
tcq <- "team cup"
# Trasforma la query in un vettore con rappresentazione bag of words.
tcv <- query( tcq, words )
# Si applica il tf-idf.
tcle <- lw_logtf( tcv ) * ( 1-entropy( tdm ) )
# Si fa il fold_in del documento nello spazio lsa.
tcls <- t( tcle ) %*% lsar$tk

# Normalizzazione del vettore query.
tclsn <- normrows( tcls )
# Visualizzazione della query nel grafico nelle dimensioni 2 e 3.
# Oggettivamente la query è vicina ai termini "team" e "cup" (per angolo dall'origine) e alla zona
# ad alta concentrazione di articoli sportivi.
points( tclsn[2], tclsn[3], cex=3, pch=18, col="red" )

# Funzione che fa il coseno tra il vettore Q e la matrice X.
cosines <- function(X, Q) apply( X, 1, cosine, as.vector( Q ) )
# Definizione funzione che restituisce le posizioni degli N valori più elevati di un vettore X.
top <- function(X, N) order( X, decreasing=TRUE )[1:N]

# Elenco dei 5 documenti che risultano semanticamente più simili alla query "team cup" (considerando solo la dimensione 2 e 3).
# Si nota che alcuni documenti non sono realmente correlati alla query.
texts[ top( cosines( dls[,2:3], tcls[2:3] ), 5 ) ] 

# Si sceglie uno spazio LSA con più dimensioni e quindi con minore perdita d'informazione.
# Si individua il punto di knee nella sequenza degli autovalori visualizzati in precedenza.
fordiff <- function(x) x[ 2:length( x ) ] - x[ 1:( length( x ) - 1 ) ]
# Si applica la funzione di curvatura agli autovalori.
skd <- fordiff( lsar$sk ) # derivata prima
skdd <- fordiff( skd ) # derivata seconda
skcurv <- skdd[1:20] / ( 1+( skd[1:20] )^2 )^1.5
plot( 1:20, skcurv, type="b" )

# Il primo minimo locale è 5, perciò si selezionano le prime 5 dimensioni dello spazio LSA.
# Si nota che ora tutti gli articoli sono relativi ai termini "team" e "cup".
texts[ top ( cosines( dls[,1:5], tcls[1:5] ), 5 ) ]

# Visualizza il numero dei documenti per ogni categoria con la relativa %.
# Gli articoli sportivi sono 511 e sono il 23% dell'intero dataset.
table( categories ); 100 * table( categories ) / length( categories )

# R-precision, con N = 511, si hanno 477 categorie sportive (93,3%), nettamente maggiore al 23% del valore atteso.
table( categories[ top( cosines( dls[,1:5], tcls[1:5] ), 511 ) ] )

# Calcola la tabella di contingenza.
tcq.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], tcls[1:5] ), 511 ), categories=="sport" )
dimnames( tcq.cat ) <- list( team_cup = c( "No", "Si" ), sport = c( "No", "Si" ) )
# 477 articoli sportivi restituiti su 117 attesi in 511 articoli (511-precision).
# Questo indica alta correlazione  tra "team cup" e articoli sportivi.
tcq.cat

# Esegue una verifica oggettiva mediante test chiquadro (X^2= 1857.2 e p-value < 2.2e^-16, altissima correlazione).
chisq.test( tcq.cat, correct=FALSE )

# Ordina gli articoli per rilevanza semantica decrescente rispetto alla query tcls (visualizza le similarità coseno).
nearest <- sort( cosines( dls[,1:5], tcls[1:5] ), decreasing=T )
# Crea una lista con 2225 volte la parola black.
catecols2 <- rep( "black", 2225 )
# Pone a red gli articoli sportivi.
catecols2[ categories == "sport" ] <- "red"

# Dal grafico della similarità coseno si nota che gli articoli sportivi 
# sono quelli con maggiore similarità semantica con la query "team cup". 
plot( 1:511, nearest[1:511], pch=20, cex=0.7, col=catecols2[ strtoi( names( nearest ) ) ] )

# Si cercano altri termini semanticamente più correlati alla coppia "team cup", perciò si applica
# il calcolo della similarità tra la query e i termini.
tksrs <- lsar$tk %*% diag( sqrt( lsar$sk ) )
tcdksrs <- tcls %*% diag( lsar$sk^-0.5 )

# Elenco dei termini in ordine crescente di similarità semantica con la coppia "team cup".
sort( cosines( tksrs[,1:5], tcdksrs[1:5] ) ) 

# Si scelgono i termini con elevata similarità coseno e norma massima, perciò si estrae la norma 
# di quelli con similarità coseno a "team cup" maggiore di 0.99.  
tnorms5 <- apply( tls[,1:5], 1, norm, "2" )
# Il termine con norma maggiore è "coach" (più rilevante).
tnorms5[ cosines( tksrs[,1:5], tcdksrs[1:5] ) > 0.99 ]

# Funzione per velocizzare il calcolo della similarità tra query e termini in lsa.
makequery <- function( Q, TDM, LSA ) {
  V <- query( Q, rownames( TDM ) )
  W <- lw_logtf( V ) * ( 1-entropy( TDM ) )
  LS <- t( W ) %*% LSA$tk
  DK <- LS %*% diag( LSA$sk^-1 )
  DKSRS <- DK %*% diag( sqrt( LSA$sk ) )
  list( query=Q, bin=V, v=W, dk=DK, ls=LS, dksrs=DKSRS )
}

# Costruire una nuova query utilizzando la funzione appena definita. 
tcc <- makequery( "team cup coach", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], tcc$ls[1:5] ), 5 ) ]
# La distribuzione delle classi è uguale alla precedente.
table( categories[ top( cosines( dls[,1:5], tcc$ls[1:5] ), 511 ) ] )
# Calcola quanti documenti ci sono in comune tra quest'ultima query più specifica e la query precedente "team cup".
# 510 in comune su 511, quindi il risultato è il medesimo anche aggiungendo "coach".
length( intersect( top( cosines( dls[,1:5], tcls[1:5]), 511), top( cosines( dls[,1:5], tcc$ls[1:5]), 511 ) ) )

# Si sceglie il 4° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], tcc$dksrs[1:5] ) )
# Si sceglie il termine ireland.
tnorms5[ cosines( tksrs[,1:5], tcc$dksrs[1:5] ) > 0.98 ]

tcci <- makequery( "team cup coach ireland", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], tcci$ls[1:5] ), 10 ) ]
# La distribuzione delle classi è uguale alla precedente.
table( categories[ top( cosines( dls[,1:5], tcci$ls[1:5] ), 511 ) ] )
# Calcola quanti documenti ci sono in comune tra quest'ultima query più specifica e la query precedente "team cup".
# 508 in comune su 511, quindi il risultato è il medesimo anche aggiungendo "ireland".
length( intersect( top( cosines( dls[,1:5], tcls[1:5]), 511), top( cosines( dls[,1:5], tcci$ls[1:5]), 511 ) ) )

#####################################################################################################################
# Il topic più ricorrente per gli articoli sportivi è correlato ai termini "team cup coach ireland", ovvero negli   #
# articoli si parla delle interviste fatte a diversi allenatori (coach) relative ai match della loro squadra (team) #
# in diverse competizione di coppa (cup) contro l'Irlanda (ireland) o nel futuro scontro contro essa.               #                                                                   #
#####################################################################################################################


#######################################################
# Si analizzano gli articoli di categoria "politics"  #
#######################################################

# Viene visualizzato il grafico con termini e documenti sovrapposti.
plot( dlsn[,-1], pch=20, cex=0.8, col=catecols ); text( tlsn[,-1], labels=words, cex=0.8 ); points( 0, 0, pch=20, cex=3, col="green" )

# Si nota che il termine "party" è tra i più rappresentativi nella concentrazione dei documenti di politica.
# Si verifica se nello spazio originale c'è una correlazione lessicale tra "party" e la categoria "politics" (tabella di contingenza).
party.cat <- table( tdm["party",]>0, categories=="politics" )
# Assegna i nomi di righe e colonne alla tabella di contingenza.
dimnames( party.cat ) <- list( party = c( "No", "Si" ), politics = c( "No", "Si" ) )
# Effettua la verifica con test chiquadro sulla tabella di contingenza.
chisqtest = chisq.test( party.cat, correct=FALSE )
# Forte correlazione tra il termine "party" e gli articoli di categoria "politics".
chisqtest; chisqtest$expected; chisqtest$observed

# Si cerca il termine semanticamente più vicino a "party" nello spazio LSA (con la similarità coseno).
# Avendo norma alta, si sceglie "labour".
associate( tls[,2:3], "party", threshold=0.9 )

# Si riesegue il test sulla presenza contemporanea dei termini "labour" e "party".
lp.cat <- table( tdm["labour",]>0 & tdm["party",]>0, categories=="politics" )
dimnames( lp.cat ) <- list( labour_party = c( "No", "Si" ), politics = c( "No", "Si" ) )
# Forte dipendenza tra la presenza dei termini e la categoria "politics". 
chisq.test( tc.cat, correct=FALSE )

# Visualizza il numero dei documenti per ogni categoria con la relativa %.
# Gli articoli sulla politica sono 417 e sono quasi il 19% dell'intero dataset.
table( categories ); 100 * table( categories ) / length( categories )

# Costruisce una query. 
lp <- makequery( "labour party", tdm, lsar )
# Si sceglie il 3° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], lp$dksrs[1:5] ) )
# Si seleziona come termine "blair".
tnorms5[ cosines( tksrs[,1:5], lp$dksrs[1:5] ) > 0.98 ]
# Calcolo la tabella di contingenza rispetto alla query.
lp.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], lp$ls[1:5] ), 417 ), categories=="politics" )
dimnames( lp.cat ) <- list( labour_party = c( "No", "Si" ), politics = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( lp.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed

# Si crea una nuova query.
lpb <- makequery( "labour party blair", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], lpb$ls[1:5] ), 5 ) ]
table( categories[ top( cosines( dls[,1:5], lpb$ls[1:5] ), 417 ) ] )

# Calcolo la tabella di contingenza rispetto alla query.
lpb.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], lpb$ls[1:5] ), 417 ), categories=="politics" )
dimnames( lpb.cat ) <- list( labour_party_blair = c( "No", "Si" ), politics = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( lpb.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed

# Si sceglie il 4° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], lpb$dksrs[1:5] ) )
# Si seleziona come termine "tory".
tnorms5[ cosines( tksrs[,1:5], lpb$dksrs[1:5] ) > 0.98 ]

# Nuova query.
lpbt <- makequery( "labour party blair tory", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], lpbt$ls[1:5] ), 5 ) ]
table( categories[ top( cosines( dls[,1:5], lpbt$ls[1:5] ), 417 ) ] )

# Si sceglie il 5° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], lpbt$dksrs[1:5] ) )
# Si seleziona come termine "tories".
tnorms5[ cosines( tksrs[,1:5], lpbt$dksrs[1:5] ) > 0.98 ]

# Nuova query.
lpbtt <- makequery( "labour party blair tory tories", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], lpbtt$ls[1:5] ), 5 ) ]
table( categories[ top( cosines( dls[,1:5], lpbtt$ls[1:5] ), 417 ) ] )

#####################################################################################################################
# Il topic più ricorrente per gli articoli sulla politica è correlato ai termini "labour party blair tory tories",  #
# ovvero negli articoli si parla degli scontri tra il partito (party) laburista (labour) britannico con il politico #
# Tony Blair (blair) e il partito conservatore Tory (tory) con i suoi sostenitori (tories).                         #
#####################################################################################################################


#######################################################
# Si analizzano gli articoli di categoria "tech"      #
#######################################################

plot( dlsn[,-1], pch=20, cex=0.8, col=catecols ); text( tlsn[,-1], labels=words, cex=0.8 ); points( 0, 0, pch=20, cex=3, col="green" )

# Si nota che il termine "internet" è tra i più rappresentativi nella concentrazione dei documenti di tecnologia.
# Si verifica se nello spazio originale c'è una correlazione lessicale tra "internet" e la categoria "tech" (tabella di contingenza).
internet.cat <- table( tdm["internet",]>0, categories=="tech" )
# Assegna i nomi di righe e colonne alla tabella di contingenza.
dimnames( internet.cat ) <- list( internet = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Effettua la verifica con test chiquadro sulla tabella di contingenza.
chisqtest = chisq.test( internet.cat, correct=FALSE )
# Forte correlazione tra il termine "internet" e gli articoli di categoria "tech".
chisqtest; chisqtest$expected; chisqtest$observed

# Si cerca il termine semanticamente più vicino a "internet" nello spazio LSA (con la similarità coseno).
# Avendo norma alta, si sceglie "software".
associate( tls[,2:3], "internet", threshold=0.9 )

# Si riesegue il test sulla presenza contemporanea dei termini "internet" e "software".
is.cat <- table( tdm["internet",]>0 & tdm["software",]>0, categories=="tech" )
dimnames( is.cat ) <- list( internet_software = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Forte dipendenza tra la presenza dei termini e la categoria "tech". 
chisq.test( is.cat, correct=FALSE )

# Visualizza il numero dei documenti per ogni categoria con la relativa %.
# Gli articoli sulla tecnologia sono 401 e sono il 18% dell'intero dataset.
table( categories ); 100 * table( categories ) / length( categories )

# Costruisce una query. 
is <- makequery( "internet software", tdm, lsar )
# Calcolo la tabella di contingenza rispetto alla query.
is.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], is$ls[1:5] ), 401 ), categories=="tech" )
dimnames( is.cat ) <- list( internet_software = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( is.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed
# Si sceglie il 3° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], is$dksrs[1:5] ) )
# Si seleziona come termine "users".
tnorms5[ cosines( tksrs[,1:5], is$dksrs[1:5] ) > 0.98 ]

# Costruisce una query. 
isu <- makequery( "internet software users", tdm, lsar )
# Calcolo la tabella di contingenza rispetto alla query.
isu.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], isu$ls[1:5] ), 401 ), categories=="tech" )
dimnames( isu.cat ) <- list( internet_software_users = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( isu.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed
# Si sceglie il 4° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], isu$dksrs[1:5] ) )
# Si seleziona come termine "phone".
tnorms5[ cosines( tksrs[,1:5], isu$dksrs[1:5] ) > 0.97 ]

# Costruisce una query. 
isup <- makequery( "internet software users phone", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], isup$ls[1:5] ), 5 ) ]
# Calcolo la tabella di contingenza rispetto alla query.
isup.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], isup$ls[1:5] ), 401 ), categories=="tech" )
dimnames( isup.cat ) <- list( internet_software_users_phone = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( isup.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed
# Si sceglie il 5° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], isup$dksrs[1:5] ) )
# Si seleziona come termine "mobile".
tnorms5[ cosines( tksrs[,1:5], isup$dksrs[1:5] ) > 0.95 ]

# Costruisce una query. 
isupm <- makequery( "internet software users phone mobile", tdm, lsar )
# Primi 5 articoli più rilevanti per la nuova query.
texts[ top( cosines( dls[,1:5], isupm$ls[1:5] ), 15 ) ]
# Calcolo la tabella di contingenza rispetto alla query.
isupm.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], isupm$ls[1:5] ), 401 ), categories=="tech" )
dimnames( isupm.cat ) <- list( internet_software_users_phone_mobile = c( "No", "Si" ), tech = c( "No", "Si" ) )
# Esegue il test chiquadro.
chisqtest = chisq.test( isupm.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed

############################################################################################################################
# Il topic più ricorrente per gli articoli sulla tecnologia è correlato ai termini "internet software users mobile phone", #
# ovvero negli articoli si parla di servizi internet (internet, software), di varia natura, utilizzabili dagli utenti      #   
# (users) che possiedono telefoni cellulari (mobile phone).                                                                #                                                                      #
############################################################################################################################

############################################################
# Si analizzano gli articoli di categoria "entertainment"  #      
############################################################

plot( dlsn[,-1], pch=20, cex=0.8, col=catecols ); text( tlsn[,-1], labels=words, cex=0.8 ); points( 0, 0, pch=20, cex=3, col="green" )

# Si nota che il termine "film" è tra i più rappresentativi nella concentrazione dei documenti di intrattenimento.
# Si verifica se nello spazio originale c'è una correlazione lessicale tra "film" e la categoria "intrattenimento" (tabella di contingenza).
film.cat <- table( tdm["film",]>0, categories=="entertainment" )
# Assegna i nomi di righe e colonne alla tabella di contingenza.
dimnames( film.cat ) <- list( film = c( "No", "Si" ), entertainment = c( "No", "Si" ) )
# Effettua la verifica con test chiquadro sulla tabella di contingenza.
chisqtest = chisq.test( film.cat, correct=FALSE )
# Forte correlazione tra il termine "film" e gli articoli di categoria "entertainment".
chisqtest; chisqtest$expected; chisqtest$observed

# Si cerca il termine semanticamente più vicino a "film" nello spazio LSA (con la similarità coseno).
# Avendo norma alta, si sceglie "world".
associate( tls[,2:3], "film", threshold=0.9 )

# Si riesegue il test sulla presenza contemporanea dei termini "film" e "world".
fw.cat <- table( tdm["film",]>0 & tdm["world",]>0, categories=="entertainment" )
dimnames( fw.cat ) <- list( film_world = c( "No", "Si" ), entertainment = c( "No", "Si" ) )
# Forte dipendenza tra la presenza dei termini e la categoria "tech". 
chisq.test( fw.cat, correct=FALSE )

# Visualizza il numero dei documenti per ogni categoria con la relativa %.
# Gli articoli di intrattenimento sono 386 e sono il 17% dell'intero dataset.
table( categories ); 100 * table( categories ) / length( categories )

# Costruisce una query. 
fw <- makequery( "film world", tdm, lsar )
# Calcolo la tabella di contingenza rispetto alla query.
fw.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], fw$ls[1:5] ), 386 ), categories=="entertainment" )
dimnames( fw.cat ) <- list( film_world = c( "No", "Si" ), entertainment = c( "No", "Si" ) )
# Esegue il test chiquadro (molto correlato).
chisqtest = chisq.test( fw.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed
# Si sceglie il 3° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], fw$dksrs[1:5] ) )
# Si seleziona come termine "awards".
tnorms5[ cosines( tksrs[,1:5], fw$dksrs[1:5] ) > 0.99 ]

# Costruisce una query. 
fwa <- makequery( "film world awards", tdm, lsar )
# Calcolo la tabella di contingenza rispetto alla query.
fwa.cat <- table( 1:nrow( dls ) %in% top( cosines( dls[,1:5], fwa$ls[1:5] ), 386 ), categories=="entertainment" )
dimnames( fwa.cat ) <- list( film_world_awards = c( "No", "Si" ), entertainment = c( "No", "Si" ) )
# Esegue il test chiquadro (molto correlato).
chisqtest = chisq.test( fwa.cat, correct=FALSE )
chisqtest; chisqtest$expected; chisqtest$observed
# Si sceglie il 4° termine da aggiungere alla query.
sort( cosines( tksrs[,1:5], fwa$dksrs[1:5] ) )
# Non ci sono altri termini correlati, si termina l'analisi.
tnorms5[ cosines( tksrs[,1:5], fwa$dksrs[1:5] ) > 0.98 ]

############################################################################################################################
# Il topic più ricorrente per gli articoli di intrattenimento è correlato ai termini "film world awards", nei quali si     #
# parla dei premi (awards) cinematografici (film).                                                                         #                                                                      #
############################################################################################################################

