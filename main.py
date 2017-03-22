from __future__ import print_function
import numpy as np
from consts.commonConsts import CommonConsts
from main.characterTable import CharacterTable
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import layers
from keras.utils.np_utils import to_categorical
from dataset.dataset import DataSet

class MainClass:
    def __init__(self):
        self.model = Sequential()

        self.ctable = CharacterTable(CommonConsts.CHARS)
        print('Start application (Entry point)')

    def createmodel(self):
        print('Creating model...')
        self.model.add(LSTM(
            128,
            input_shape=(self.maxCount + 1 + self.maxCount, len(CommonConsts.CHARS)),
            batch_size=CommonConsts.BATCH_SIZE,
            return_sequences=False,
            stateful=True
        ))

        # self.model.add(LSTM(
        #     128,
        #     return_sequences=False,
        #     stateful=True
        # ))

        self.model.add(Dense(1))
        #self.model.add(layers.Activation('softmax'))

        self.model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['accuracy']
        )

    def preparesequence(self):
        self.questions = []
        self.answers = []

        # Nessary for correct filling space letters into answer and questions
        self.maxCount = 0
        print('Prepare init sequence...')
        '''data = 'aa aaa aah radiodontic radiodontics radiodontist radiodynamic radiodynamics radioecological radioecologist radioecology radioed radioelement radiofrequ trink trinkerman trinkermen trinket trinkets trinketed trinketer trinketing trinketries trinketry trinkets trinkety trinkle trinklement trinklet trinkum trinkums trinoctial trinocular trinoctile aahaa aahaa aahsshaa aalaa aaliiiilaa aaliis aals aam aardvark aardvarks aardwolf zymosterol zymosthenic zymotechnic zymotechnical zymotechnics zymotechny zymotic zymotically zymotize zymurgies zymotoxic zymurgy zythem zyzzyvas zythum zyzzyva aardwolves aargh aaron aaronic aarrgh aarrghh aas aasvogel aasvogels ab aba abac abaca abacas abacate abacaxi abacay abaci abacinate abacination abacisci abaciscus  aaa aaaaaa aba ababa abba acca ada adda adinida affa aga aha ajaja aka akka ala alala alula ama amma ana anana anna apa ara arara asa ata atta ava awa bab bib bob boob bub cdc civic crc dad deed deedeed degged deified deked deled denned dewed did dod dud eke elle eme ere esse eve ewe eye gag gig gog hah halalah hallah hannah huh ihi iii imi immi kaiak kakkak kassak kayak kazak keek kelek kkk kook lemel level maam madam malayalam mam marram mem mesem mim minim mom mum murdrum nan non noon nun ofo oho oto otto pap pdp peep pep pip poop pop pup radar redder refer reifier repaper retter rever reviver rotator rotor sagas samas sees selles sememes semes senones seres sexes shahs siris sis solos sooloos sos stats stets sus tat tebbet tebet teet tenet terret tit tnt toot tot tst tut tyt ulu ululu umu uru utu vav waw wow xix xxx yaray yoy yay gregarine gregarinian gregarinidal gregariniform gregarinosis gregarinous gregarious gregariously gregariousness gregaritic gregatim gregau grege greggle greggriffin grego gregor gregorian gregorianist gregory gregos greige greiges greillade grein greing greisen greisens greit greith greking grelot gremial gremiale gremials gremio gremlin gremlins gremmie gremmies gremmy grenada grenade grenades grenades grenadier grenadierial grenadierly grenadiers grenadiership grenadilla grenadin grenadine grenadines grenado grenat grenatite grene grenier gres gresil gressible gressorial gressorious gret greta grete greund grew grewhound grewsome grewsomely grewsomeness grewsomer grewsomest grewt grex grey greyback greybeard greycoat greyed greyer greyest greyfish greyflies greyfly greyhen greyhens greyhound greyhounds greying greyish greylag greylags greyling greyly greyness greynesses greypate greys grid greyskin greystone greywacke greyware greywether grf gribane gribble gribbles grice'

        self.supportsLetters = data.split(' ')
        '''
        dataset = DataSet()
        self.supportsLetters = dataset.getData()
        self.supportsLetters.sort()
        check = 0
        for word in self.supportsLetters:
            # Prepare new question list with normal word and possible polinom
            wordLetterList = list(word)
            palindrom = ""

            if wordLetterList.__len__() > self.maxCount:
                self.maxCount = wordLetterList.__len__()

            for i in range(wordLetterList.__len__() - 1, -1, -1):
                palindrom += wordLetterList[i]

            if word == palindrom:
                correct = 1
            else:
                correct = 0

            question = "%s|%s" % (word, palindrom)
            self.questions.append(question)

            self.answers.append(correct)

    def vectorization(self):
        print("Start vectorization process...")
        x = np.zeros((len(self.questions), self.maxCount + 1 + self.maxCount, len(CommonConsts.CHARS)), dtype=np.int)
        y = self.answers
        for i, sentence in enumerate(self.questions):
            x[i] = self.ctable.encode(sentence, self.maxCount + 1 + self.maxCount)

        # Shuffle (x, y) in unison as the later parts of x will almost all be larger
        # digits.
        #indices = np.arange(len(y))
        #np.random.shuffle(indices)
        #x = x[indices]


        # Explicitly set apart 10% for validation data that we never train over.
        split_at = len(x) - len(x) // 10

        (self.x_train, self.x_val) = x[:split_at], x[split_at:]
        (self.y_train, self.y_val) = y[:split_at], y[split_at:]

    def startcalculations(self):
        print('Start calculation')

        self.preparesequence()
        self.vectorization()
        self.createmodel()
        print('Training')
        for i in range(1, CommonConsts.EPOCHS + 1):
            print('Epoch %d/%d' % (i, CommonConsts.EPOCHS))
            self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=CommonConsts.BATCH_SIZE,
                verbose=1,
                epochs=1,
                shuffle=False,
                validation_data=(self.x_val, self.y_val)
            )
            self.model.reset_states()
            for j in range(10):
                ind = np.random.randint(0, len(self.x_val))
                rowx, rowy = self.x_val[np.array([ind])], self.y_val[ind]
                preds = self.model.predict_classes(rowx, verbose=0)
                q = self.ctable.decode(rowx[0])
                correct = rowy
                guess = preds[0]
                if correct == guess:
                    print('\033[92m')
                else:
                    print('\033[91m')

                print('Q', q[::-1])
                print('T', correct)
                print('G', guess)
                print('---', '\033[0m')

    def printresult(self):
        print("we are expected: %s" % "")
        print("\n We are get: %s" % "")


mainClass = MainClass()
mainClass.startcalculations()
mainClass.printresult()
