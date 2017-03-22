from consts.commonConsts import CommonConsts
import numpy as np


class DataSet:
    def __init__(self):
        print("Not implement")

    def getData(self, countRows=50000, maxLetterInWord=10, polindromPeriod=50):
        dataset = []
        currentPolindromPeriond = 0
        flag = False
        for i in range(0, countRows):
            lettersCount = np.random.randint(1, maxLetterInWord)
            word = ""
            if currentPolindromPeriond == polindromPeriod and lettersCount % 2 == 0:
                lettersCount /= 2
                flag = True

            for j in range(0, lettersCount):
                letter = np.random.randint(0, CommonConsts.CHARS.__len__() - 2)
                word += CommonConsts.CHARS[letter]

            if flag:
                flag = False
                currentPolindromPeriond = 0
                bufWord = ""
                for letIndex in range(word.__len__() - 1, -1, -1):
                    bufWord += word[letIndex]
                word += bufWord

            dataset.append(word)
            currentPolindromPeriond += 1

        return dataset
