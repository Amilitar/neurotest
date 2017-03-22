


class TikTak(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def printMatrix(self):
        iterator = 0
        for bit in self.matrix:
            if iterator == 3:
                print ('\n')

            print (bit)
            iterator += 1

    def matrixSurround(self):
        print ('_' * 50)

