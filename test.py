import xlrd
from numpy import *
import random

input_matrix, result_matrix = (None for i in range(2))


def start():
    global input_matrix, result_matrix
    workbook = xlrd.open_workbook("dataset.xlsx")
    worksheet = workbook.sheet_by_index(0)

    input_arr = []
    result_arr = []

    for i in range(worksheet.nrows):
        temp_input = []
        temp_result = []
        for j in range(5):
            temp_input.append(int(worksheet.cell_value(i,j)))
        for j in range(6,9):
            temp_result.append(int(worksheet.cell_value(i,j)))
            
        input_arr.append(temp_input)
        result_arr.append(temp_result)
    input_matrix = matrix(input_arr)
    result_matrix = matrix(result_arr)

    
    
def sigmoid(x):
    return (1/(1+exp(-x)))

class NeuralNetwork:
    input_nodes, hidden_nodes, output_nodes, = (None for i in range(3))
    weights_ih, weights_ho, bias_h, bias_o = ([] for i in range(4))
    def __init__(self, nInput, nHidden, nOutput):

        self.input_nodes = nInput
        self.hidden_nodes = nHidden
        self.output_nodes = nOutput
        self.weights_ih = self.RandomiseWeights(self.hidden_nodes, self.input_nodes , self.weights_ih )
        self.weights_ho = self.RandomiseWeights(self.output_nodes, self.hidden_nodes , self.weights_ho )
        self.bias_h = self.RandomiseWeights(self.hidden_nodes, 1 , self.bias_h )
        self.bias_o = self.RandomiseWeights(self.output_nodes,  1 , self.bias_o )

        
        
        

    def FeedForward(self, input):
        
        hidden = self.weights_ih * input
        hidden = hidden + self.bias_h
        hidden = sigmoid(hidden)

        output = self.weights_ho * hidden
        output = output + self.bias_o
        output = sigmoid(output)

        return output
    
    


    def RandomiseWeights(self,range1, range2, imatrix):
        
        for i in range(range1):
            temp = []
            for j in range(range2):
                weight = random.uniform(-1,1)
                temp.append(weight)
            imatrix.append(temp)
        
        return(matrix(imatrix))

    def train(self,inputs, answer):
        return

def main():
    start()
    nueral_net = NeuralNetwork(5,4,3)
    
    output = nueral_net.FeedForward(input_matrix[0].T)

    print(output)

    


main()




