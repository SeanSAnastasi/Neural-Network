import xlrd
from numpy import *
import random
import time

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

def getGradient(x):
    return(multiply(x,(1-x)))

def sampleData(number):
    arr = []
    for i in range(len(input_matrix)):
        arr.append(i)
    return random.sample(arr,number)

class NeuralNetwork:
    input_nodes, hidden_nodes, output_nodes,learning_rate = (None for i in range(4))
    weights_ih, weights_ho, bias_h, bias_o, hidden, inputs = ([] for i in range(6))
    def __init__(self, nInput, nHidden, nOutput):

        self.input_nodes = nInput
        self.hidden_nodes = nHidden
        self.output_nodes = nOutput
        self.weights_ih = self.RandomiseWeights(self.hidden_nodes, self.input_nodes , self.weights_ih )
        self.weights_ho = self.RandomiseWeights(self.output_nodes, self.hidden_nodes , self.weights_ho )
        self.bias_h = self.RandomiseWeights(self.hidden_nodes, 1 , self.bias_h )
        self.bias_o = self.RandomiseWeights(self.output_nodes,  1 , self.bias_o )
        self.learning_rate = 0.2
        
        
        

    def FeedForward(self, input, pretty):
        self.inputs = input
        self.hidden = self.weights_ih * self.inputs.T
        self.hidden = self.hidden + self.bias_h
        self.hidden = sigmoid(self.hidden)

        output = self.weights_ho * self.hidden
        output = output + self.bias_o
        output = sigmoid(output)
        
        if pretty:
            return(output.T.round())
        else:
            return output.T
    
    


    def RandomiseWeights(self,range1, range2, imatrix):
        
        for i in range(range1):
            temp = []
            for j in range(range2):
                weight = random.uniform(-1,1)
                temp.append(weight)
            imatrix.append(temp)
        
        return(matrix(imatrix))

    def train(self,inputs, targets):

        outputs = self.FeedForward(inputs, False)
        output_errors = targets - outputs
        

        output_gradient = getGradient(outputs)
        output_gradient = multiply(output_gradient, output_errors)
        output_gradient = multiply(output_gradient, self.learning_rate)

        weights_ho_deltas = output_gradient.T * self.hidden.T
        self.weights_ho = self.weights_ho + weights_ho_deltas
        self.bias_o = self.bias_o + output_gradient.T


        hidden_errors = self.weights_ho.T * output_errors.T
        hidden_gradient = getGradient(self.hidden)
        hidden_gradient = multiply(hidden_gradient, hidden_errors)
        hidden_gradient = multiply(hidden_gradient, self.learning_rate)

        weight_ih_deltas = hidden_gradient * self.inputs
        self.weights_ih = self.weights_ih + weight_ih_deltas
        self.bias_h = self.bias_h + hidden_gradient
        
        return

def main():
    start_time = time.time()
    start()
    nueral_net = NeuralNetwork(5,4,3)
    sample = sampleData(26)
    
    
    

    for i in range(1000):
        num = random.choice(sample)
        
        nueral_net.train(input_matrix[num], result_matrix[num])
        
        
    for i  in range(len(input_matrix)):
        
        
        print(str(input_matrix[i])+" - "+ str(nueral_net.FeedForward(input_matrix[i], True)))
    

    
    print("--- %s seconds ---" % (time.time() - start_time))

main()




