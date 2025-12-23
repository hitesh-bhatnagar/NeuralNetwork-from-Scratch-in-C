#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4
#define  numberOfEpochs  10000
// simple NN that can learn XOR
double init_weights() {
	return ((double)rand())/((double)RAND_MAX);
}

double sigmoid(double x){
	return 1.0 / (1.0 + exp(-x));
}

double derivative(double x){
	return x*(1-x);
}

void shuffle(int *array, size_t n){
	if(n>1) {
		size_t i;
		for(i = 0; i < n; i++){
			size_t j = i + rand() / (RAND_MAX / (n-i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t; 
		}
	}
}



int main(void){

	srand(time(NULL));		// seed so results change every run
	const double lr = 0.1f;

	double hiddenLayer[numHiddenNodes];
	double outputLayer[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};

	double training_outputs[numTrainingSets][numOutputs] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

	// Initialize Hidden weights
	for (int i = 0; i < numInputs; i++){
		for(int j = 0; j < numHiddenNodes; j++){
			hiddenWeights[i][j] = init_weights();
		}
	}	

	// initialize output weights 
	for (int i = 0; i < numHiddenNodes; i++){
			for(int j = 0; j < numOutputs; j++){
				outputWeights[i][j] = init_weights();
		}
	}	

	// initialize biases 
	for(int i = 0; i < numOutputs; i++){
		outputLayerBias[i] = init_weights();
		
	}

	for(int i = 0; i < numHiddenNodes; i++) hiddenLayerBias[i] = init_weights();

	int trainingSetOrder[] = {0,1,2,3};



	// training the NN for 1000 epochs

	for(int epoch = 0; epoch < numberOfEpochs; epoch++){
		shuffle(trainingSetOrder, numTrainingSets);

		for(int i = 0; i < numTrainingSets; i++){
			int x = trainingSetOrder[i];

			// forward pass

			// compute hidden layer activation
			for(int j = 0; j < numHiddenNodes; j++){
				double activation = hiddenLayerBias[j];

				for(int k = 0; k < numInputs; k++){
					activation += training_inputs[x][k] * hiddenWeights[k][j];
					
				}

				hiddenLayer[j] = sigmoid(activation);
			}

			// computing output layer activation
			for(int j = 0; j < numOutputs; j++){
				double activation = outputLayerBias[j];
				for(int k = 0; k < numHiddenNodes; k++){
					activation += hiddenLayer[k] * outputWeights[k][j];
				}

				outputLayer[j] =sigmoid(activation);
			}

			if (epoch % 100 == 0 && i == 0) {
                 printf("Epoch: %d | Input: %g %g | Output: %g | Target: %g\n", 
                        epoch, training_inputs[x][0], training_inputs[x][1], 
                        outputLayer[0], training_outputs[x][0]);
            }

			// BackPropagation
			// compute change in output weights

			double deltaOutput[numOutputs];

			for(int j = 0; j < numOutputs; j++){
				double error = (training_outputs[x][j] - outputLayer[j]);
				deltaOutput[j] = error * derivative(outputLayer[j]);
			}

			// compute change in hidden weights
			double deltaHidden[numHiddenNodes];

			for(int j = 0; j < numHiddenNodes; j++){
				double error = 0.0f;

				for(int k = 0; k < numOutputs; k++){
					error += deltaOutput[k] * outputWeights[j][k];
					
				}

				deltaHidden[j] = error* derivative(hiddenLayer[j]);
			}

			// apply change in output weights

			for(int j = 0; j < numOutputs; j++){
				outputLayerBias[j] += deltaOutput[j] * lr;

				for(int k = 0; k < numHiddenNodes; k++){
					outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
				}
			}

			//  apply change in hidden weights
			for(int j = 0; j < numHiddenNodes; j++){
				hiddenLayerBias[j] += deltaHidden[j] * lr;

				for(int k = 0; k < numInputs; k++){
					hiddenWeights[k][j] += training_inputs[x][k] * deltaHidden[j] * lr;
				}
			}
		}
	}

	printf("\nFinal Results after %d epochs : \n", numberOfEpochs);

	for(int i = 0;  i < numTrainingSets; i++){
		// forward passs for display

		for(int j = 0; j < numHiddenNodes; j++){
			double activation = hiddenLayerBias[j];

			for(int k = 0; k < numInputs; k++){
				activation += training_inputs[i][k] * hiddenWeights[k][j];
			}

			hiddenLayer[j] = sigmoid(activation);
		}

		for(int j = 0; j < numOutputs; j++){
			double activation = outputLayerBias[j];
			for(int k = 0; k < numHiddenNodes; k++){

				activation += hiddenLayer[k] * outputWeights[k][j];
			}

			outputLayer[j] = sigmoid(activation);
		}

		printf("Input: %g  %g  ->  Output:  %g  (Target:  %g)\n", training_inputs[i][0], training_inputs[i][1], outputLayer[0], training_outputs[i][0]);
	}


	return 0;

}
