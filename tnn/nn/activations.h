#if !defined(_TNN_ACTIVATIONS_H)
#define _TNN_ACTIVATIONS_H

#include <math.h>


double tnn_activation_func_sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}
double tnn_activation_func_sigmoid_derivative(double x) {
	const double temp = tnn_activation_func_sigmoid(x);
	return temp * (1 - temp);
}

double tnn_activation_func_tanh(double x) {
	return tanh(x);
}
double tnn_activation_func_tanh_derivative(double x) {
	return 1.0 - pow(tnn_activation_func_tanh(x), 2);
}

double tnn_activation_func_linear(double x) {
	return x;
}
double tnn_activation_func_linear_derivative(double x) {
	return 1.0;
}

double tnn_activation_func_relu(double x) {
	if (x > 0) { return x; }
	return 0;
}
double tnn_activation_func_relu_derivative(double x) {
	if (x > 0) { return 1; }
	return 0; // no learning
}

double tnn_activation_func_leakyRelu(double x) {
	if (x > 0) { return x; }
	return 0.05 * x;

}
double tnn_activation_func_leakyRelu_derivative(double x) {
	if (x > 0) { return 1; }
	return 0.05;
}

typedef struct tnn_activation
{
	double (*activation)(double);
	double (*derivative)(double);
} tnn_activation;

const tnn_activation tnn_activation_sigmoid = {
	tnn_activation_func_sigmoid, 
	tnn_activation_func_sigmoid_derivative
};
const tnn_activation tnn_activation_tanh = {
	tnn_activation_func_tanh, 
	tnn_activation_func_tanh_derivative
};
const tnn_activation tnn_activation_linear = {
	tnn_activation_func_linear,
	tnn_activation_func_linear_derivative
};
const tnn_activation tnn_activation_relu = {
	tnn_activation_func_relu,
	tnn_activation_func_relu_derivative
};
const tnn_activation tnn_activation_leakyRelu = {
	tnn_activation_func_leakyRelu,
	tnn_activation_func_leakyRelu_derivative
};


#endif // _TNN_ACTIVATIONS_H