#if !defined(_TNN_DENSENET_H)
#define _TNN_DENSENET_H

#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>

#include "activations.h"
#include "../core/mersenneTwister.h"
#include "../core/timer.h"

typedef struct tnn_denseNet
{
	double lowerRange, upperRange;
	tnn_activation *activation;

	size_t num_layers;
	size_t *layer_sizes;
	size_t input_size;
	size_t output_size;

	double ***weights;
	double **bias_weights;

	double **output_values;
	double **output_derivative_values;
	double **output_error_values;
	double *input_layer;
	double *output_layer;

	tnn_mersenneTwister_64 rng;
} tnn_denseNet;

//////////////////////
//  The Algorithem  //
//////////////////////
void tnn_feedForward_fast_denseNet(tnn_denseNet *net)
{
	//double start = tnn_time();
	
	for(size_t layer = 1; layer < net->num_layers; ++layer)
		for(size_t neuron = 0; neuron < net->layer_sizes[layer]; ++neuron)
		{
			double sum = net->bias_weights[layer][neuron];
			for(size_t prevNeuron = 0; prevNeuron < net->layer_sizes[layer - 1]; ++prevNeuron)
				sum += net->output_values[layer - 1][prevNeuron] * net->weights[layer][neuron][prevNeuron];
			net->output_values[layer][neuron] = net->activation[layer].activation(sum);
		}
	//printf("%fs took tnn_feedForward_fast_denseNet\n", tnn_time() - start);
}
void tnn_feedForward_denseNet(tnn_denseNet *net)
{
	//double start = tnn_time();
	
	for(size_t layer = 1; layer < net->num_layers; ++layer)
		for(size_t neuron = 0; neuron < net->layer_sizes[layer]; ++neuron)
		{
			double sum = net->bias_weights[layer][neuron];
			for(size_t prevNeuron = 0; prevNeuron < net->layer_sizes[layer - 1]; ++prevNeuron)
				sum += net->output_values[layer - 1][prevNeuron] * net->weights[layer][neuron][prevNeuron];
			net->output_values[layer][neuron] = net->activation[layer].activation(sum);
			net->output_derivative_values[layer][neuron] = net->activation[layer].activation(net->output_values[layer][neuron]);
		}

	//printf("%fs took tnn_feedForward_denseNet\n", tnn_time() - start);
}
void tnn_backPropergate_denseNet(tnn_denseNet *net, double *target, double learningRate)
{
	//double start = tnn_time();

	for (size_t neuron = 0; neuron < net->layer_sizes[net->num_layers - 1]; ++neuron)
		net->output_error_values[net->num_layers  - 1][neuron] =
				(net->output_values[net->num_layers  - 1][neuron] - target[neuron]) 
				* net->output_derivative_values[net->num_layers  - 1][neuron];

	for (size_t layer = net->num_layers - 2; layer > 0; --layer)
		for (size_t neuron = 0; neuron < net->layer_sizes[layer]; ++neuron)
		{
			double sum = 0;
			for (size_t nextNeuron = 0; nextNeuron < net->layer_sizes[layer + 1]; ++nextNeuron)
				sum += net->weights[layer + 1][nextNeuron][neuron] * net->output_error_values[layer + 1][nextNeuron];
			net->output_error_values[layer][neuron] = sum * net->output_derivative_values[layer][neuron];
		}

	// Update weights
	for (size_t layer = 1; layer < net->num_layers; ++layer)
		for(size_t neuron = 0; neuron < net->layer_sizes[layer]; ++neuron)
		{
			double delta = -learningRate * net->output_error_values[layer][neuron];
			net->bias_weights[layer][neuron] += delta;
			for(size_t prevNeuron = 0; prevNeuron < net->layer_sizes[layer - 1]; ++prevNeuron)
				net->weights[layer][neuron][prevNeuron] += delta * net->output_values[layer - 1][prevNeuron];
		}
	//printf("%fs took tnn_backPropergate_denseNet\n", tnn_time() - start);
}

/////////////////////////
//  Import and export  //
/////////////////////////
tnn_denseNet * tnn_import_denseNet(tnn_denseNet *net, const char *filename)
{
	FILE *f = fopen(filename, "rb");
	if (!f)
	{
		printf("Unable to open file(\"%s\")!", filename);
		return NULL;
	}

	fread(&net->num_layers, sizeof(size_t), 1, f);
	fread(net->layer_sizes, sizeof(size_t), net->num_layers, f);

	for (size_t i = 1; i < net->num_layers; ++i)
	{
		fread(net->bias_weights[i], sizeof(double), net->layer_sizes[i], f);
		for (size_t j = 0; j < net->layer_sizes[i]; ++j)
			fread(net->weights[i][j], sizeof(double), net->layer_sizes[i - 1], f);
	}

	fclose(f);
	return net;
}
void tnn_export_denseNet(tnn_denseNet *net, const char *filename)
{
	FILE *f = fopen(filename, "wb");
	if (!f)
	{
		printf("Unable to open file(\"%s\")!", filename);
		return;
	}

	fwrite(&net->num_layers, sizeof(size_t), 1, f);
	fwrite(net->layer_sizes, sizeof(size_t), net->num_layers, f);

	for (size_t i = 1; i < net->num_layers; ++i)
	{
		fwrite(net->bias_weights[i], sizeof(double), net->layer_sizes[i], f);
		for (size_t j = 0; j < net->layer_sizes[i]; ++j)
			fwrite(net->weights[i][j], sizeof(double), net->layer_sizes[i - 1], f);
	}

	fclose(f);
}

////////////////////////////////
//  Creation and destruction  //
////////////////////////////////
void tnn_reset_denseNet(tnn_denseNet *net)
{
	for(size_t layer = 1; layer < net->num_layers; ++layer)
		for(size_t neuron = 0; neuron < net->layer_sizes[layer]; ++neuron)
		{
			net->bias_weights[layer][neuron] = (((double)tnn_mt64_next(&net->rng) / (double)UINT64_MAX * (abs(net->lowerRange) + net->upperRange)) + net->lowerRange);
			
			for(size_t prevNeuron = 0; prevNeuron < net->layer_sizes[layer - 1]; ++prevNeuron)
				net->weights[layer][neuron][prevNeuron] =
						(((double)tnn_mt64_next(&net->rng) / (double)UINT64_MAX 
						* (abs(net->lowerRange) + net->upperRange))
						+ net->lowerRange);
		}
}
tnn_denseNet *tnn_create_denseNet(double upper, double lower, size_t seed, size_t num_layers, ...)
{
	va_list valist;
   
	tnn_denseNet *net = (tnn_denseNet*)malloc(sizeof(tnn_denseNet));
	net->lowerRange = upper;
	net->upperRange = lower,
	net->activation = (tnn_activation*)malloc(num_layers * sizeof(tnn_activation));
	net->num_layers = num_layers;
	net->layer_sizes = (size_t*)malloc(num_layers * sizeof(size_t));

	va_start(valist, num_layers);
	for (size_t i = 0; i < num_layers; ++i)
	{
		net->layer_sizes[i] = va_arg(valist, size_t);
		net->activation[i] = va_arg(valist, tnn_activation);
   	}
	va_end(valist);

	net->input_size = net->layer_sizes[0];
	net->output_size = net->layer_sizes[net->num_layers - 1];

	net->bias_weights = (double**)malloc(num_layers * sizeof(double*));
	net->weights = (double***)malloc(num_layers * sizeof(double**));
	net->output_values = (double**)malloc(num_layers * sizeof(double*));
	net->output_derivative_values = (double**)malloc(num_layers * sizeof(double*));
	net->output_error_values = (double**)malloc(num_layers * sizeof(double*));

	for (size_t i = 0; i < num_layers; ++i)
	{
		net->output_values[i] = (double*)malloc(net->layer_sizes[i] * sizeof(double));
		net->output_derivative_values[i] = (double*)malloc(net->layer_sizes[i] * sizeof(double));
		net->output_error_values[i] = (double*)malloc(net->layer_sizes[i] * sizeof(double));
		if(i > 0)
		{
			net->bias_weights[i] = (double*)malloc(net->layer_sizes[i] * sizeof(double));
			net->weights[i] = (double**)malloc(net->layer_sizes[i] * sizeof(double*));
			for (size_t j = 0; j < net->layer_sizes[i]; ++j)
				net->weights[i][j] = (double*)malloc(net->layer_sizes[i - 1] * sizeof(double));
		}
	}
	
	net->input_layer = net->output_values[0];
	net->output_layer = net->output_values[num_layers - 1];

	tnn_mt64_seed(&net->rng, seed);
	tnn_reset_denseNet(net);

	return net;
}
void tnn_free_denseNet(tnn_denseNet *net)
{
	net->input_layer = NULL;
	net->output_layer = NULL;

	for (size_t i = 1; i < net->num_layers; ++i)
	{
		free(net->output_values[i]);
		free(net->output_derivative_values[i]);
		free(net->output_error_values[i]);
		if(i > 0)
		{
			for (size_t j = 0; j < net->layer_sizes[i]; ++j)
				free(net->weights[i][j]);
			free(net->weights[i]);
			free(net->bias_weights[i]);
		}
	}

	free(net->output_values);
	free(net->output_derivative_values);
	free(net->output_error_values);

	free(net->weights);
	free(net->bias_weights);

	free(net->activation);
	free(net->layer_sizes);

	free(net);
}

#endif // _TNN_DENSENET_H