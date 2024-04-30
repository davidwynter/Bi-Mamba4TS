# Bi-Mamba4TS
A python implementation of the 2404.15772v1.pdf paper on arxiv.org

1st draft, untested

Based on the paper, there are 4 components:

- SRA Decider: Decides the tokenization strategy based on Pearson correlation coefficients.
- Patch Tokenization: Converts time series into patch-wise tokens.
- Bi-Mamba Encoder: A bidirectional encoder for handling the time series data.
- Loss Function: Typically, MSE (Mean Squared Error) for regression tasks.

## SRA Decider

The SRA Decider in the Bi-Mamba4TS model is designed to choose between channel-independent and channel-mixing 
tokenization strategies based on the Pearson correlation coefficients among different series. The decision is 
based on a threshold λλ which you set (defaulting to 0.6 in the skeleton).

### SRA Decider Logic:

The SRA_Decider module should:

Calculate the Pearson correlation coefficients between each pair of series.
Use a threshold λλ to determine the degree of correlation that justifies switching from a
channel-independent strategy to a channel-mixing strategy.

#### Explanation:

Normalization: For each series in the batch, the data is normalized by subtracting the mean and 
dividing by the standard deviation.
Correlation Calculation: The Pearson correlation coefficients are calculated using the formula 
Correlation(X,Y)=∑(X−X‾)(Y−Y‾)∑(X−X‾)2∑(Y−Y‾)2Correlation(X,Y)=∑(X−X)2∑(Y−Y)2

​∑(X−X)(Y−Y)​, which simplifies to a matrix multiplication of normalized series when each series is normalized.
Decision Making: The decision to use channel-mixing or channel-independent tokenization is based on whether 
the maximum correlation coefficient in the matrix (excluding self-correlations) exceeds the threshold λλ.

#### Integration:

This function should be integrated into your training loop where you pass the current batch of your multivariate 
time series data through this decider to choose the appropriate tokenization strategy dynamically based on the 
data's inter-series relationships. Adjustments may be needed depending on the exact shape and nature of your 
data inputs.

## PatchTokenizer

The main task is to convert a sequence of multivariate time series data into patches. 
This transformation allows the model to focus on local sub-sequences or "patches" of the data.
This can be critical for capturing local temporal patterns more effectively.
Patch Tokenization:

Objective: Divide each univariate series into non-overlapping patches.
Input Shape: Typically, the input to the PatchTokenizer would be of shape [batch_size, num_series, sequence_length].
Output Shape: After patch tokenization, the output should be [batch_size, num_series, num_patches, patch_size], 
where num_patches is the number of patches that can be created from sequence_length.

### Explanation:

Input Shape and Patch Calculation: The function starts by extracting the dimensions of the input tensor and 
then computes how many full patches can be extracted from each time series.
Validation: It checks if the sequence_length is perfectly divisible by patch_size. If not, it raises a ValueError. 
This is crucial to ensure that each patch has the same size, which is important for consistent processing by 
subsequent model components.
Reshaping for Patches: The tensor is reshaped to group elements into patches. The view method is used to reshape 
the tensor without copying the data, but it requires that the number of elements remains constant.

### Integration and Usage:

Integrate this component in your model's forward function where it will preprocess the multivariate time series data 
before passing it to the encoder or other components. Make sure that your data dimensions are correctly managed, and 
the sequence_length is indeed divisible by patch_size for every batch of data.

This implementation provides a foundational structure for the PatchTokenizer component. Depending on your specific 
requirements and data characteristics, further customization might be necessary, especially concerning how to 
handle edge cases where the sequence length is not a perfect multiple of the patch size.

# TODO 
Initialize the model and prepare the dataset.
Implement the training loop using the loss function (MSE) and an optimizer (like Adam).
Evaluate the model on your validation/test dataset.
