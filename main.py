extern crate ndarray;
extern crate ndarray_stats;

use ndarray::{Array2, ArrayView1};
use ndarray_stats::CorrelationExt;

struct SRA_Decider {
    threshold: f64, // λ
}

impl SRA_Decider {
    fn new(threshold: f64) -> Self {
        SRA_Decider { threshold }
    }

    /// Calculate correlations and decide the tokenization strategy based on counts and their ratio.
    /// 
    /// # Arguments
    /// * `x` - A 2D array where each row is a time series and each column is a time point.
    ///
    /// # Returns
    /// * Returns 1 for channel-mixing strategy, or 0 for channel-independent strategy.
    fn forward(&self, x: &Array2<f64>) -> usize {
        let num_series = x.nrows();
        let mut count_above_threshold = vec![0; num_series];
        let mut count_positive = vec![0; num_series];

        // Iterate over pairs of series to compute the correlation coefficients
        for i in 0..num_series {
            for j in i + 1..num_series {
                let series_i = x.row(i);
                let series_j = x.row(j);
                if let Ok(correlation) = series_i.pearson_correlation(&series_j) {
                    if correlation >= self.threshold {
                        count_above_threshold[i] += 1;
                        count_above_threshold[j] += 1;
                    }
                    if correlation > 0.0 {
                        count_positive[i] += 1;
                        count_positive[j] += 1;
                    }
                }
            }
        }

        let max_above_threshold = count_above_threshold.iter().max().copied().unwrap_or(0);
        let max_positive = count_positive.iter().max().copied().unwrap_or(0);

        // Calculate the relation ratio
        let ratio = if max_positive > 0 {
            max_above_threshold as f64 / max_positive as f64
        } else {
            0.0
        };

        // Decide the strategy based on the ratio and threshold
        if ratio >= 1.0 - self.threshold {
            1 // Use channel-mixing strategy
        } else {
            0 // Use channel-independent strategy
        }
    }

class TokenizationStrategy:
    CHANNEL_INDEPENDENT = 'channel_independent'
    CHANNEL_MIXING = 'channel_mixing'


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x, strategy=TokenizationStrategy.CHANNEL_INDEPENDENT):
        batch_size, num_series, sequence_length = x.shape
        if sequence_length % self.patch_size != 0:
            raise ValueError("sequence_length must be divisible by patch_size")

        num_patches = sequence_length // self.patch_size
        x_patched = x.view(batch_size, num_series, num_patches, self.patch_size)

        if strategy == TokenizationStrategy.CHANNEL_INDEPENDENT:
            # Process each channel individually
            return x_patched
        elif strategy == TokenizationStrategy.CHANNEL_MIXING:
            # Group patches with the same index of different series
            return x_patched.permute(0, 2, 1, 3)
        return x_patched


class MambaBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.conv1d = nn.Conv1d(out_features, out_features, kernel_size=3)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.conv1d(x))
        return self.linear2(x)


class BiMambaEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.forward_block = MambaBlock(in_features, out_features)
        self.backward_block = MambaBlock(in_features, out_features)

    def forward(self, x):
        forward_output = self.forward_block(x)
        backward_input = torch.flip(x, dims=[1])  # Flip on time dimension
        backward_output = self.backward_block(backward_input)
        return forward_output + backward_output


class BiMamba4TS(nn.Module):
    def __init__(self):
        super().__init__()
        self.decider = SRA_Decider(num_series=10)  # Example value
        self.tokenizer = PatchTokenizer(patch_size=24)  # Example value
        self.encoder = BiMambaEncoder(in_features=128, out_features=256)  # Example sizes
        self.regressor = nn.Linear(256, 1)  # Predicting one value at a time

    def forward(self, x, correlations):
        # Determine the strategy from the decider's output
        strategy_indices = self.decider(correlations)
        strategies = [TokenizationStrategy.CHANNEL_MIXING if idx else TokenizationStrategy.CHANNEL_INDEPENDENT for idx in strategy_indices]

        outputs = []
        for i, strategy in enumerate(strategies):
            # Apply the tokenization strategy determined by the SRA_Decider
            tokenized_patches = self.tokenizer(x[i], strategy=strategy)
            # Encode the tokenized patches
            encoded_output = self.encoder(tokenized_patches)
            # Apply the regressor to get the final output
            final_output = self.regressor(encoded_output)
            outputs.append(final_output)

        return torch.stack(outputs)


def main():
    # Assume each series has a length of 240 (divisible by 24 for patching) and we have 10 series
    # Create synthetic batch data: 5 batches, 10 series per batch, 240 length per series
    batch_size = 5
    num_series = 10
    sequence_length = 240
    input_data = torch.randn(batch_size, num_series, sequence_length)

    # Create synthetic correlation data: 5 batches, 10 series (just for demonstration, normally it would be precomputed)
    correlations = torch.randn(batch_size, num_series, num_series)

    # Initialize the model
    model = BiMamba4TS()

    # Run the model
    output = model(input_data, correlations)

    # Print the output
    print("Output from BiMamba4TS Model:", output)


if __name__ == "__main__":
    main()
}
