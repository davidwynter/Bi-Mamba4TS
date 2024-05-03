import torch
import torch.nn as nn
import torch.nn.functional as F


class SRA_Decider(nn.Module):
    def __init__(self, num_series, threshold=0.6):
        super().__init__()
        self.num_series = num_series
        self.threshold = threshold

    def forward(self, x):
        batch_size, num_series, sequence_length = x.shape
        results = []
        for i in range(batch_size):
            series_batch = x[i]
            mean = torch.mean(series_batch, dim=1, keepdim=True)
            std = torch.std(series_batch, dim=1, keepdim=True)
            norm_series = (series_batch - mean) / std
            correlation_matrix = torch.mm(norm_series, norm_series.t()) / sequence_length
            correlation_matrix.fill_diagonal_(0)
            
            # Count correlations above threshold and positive correlations
            count_above_threshold = (correlation_matrix > self.threshold).sum().item()
            count_positive = (correlation_matrix > 0).sum().item()

            # Calculate ratio and decide the strategy
            ratio = count_above_threshold / count_positive if count_positive != 0 else 0
            strategy = 1 if ratio >= 1 - self.threshold else 0
            results.append(strategy)
        
        return torch.tensor(results, dtype=torch.int64, device=x.device)


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


