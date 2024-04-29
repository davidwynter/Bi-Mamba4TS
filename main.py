import torch
import torch.nn as nn
import torch.nn.functional as F

class SRA_Decider(nn.Module):
    def __init__(self, num_series, threshold=0.6):
        super().__init__()
        self.num_series = num_series
        self.threshold = threshold

    def forward(self, x):
        # x is a batch of multivariate time series data with shape [batch_size, num_series, sequence_length]
        batch_size, num_series, sequence_length = x.shape

        # Calculate Pearson correlation coefficients matrix for each batch
        results = []
        for i in range(batch_size):
            series_batch = x[i]  # Shape [num_series, sequence_length]
            mean = torch.mean(series_batch, dim=1, keepdim=True)
            std = torch.std(series_batch, dim=1, keepdim=True)
            norm_series = (series_batch - mean) / std  # Normalize
            correlation_matrix = torch.mm(norm_series, norm_series.t()) / sequence_length
            correlation_matrix.fill_diagonal_(0)  # Remove self-correlations

            # Determine the strategy based on the maximum correlation coefficient
            max_corr = torch.max(correlation_matrix)
            if max_corr >= self.threshold:
                results.append(1)  # Channel-mixing strategy
            else:
                results.append(0)  # Channel-independent strategy
        
        return torch.tensor(results, dtype=torch.int64, device=x.device)

class PatchTokenizer(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        # x is a batch of multivariate time series data with shape [batch_size, num_series, sequence_length]
        batch_size, num_series, sequence_length = x.shape

        # Calculate the number of patches that can be created
        num_patches = sequence_length // self.patch_size

        # Ensure the sequence length is divisible by the patch size
        if sequence_length % self.patch_size != 0:
            raise ValueError("sequence_length must be divisible by patch_size")

        # Reshape the input to create patches
        # New shape: [batch_size, num_series, num_patches, patch_size]
        x_patched = x.view(batch_size, num_series, num_patches, self.patch_size)

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
        strategy = self.decider(correlations)
        x = self.tokenizer(x)
        x = self.encoder(x)
        return self.regressor(x)

model = BiMamba4TS()
