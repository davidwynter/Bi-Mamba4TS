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
}
