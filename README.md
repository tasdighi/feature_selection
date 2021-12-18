# feature_selection
### Label perspective 
(whether label information is involved during the selection phase
1. supervised
2. Unsupervised
3. Semi-supervised
### Selection strategy perspective
1. Wrapper methods
   - rely on predictive performance of learning algorithm and repeat shrink/grow feature until some stopping criteria satisfied: achieve high accuracy
   - Selection algorithm
     - Forward
     - Backward
   - Search strategy
     - Sequential search
     - Best-first search
     - Branch and bound search
2. Filter methods
   - independent of any learning algorithm, rely on certain characteristics of data to 
   - single feature evaluation: measure quality of feature by all metrics
     - frequency based
     - features and labels based. (exp: mutual information and chi square statistic
       - fisher:  construct the lower-dimensional space which maximizes the between-class variance and minimizes the within-class variance.
       - chi
       - relief
     - information theory. exp: KL-Divergence, information gain
     - Gini indexing
3. Embedded methods
   - Filter methods (efficiency) + wrapper methods (Interaction with learning algorithm)
