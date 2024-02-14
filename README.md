# Code of "Know in AdVance: Linear-Complexity Forecasting of Ad Campaign Performance with Evolving User Interest"

AdVance is a framework for advertisers to estimate an ad campaign's expected cost and yield with specific criteria. The framework contains three modules: 1) the local interest module adds time-stamp positional embedding to user historical clicks and compresses all history behaviors into fatigue vectors. 2) The auction representation module applies self- and cross-attention to candidate ads, historical clicks, user features, and context. It adopts a supervised paradigm to learn a representation for each auction effectively. Specifically, it introduces two sub-models: one for predicting the winning probability for each candidate and one predicts the user's pCTR/pCVR towards each candidate. 3) the global campaign module takes the input of a series of auction representations and predicts the campaign performance based on the summary vector and the accumulated results from every auction. 

Due to user privacy and company business interests concerns, we are forbidden to release the bid records. In the code, we focus on the main structure of AdVance, which aims to facilitate the reader's understanding of the proposed framework. 