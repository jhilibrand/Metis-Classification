# Classification Project Proposal

Question: 
Can we use classification methods to create an better sector/industry map of US equities? Is there a systematic way to generate a companies closest peer, other than observing market price correlation? 

Data Description: 
I will limit my sample to all the constituents in the Russell 3000 (the largest 3000 companies in the US in terms of market capitalization). Using this company list, my features will be pulled from the Yahoo finance library. These features will at a minimum include indicators such as dividend rate, profit margins, market cap, price to book, trailing EPS, implied volatility, etc. 
Ideally this analysis may serve two functions: 
1. anomally detection: why are certain companies being misclassified on fundamentals? is that a useful valuation metric? 
2. peer generation: can we have a robust methodology for generating closest peers on a fundamental level, not market price correlation? 

Tools/MVP: 
The majority of my analysis will be done using pandas, matplotlib, sklearn, and a few other tools. An MVP for this project would be a baseline model that has been able to make categorical predictions for a subset of my companies, having been trained on the remainder.  
