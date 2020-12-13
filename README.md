0. Most of the code is from the open sourced implementation of AWD-LSTM and MoS. 
1. base_models contain the bottom layers of the AWD-LSTM network that are common 
to all models under comparison.
2. the differences for Softmax, SS, GSS, LMS-PLIF, MoS, and MoC models are 
grouped accordingly in top_models.
3. main.py is the file to lookout for model training and evaluation. 
4. analysis.py has the code for most of the analysis that we had presented in our 
paper.
5. To make the code work, replace the strings "<your_X>" to the right values.
X could be any substring. Search them accordingly. We replaced some of the strings
with such values to preserve anonymity. 

