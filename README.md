## On the Softmax Bottleneck of Recurrent Language Models

This repository contains the code, data, and supplementary material for our paper which got accepted at the main track of AAAI 2021. 

## Some pointers
-  Most of the code is from the open sourced implementation of AWD-LSTM and MoS. 
- base_models contain the bottom layers of the AWD-LSTM network that are common 
to all models under comparison.
- the differences for Softmax, SS, GSS, LMS-PLIF, MoS, and MoC models are 
grouped accordingly in top_models.
- main.py is the file to lookout for model training and evaluation. 
- analysis.py has the code for most of the analysis that we had presented in our 
paper.
- To make the code work, replace the strings "<your_X>" to the right values.
X could be any substring. Search them accordingly. We replaced some of the strings
with such values to preserve anonymity. 

