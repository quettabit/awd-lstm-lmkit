## pdfs directory

It contains the pdfs of the paper, supplementary material, reviews, author response, and meta review. 

## Note
-  Most of the code is from the open sourced implementation of AWD-LSTM and MoS.
- `base_models` contain the bottom layers of the AWD-LSTM network that are common 
to all models under comparison.
- the differences for Softmax, SS, GSS, LMS-PLIF, MoS, and MoC models are 
grouped accordingly in `top_models`.
- `main.py` is the file to lookout for model training and evaluation. 
- `analysis.py` has the code for most of the analysis that we had presented in our 
paper.
- To make the code work, replace the strings "<your_X>" to the right values.
X could be any substring. Search them accordingly. We replaced some of the strings
with such values to preserve anonymity. 

## More information

If you are curious or still looking for more information, please look at my [master's thesis](https://ruor.uottawa.ca/handle/10393/41412).

## TODO

- Add all the required attributions
- Docs about the code
- Link the paper

