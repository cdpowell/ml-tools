This zip file contains 6 datasets. 

synthetic-*.csv file description:
First two columns are the feature values.
The final column is the class label.

pokemonStats.csv and pokemonLegendary.csv file description:
pokemonStats.csv and pokemonLegendary.csv contain statistics about various Pokemon (although their names have not been included). pokemonStats.csv contains feature information about each pokemon. This information includes stat information (such as values for special attack, attack, defense, etc.), typing information, and information on which generation the Pokemon is from. pokemonLegendary.csv only contains one column which indicates whether the associated pokemon is legendary or not. This dataset is meant to be paired with pokemonStats.csv. Thus, row 1 in pokemonStats.csv is tightly coupled with row 1 in pokemonLegendary.csv. They are meant to be used together to create the full dataset. Also to note, these datasets have been artificially oversampled. It turns out there aren't that many legendary Pokemon out there. To make this a more manageable problem, I upsampled the minority class using the SMOTE algorithm.
