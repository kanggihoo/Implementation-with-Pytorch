#!/bin/bash
# Download Cornell Movie Dialogs Corpus using curl
curl  -o corpus.zip  http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip


# Unzip the downloaded file
unzip -qq corpus.zip -d corpus

# Remove the downloaded zip file
rm corpus.zip

# # # Create a directory for datasets if not exists
mkdir -p datasets

# # # Move necessary files to the datasets directory
mv corpus/cornell\ movie-dialogs\ corpus/movie_conversations.txt ./datasets
mv corpus/cornell\ movie-dialogs\ corpus/movie_lines.txt ./datasets
rm -rf corpus


echo "Download and setup completed."