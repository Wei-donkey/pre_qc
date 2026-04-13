# Scripts for Paper hourly precipitation quality control


## Description
This project was rebuilt on April 13 2026, incorporating git and GitHub.

The main purpose of this project is to prepare the datasets and have them checked 
using our novel method and methods proposed by previous researchers.

Since the original hourly data are stored in Oracle 19c, connection to Oracle is required.
Using sqlAlchemy's thin mode, Oracle database can be connected without Oracle client.
After the quality inspection, all referenced data used to check the quality of target data
will be saved to a local directory.
The intermediate results, final tables, and figures will be saved locally as well.

## Project Structure
__init__.py: 
'src/': Contains Python processing scripts.
'data/': Raw, processed, and external data (ignored by Git).
'figures/': Output plots (ignored by Git).

## Usage
Scripts starting with ora_ connect Oracle database and cannot run out of the domain network;
Scripts starting with stat_ deal with the output file saved in local directories;
Scripts starting with plot_ produce figures based on the processed data.
rhre_recog_assess.py is used to save functions for recognizing and assessing RHRE strength.