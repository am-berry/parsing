### Guide to this repository

- `getters` directory contains code for `wget` all posts from pushshift.io

- `src` directory contains Rust code for getting all tldr pairs -> you need to have cargo to compile this code 

- `preprocessing` directory contains all code for getting from the output of Rust code to the `csv` file (`prp.csv`) used to develop the train and test sets. Warning: these all need a lot of memory.

- `data` directory contains the train, test data along with the larger processed dataset they came from. The intermediate files produced in the above were removed for the sake of memory. 

- All result .csv files are placed in the `results` directory

- The semantic text similarity model training and test scripts, along with the data are in the `sts` directory

- The location of each experiment is self explanatory.

- For the t5 experiments, we only include the final trained model as including all checkpoint/validated models (16 total) would be over 16 GB in size. All models available upon request.

- Each Python script contains commented code explaining the processes

- The PyTorch-related scripts can be somewhat difficult to get running on different hardware. For reference, the version in `requirements.txt` was compiled from source to run on an RTX 3080 GPU with CUDA 11.0. 

- For all experiments, the ROUGE results are placed in a rouge_results/ directory 
