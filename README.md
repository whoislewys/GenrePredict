# Instructions
1. [Download the training data](http://opihi.cs.uvic.ca/sound/genres.tar.gz)
2. Untar & unzip it to the data folder
3. Run command: `cd src && python train.py`

Remember to activate your conda environment with the other ML & audio packages needed,
or else train.py won't run

## Directory Structure
After decompressing the dataset, your directory structure should look like:

```
├── data/
   ├── genres
      ├── blues
      ├── classical
      ├── country
      .
      .
      .
      ├── rock
├── src/
```