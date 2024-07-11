# Usage
1. Install Python 3.7. For convenience, 
execute the following command.

   `pip install -r requirements.txt`

2. Generate Statistical Results. 
Run `Fig1.py` to get the national patterns and zonal differences 
in HABs response to nutrient reduction. 
Run `Fig2.py` to get how climate change promoted HABs 
and altered the expected effect of nutrient reduction. 
The Data used in Statistics are already in the project.

3. Generate Modeling Results. 
The Data used in Modeling are provided in [Google Drive](https://drive.google.com/drive/folders/1zNG8akvqXo5uwaStmMz1kIhmmkDEpfAE?usp=sharing). 
Download and place the model input data in the folder `./Data`. Run `ModelParasSensitiveAanlysis.py` to get the model paras sensitive results. 
Run `ModelOptimization.py` to get the optimized paras values and the calibrated model results. 
Run `ScenarioAnalysis.py` to get the model scenario results.