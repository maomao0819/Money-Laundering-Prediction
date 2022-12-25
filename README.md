# Money-Laundering-Prediction

Competition: https://tbrain.trendmicro.com.tw/Competitions/Details/24
data: https://tbrain.trendmicro.com.tw/Competitions/Details/24

## Usage
* download dataset
```
bash get_data.sh
```
* pipelines
    * TA.py
    * base.py (main) + utils_base.py + dataset_base.py + model.py 
        * modify from TA
        * one alert key one data (collect records with the nearest given date with the same alert key)
        * manually enable or disable the model in base.py 
    * main.py (main) + parser.py + utils.py + engine.py + dataset.py + model.py 
        * one alert key multiple data (collect records with the near given date with the same alert key)
        * more config in parser.py
        * manually enable or disable the ML model in engine.py predict()
    
* main usage
```
bash get_data.sh
python main.py # or base.py
```