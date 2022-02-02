# NormalMapsGenerator
Python scripts generating normal map(s) for texture(s)

## Example results

![Иллюстрация к проекту](https://github.com/mrglaster/NormalMapsGenerator/blob/master/example_result.png)

## Usage

### Step 1: Install required libraries

```
pip install requirements.txt
```

or 


```
pip install numpy==1.22.1
pip install onnx==1.10.2
pip install onnxoptimizer==0.2.6
pip install onnxruntime==1.10.0
pip install Pillow==9.0.0

```




### Step 2: run the python script:

```
python nmaps_maker.py --input my_image.png --overlap small
```



## Command line arguments:




``` -input ```    path to image or folder with images. List of supported image formats you can finde here: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

``` --overlap ```    overlap parameter of result normal map. allowed values: ``` small, medium, large```

## Links



Project uses code of DeepBump tool by HugoTini. Link to repository:  https://github.com/HugoTini/DeepBump

