# Spectrogram Segmentation

The successful application of [semantic segmentation](https://www.ibm.com/topics/semantic-segmentation) to radiofrequency (RF) spectrograms has significant 
implications for [spectrum sensing](https://iopscience.iop.org/article/10.1088/1742-6596/2261/1/012016#:~:text=In%20cognitive%20radio%2C%20spectrum%20sensing,user%20can%20use%20the%20spectrum.), and serves as a foundational example showcasing the near-term feasibility of 
[intelligent radio](https://www.qoherent.ai/intelligentradio/) technology.

In this example, we use [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/docs/pytorch/stable/) to train a segmentation model to identify and
differentiate between 5G NR and 4G LTE signals within wideband spectrograms.

Classification results are comparable to those achieved by MathWorks' custom network, albeit with more learnables. 
For more information, please refer to the following article by MathWorks: 
[Spectrum Sensing with Deep Learning to Identify 5G and LTE Signals](https://www.mathworks.com/help/comm/ug/spectrum-sensing-with-deep-learning-to-identify-5g-and-lte-signals.html).

If you found this example interesting or helpful, don't forget to give it a star! ⭐


## 🚀 Getting Started

This example is provided as a Jupyter Notebook. You have the option to either run this example locally or in Google 
Colab.

To run this example locally, you'll need to download the project and dataset and set up a Python 
virtual environment. If this seems daunting, we recommend running this example on Google Colab (Coming soon!).

### Running this example locally

Please note that running this example locally will require approximately 6.1 GB of free space. Please ensure you 
have sufficient space available prior to proceeding.

1. Ensure that [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads/) are installed on the computer where you plan to run this example. 
Additionally, if you'd like to accelerate model training with a GPU, you'll require [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).


2. Clone this repository to your local computer:
```commandline
git clone https://github.com/MahmoudUwk/Spectrogram-Segmentation.git
```


3. Create and activate a Python [virtual environment](https://docs.python.org/3/library/venv.html). This is a best practice for isolating project dependencies.

<details>
<summary><strong>Windows</strong></summary>

Use the following command to create a new directory named `venv` within the project directory:
```commandline
python -m venv venv
```

Then, activate the virtual environment with:
```commandline
venv\Scripts\activate
```

</details>

<details>
<summary><strong>Linux/Mac</strong></summary>

Use the following command to create a new directory named `venv` within the project directory:
```commandline
python3 -m venv venv
```

Then, activate the virtual environment with:
```commandline
source venv/bin/activate
```

</details>

Activating the virtual environment should modify the command prompt to show `(venv)` at the beginning, indicating 
that the virtual environment is active.


4. Install project dependencies from the provided `requirements.txt` file:
```commandline
pip install -r requirements.txt
```


5. Download the spectrum sensing dataset.

<details>
<summary><strong>Windows</strong></summary>

```commandline
python download_dataset.py
```

</details>

<details>
<summary><strong>Linux/Mac</strong></summary>

```commandline
python3 download_dataset.py
```

</details>

This will download the source data from MathWorks, correct a known mask alignment issue,
and produce `spectrum_sensing_dataset.hdf5` in the project's root directory.


6. Register the environment kernel with Jupyter:
```commandline
ipython kernel install --user --name=venv --display-name "Spectrogram Segmentation (venv)"
```


7. Open the notebook, `spectrogram_segmentation.ipynb`, specifying to use the new kernel:
```commandline
jupyter notebook spectrogram_segmentation.ipynb --MultiKernelManager.default_kernel_name=venv
```


8. Give yourself a pat on the back - you're all set up and ready to explore the example! For more information on 
navigating the Jupyter Notebook interface and executing code cells, please check out this tutorial by the Codecademy 
Team: [How To Use Jupyter Notebooks](https://www.codecademy.com/article/how-to-use-jupyter-notebooks).

Depending on your system specifications and the availability of a CUDA, running this example locally may take 
several minutes. If a cell is taking too long to execute, you can interrupt its execution by clicking the "Kernel" 
menu and selecting "Interrupt Kernel" or by pressing `Ctrl + C` in the terminal where Jupyter Notebook is running.


9. After you finish exploring, consider removing the dataset from your system and deleting the virtual environment to 
free up space. Remember to deactivate the virtual environment using the deactivate command before deleting the folder.


## 🤝 Contribution

We welcome contributions from the community! Whether it's an enhancement, bug fix, or improved explanation, 
your input is valuable. For significant changes, or if you'd like to prepare a separate tutorial, kindly 
[contact us](mailto:info@qoherent.ai) beforehand.

If you encounter any issues or to report a security vulnerability, please submit a bug report to the GitHub Issues 
page [here](https://github.com/qoherent/spectrogram-segmentation/issues).

Has this example inspired a project or research initiative related to intelligent radio? Please [get in touch](mailto:info@qoherent.ai); 
we'd love to collaborate with you! 📡🚀

Finally, be sure to check out our open-source project: [RIA Core](https://github.com/qoherent/ria) (Coming soon!).


## 🤝 Contribution

newline: git clone https://github.com/MahmoudUwk/Spectrogram-Segmentation.git
```

line 5: python3 download_dataset.py
```

line 6: Register the environment kernel with Jupyter:
```commandline
ipython kernel install --user --name=venv --display-name "Spectrogram Segmentation (venv)"
```

line 7. Open the notebook, `spectrogram_segmentation.ipynb`, specifying to use the new kernel
```commandline
jupyter notebook spectrogram_segmentation.ipynb --MultiKernelManager.default_kernel_name=venv
```

line 8. Give yourself a pat on the back - you're all set up and ready to explore the example! For more information on navigating the Jupyter Notebook interface and executing code cells, please check out this tutorial by the Codecademy 
28 Team: [How To Use Jupyter Notebooks](https://www.codecademy.com/article/how-to-use-jupyter-notebooks).

Depending on your system specifications and the the availability of a CUDA, running this example locally may take several minutes. If a cell is taking too long to execute, you can interrupt its execution by clicking the "Kernel" menu and selecting "Interrupt Kernel" or by pressing `Ctrl + C` in the terminal where Jupyter Notebook is running.


9. After you finish exploring, consider removing the dataset from your system and deleting the virtual environment to free up space. Remember to deactivate the virtual environment using the deactivate command before deleting the folder.


## Authorship

```

The DeepLabv3 models used in this example were initially proposed by Chen _et al._ and are further discussed 
in their 2017 paper titled '[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)'. The MobileNetV3 
backbone used in this example was developed by Howard _et al._ and is further discussed in their 2019 paper titled 
'[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)'. Models were accessed through [`torchvision`](https://pytorch.org/vision/stable/models/deeplabv3.html).

A special thanks to the PyTorch and Lightning teams for providing the foundational machine learning frameworks used in 
this example.
