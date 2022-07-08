## Getting Started

1- Install docker, and setup it up for 'Windows Subsystem for Linux 2' or wsl2. [link](https://docs.docker.com/desktop/windows/wsl/)  
2- Clone this repository to the directory of choice:
```
git clone https://github.com/mtorkashvand/trefide.git
```  
3- Build the docker image:
```
cd trefide
docker build -t trefide .
```  
4- Test the image by running demos provided by the authors:
```
docker run -it -p 34000:34000 trefide
```
This runs a docker container from the `trefide` docker image. Copy the line highlighted in yellow in the screenshot below and paste it in the browser of
your choice. Test demos in `funimag_demo` and `trefide_demo` folders to confirm this image behaves as advertised.  

<p align="center">
  <img width="750" height="375" src="https://user-images.githubusercontent.com/31863323/178042162-b8fca5dc-ac42-47fd-94a1-a25589a5efe8.PNG">
</p>



## Notes from the master branch

TreFiDe is the software package accompanying the research publication
["Penalized matrix decomposition for denoising, compression, and improved
demixing of functional imaging data"](https://doi.org/10.1101/334706).

TreFiDe is an imporved appproach to compressing and denoising functional image
data. The method is based on a spatially-localized penalized matrix
decomposition (PMD) of the data to separate (low-dimensional) signal from
(temporally-uncorrelated) noise. This approach can be applied in parallel on
local spatial patches and is therefore highly scalable, does not impose
non-negativity constraints or require stringent identifiability assumptions
(leading to significantly more robust results compared to NMF), and estimates
all parameters directly from the data, so no hand-tuning is required. We have
applied the method to a wide range of functional imaging data (including
one-photon, two-photon, three-photon, widefield, somatic, axonal, dendritic,
calcium, and voltage imaging datasets): in all cases, we observe ~2-4x
increases in SNR and compression rates of 20-300x with minimal visible loss of
signal, with no adjustment of hyperparameters; this in turn facilitates the
process of demixing the observed activity into contributions from individual
neurons. We focus on two challenging applications: dendritic calcium imaging
data and voltage imaging data in the context of optogenetic stimulation. In
both cases, we show that our new approach leads to faster and much more robust
extraction of activity from the video data.


## References
```
@article {Buchanan334706,
    author = {Buchanan, E. Kelly and Kinsella, Ian and Zhou, Ding and Zhu, Rong and Zhou, Pengcheng and Gerhard, Felipe and Ferrante, John and Ma, Ying and Kim, Sharon and Shaik, Mohammed and Liang, Yajie and Lu, Rongwen and Reimer, Jacob and Fahey, Paul and Muhammad, Taliah and Dempsey, Graham and Hillman, Elizabeth and Ji, Na and Tolias, Andreas and Paninski, Liam},
    title = {Penalized matrix decomposition for denoising, compression, and improved demixing of functional imaging data},
    elocation-id = {334706},
    year = {2019},
    doi = {10.1101/334706},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2019/01/21/334706},
    eprint = {https://www.biorxiv.org/content/early/2019/01/21/334706.full.pdf},
    journal = {bioRxiv}
}
```
