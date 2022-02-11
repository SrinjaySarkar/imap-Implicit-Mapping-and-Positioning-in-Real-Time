Pytorch Implementation of iMAP: Implicit Mapping and Positioning in Real-Time.Based on the paper:

  > [iMAP: Implicit Mapping and Positioning in Real-Time](https://arxiv.org/abs/2103.12352)\
  > Edgar Sucar, Shikun Liu, Joseph Ortiz, Andrew J. Davison\
  > arXiv:2103.12352


  Experiments on implementing SLAM using implicit representations and Nerf like volume rednering with a few added changes to the architecture.

  Download the official TUM RGBD dataset [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_teddy)

  ### To-DO
- [x] Add visualizaton of results obtained.
- [ ] Change architrecture from single MLP to heirarchical feature grids in order to improve local reconstruction details
- [ ] Change architrecture from single MLP to [progressive representations](https://arxiv.org/pdf/2202.04713.pdf). 
- [ ] Add viewing direction for mapping to check if modelling specularities improves results.
- [ ] Adding semantic branch for keyframes.

### Paper Notes 
Notes I referred to while implementation.

- [iMAP](https://dramatic-durian-120.notion.site/iLabel-iMAP-50087dace65d45269fa54e4515bd3ebe)

### Results
I used a much smaller than the original Nerf Model used in the paper (exactly half the number of parameter) hence the results are not as good as in the paper.Please downlaod the "teddy.mp4" for the original video.


<img src="https://user-images.githubusercontent.com/25768975/152690630-c315e0e9-247b-4136-bd30-af655d3e8c29.gif" width="250" height="250"/>
<!---
![Frieburg Teddy](https://user-images.githubusercontent.com/25768975/152690630-c315e0e9-247b-4136-bd30-af655d3e8c29.gif,width="250" height="250")
--->
