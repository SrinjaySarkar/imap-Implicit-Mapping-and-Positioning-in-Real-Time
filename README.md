Pytorch Implementation of iMAP: Implicit Mapping and Positioning in Real-Time.Based on the paper:

  > [iMAP: Implicit Mapping and Positioning in Real-Time](https://arxiv.org/abs/2103.12352)\
  > Edgar Sucar, Shikun Liu, Joseph Ortiz, Andrew J. Davison\
  > arXiv:2103.12352


  Experiments on implementing SLAM using implicit representations and Nerf like volume rednering with a few added changes to the architecture.

  Download the official TUM RGBD dataset [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_teddy)

  ### To-DO
- [x] Add visualizaton of results obtained.
- [ ] Change architrecture from single MLP to heirarchical feature grids in order to improve local reconstruction details. 
- [ ] Add viewing direction for mapping to check if modelling specularities improves results.
- [ ] Adding semantic branch for keyframes.

### Paper Notes 
Notes I referred to while implementation.

- [iMAP](https://dramatic-durian-120.notion.site/iLabel-iMAP-50087dace65d45269fa54e4515bd3ebe)

### Results
(./teddy.avi "Reconstructed Bag")

