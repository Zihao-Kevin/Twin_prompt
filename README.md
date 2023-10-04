# Introduction
Federated Learning (FL) is often impeded by communication overhead issues. Prompt tuning, as a potential solution, has been introduced to only adjust a few trainable parameters rather than the whole model. However, current single-modality prompt tuning approaches fail to comprehensively portray local clients' data. To overcome this limitation, we present Twin Prompt Federated learning (TPFL), a pioneering solution that integrates both visual and textual modalities, ensuring a more holistic representation of local clients' data characteristics. Furthermore, in order to tackle the data heterogeneity issues, we introduce the Augmented TPFL (ATPFL) employing the contrastive learning to TPFL, which not only enhances the global knowledge acquisition of client models but also fosters the development of robust, compact models. The effectiveness of TPFL and ATPFL is substantiated by our extensive evaluations, consistently showing superior performance compared to all baselines.


## How to Run

You can run `federated_main.py` with some specified arguments.

### Training

`--root` takes as input a path to dataset, like `caltech101` or `oxford_flowers`.

`--config-file` means which config file to use, such as `rn50_ep50` or `vit_b16_ep50`.

You can select variables like shots, users by changing `cfg` or you can change every arguments you like in `main_pipeline.sh`.

### For example
**ATPFL (M=16, end)**:
If you want to train caltech100 with 2 shots, backbone rn50 and total independent non-iid setting.
You can specify that:
`TRAINER=PromptFL`
`DATA=caltech101`
`SHOTS=2`
`REPEATRATE=0.0`
and run `bash main_pipeline.sh rn50_ep50 end 16 False False False`
**FinetuningFL**:
If you want to train caltech100 with fintuning, backbone rn50 and total independent non-iid setting.
You can specify that:
`TRAINER=Baseline`
`DATA=caltech101`
`SHOTS=1`
`REPEATRATE=0.0`
and run `bash main_pipeline.sh rn50_ep50 end 16 False False True`

After the experiments, all the results are finished and save to `output/`.
We build and modify the code based on Dassl and CoOp.
We will release the full-version and detailed description later to help faciliate the community and further study.


## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@inproceedings{zhao2023inclusive,
  title={Inclusive Data Representation in Federated Learning: A Novel Approach Integrating Textual and Visual Prompt},
  author={Zhao, Zihao and Shi, Zhenpeng and Liu, Yang and Ding, Wenbo},
  booktitle={Adjunct Proceedings of the 2023 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings of the 2023 ACM International Symposium on Wearable Computers},
  year={2023}
}
```
