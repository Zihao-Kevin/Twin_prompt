#!/bin/bash

#bash main_pipeline_caltect_baseline.sh vit_b16_ep50 end 16 1 False False False 4
#bash main_pipeline_dtd_baseline.sh vit_b16_ep50 end 16 1 False False False 4
#bash main_pipeline_eurosat_baseline.sh vit_b16_ep50 end 16 1 False False False 4
#bash main_pipeline_flower_baseline.sh vit_b16_ep50 end 16 1 False False False 4
#bash main_pipeline_pet_baseline.sh vit_b16_ep50 end 16 1 False False False 4
bash main_pipeline_stanfordcar_baseline.sh vit_b16_ep50 end 16 1 False False False 4
bash main_pipeline_ucf_baseline.sh vit_b16_ep50 end 16 1 False False False 4

bash main_pipeline_caltect_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_dtd_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_eurosat_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_flower_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_pet_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_stanfordcar_baseline.sh rn50_ep50 end 16 1 False False False 4
bash main_pipeline_ucf_baseline.sh rn50_ep50 end 16 1 False False False 4