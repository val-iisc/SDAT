#CDAN (VisDA2017-ViT)
python cdan.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --epochs 15 --seed 0 --lr 0.01 --per-class-eval --train-resizing cen.crop --log logs/cdan_vit/VisDA2017 --log_name visda_cdan_vit --gpu 0 --no-pool --log_results

#CDAN_SDAT (VisDA2017-ViT)
python cdan_sdat.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --epochs 15 --seed 0 --lr 0.01 --per-class-eval --train-resizing cen.crop --log logs/cdan_sdat_vit/VisDA2017 --log_name visda_cdan_sdat_vit --gpu 0 --no-pool --rho 0.005 --log_results

#CDAN_MCC (VisDA2017-ViT)
python cdan_mcc.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --epochs 15 --seed 0 --lr 0.002 --per-class-eval --train-resizing cen.crop --log logs/cdan_mcc_vit/VisDA2017 --log_name visda_cdan_mcc_vit --gpu 0 --no-pool --log_results

#CDAN_MCC_SDAT (VisDA2017-ViT)
python cdan_mcc_sdat.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --epochs 15 --seed 0 --lr 0.002 --per-class-eval --train-resizing cen.crop --log logs/cdan_mcc_sdat_vit/VisDA2017 --log_name visda_cdan_mcc_sdat_vit --gpu 0 --no-pool --rho 0.02 --log_results



