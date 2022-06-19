#CDAN (Office-Home-ViT)
python cdan.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Cl --log_name Ar2Cl_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_vit --gpu 0  --log_results

python cdan.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_vit --gpu 0  --log_results

python cdan.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_vit --gpu 0  --log_results

python cdan.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_vit --gpu 0  --log_results
python cdan.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_vit/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_vit --gpu 0  --log_results

#CDAN_SDAT (Office-Home-ViT)
python cdan_sdat.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Ar2Cl --log_name Ar2Cl_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results

python cdan_sdat.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results

python cdan_sdat.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results

python cdan_sdat.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results
python cdan_sdat.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_sdat_vit/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_sdat_vit --gpu 0 --rho 0.02 --log_results

#CDAN_MCC (Office-Home-ViT)
python cdan_mcc.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Ar2Cl --log_name Ar2Cl_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results

python cdan_mcc.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results

python cdan_mcc.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results

python cdan_mcc.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results
python cdan_mcc.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_vit/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_mcc_vit --gpu 0 --lr 0.002 --log_results

#CDAN_MCC_SDAT (Office-Home-ViT)
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Ar2Cl --log_name Ar2Cl_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results

python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results

python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Rw -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results

python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Ar -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Cl -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Pr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --no-pool --log logs/cdan_mcc_sdat_vit/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_mcc_sdat_vit --gpu 0 --rho 0.02 --lr 0.002 --log_results
