#!/bin/bash

# List of Python scripts to run in serial order along with their arguments
python_scripts_with_args=(
    "train_cls_loc_jointly_new.py --session_name=fgfrontselect0.3-patchweight0.2 --patch_select_cri=fgratio --patch_select_part_fg=front"
    "train_cls_loc_jointly_new.py --session_name=fgbackselect0.3-patchweight0.2 --patch_select_cri=fgratio --patch_select_part_fg=back"
    "train_cls_loc_jointly_new.py --session_name=confidmidselect0.3-patchweight0.2 --patch_select_cri=confid --patch_select_part_fg=mid"
    "train_cls_loc_jointly_new.py --session_name=confidfrontselect0.3-patchweight0.2 --patch_select_cri=confid --patch_select_part_fg=front"
    "train_cls_loc_jointly_new.py --session_name=confidbackselect0.3-patchweight0.2 --patch_select_cri=confid --patch_select_part_fg=back"
)

# Loop through the scripts and run them one by one with arguments
for script_with_args in "${python_scripts_with_args[@]}"; do
    script="${script_with_args%% *}"     # Extract the script name
    args="${script_with_args#* }"        # Extract the arguments after the space

    echo "Running $script with arguments: $args"
    nohup python "$script" $args &
    wait
done

echo "All scripts have been executed."

# nohup python train_cls_loc_jointly_new.py --session_name=fgmidselect0.3-patchweight0.2 \
#     --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part_fg=mid &

# 0828
# nohup python train_cls_loc_jointly_new.py --session_name=base-patchweight0.01 --patch_loss_weight=0.01 2>base-pw0.01 &

# nohup python train_cls_loc_jointly_new.py --session_name=base-patchweight0.1 --patch_loss_weight=0.1 2>base-pw0.1 &
# sleep 10s
# nohup python train_cls_loc_jointly_new.py --session_name=randompatch-patchweight0.2 --patch_loss_weight=0.2 \
#     --max_epoches=3 --patch_gen=randompatch --patch_select_close=True \
#      0>randompatch-pw0.2.out 1>randompatch-pw0.2.log 2>randompatch-pw0.2.err &

# # sleep 30s
# nohup python train_cls_loc_jointly_new.py --session_name=randomselect0.3-patchweight0.2 --patch_loss_weight=0.2 \
#     --max_epoches=3 --patch_gen=4patch --patch_select_cri=random --patch_select_ratio=0.3  \
#      0>randomselect0.3-pw0.2.out 1>randomselect0.3-pw0.2.log 2>randomselect0.3-pw0.2.err &

# sleep 30s
# nohup python train_cls_loc_jointly_new.py --session_name=fgmidselect0.3-patchweight0.2 --patch_loss_weight=0.2 \
#     --max_epoches=3 --patch_gen=4patch --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part_fg=mid \
#      0>fgmidselect0.3-pw0.2.out 1>fgmidselect0.3-pw0.2.log 2>fgmidselect0.3-pw0.2.err &

# # 2.4
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-diversity_thres0.9-0.4-seed7-2.10 \
#         --patch_gen=randompatch\
#         --patch_select_cri=random --patch_select_ratio=0.4 --patch_select_checksimi=1 --patch_select_checksimi_thres=0.9\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-diversity_thres0.9-0.4-seed7-2.6 \
#         --patch_gen=randompatch\
#         --patch_select_cri=random --patch_select_ratio=0.4 --patch_select_checksimi=1 --patch_select_checksimi_thres=0.9\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-confidfront0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=confid --patch_select_ratio=0.4 --patch_select_part_confid=front\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-confidmid0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=confid --patch_select_ratio=0.4 --patch_select_part_confid=mid\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-confidback0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=confid --patch_select_ratio=0.4 --patch_select_part_confid=back\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-0.3fgmid_confidback0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=fgAndconfid --patch_select_ratio=0.4 --patch_select_part_fg=mid --patch_select_part_confid=back\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-0.3fgfront_confidback0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=fgAndconfid --patch_select_ratio=0.4 --patch_select_part_fg=front --patch_select_part_confid=back\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-0.3fgback_confidback0.4-seed7 \
#         --patch_gen=randompatch --patch_select_cri=fgAndconfid --patch_select_ratio=0.4 --patch_select_part_fg=back --patch_select_part_confid=back\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s


# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-fgmid0.4-seed7-newcodetest \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part_fg=mid\
#         --patch_loss_weight=0.05 --max_epoches=3
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.001-all-10patch_randomstart-fgmid0.4-seed7 --max_epoches=3 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --patch_loss_weight=0.001
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.1-all-10patch_ccrop_a1.0_scale0.25_ratio11-fgmid0.4-seed7 --max_epoches=3 \
#         --patch_gen=contrastivepatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --ccrop_alpha=1.0 --patch_loss_weight=0.1

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.1-all-10patch_ccrop_a0.7_scale0.25_ratio11-fgmid0.4-seed7 --max_epoches=3 \
#         --patch_gen=contrastivepatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --ccrop_alpha=0.7 --patch_loss_weight=0.1


# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-padding0.25-10patch_randomstart-fgfront0.4-seed7 --max_epoches=3 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=front --proposal_padding=0.25
# sleep 10s
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-padding0.25-10patch_randomstart-fgback0.4-seed7 --max_epoches=3 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=back --proposal_padding=0.25
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-fgfront0.4-seed7 --max_epoches=3 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=front
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-10patch_randomstart-fgback0.4-seed7 --max_epoches=3 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=back
# sleep 10s

# # 0.3
# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-4patch-fgrandom0.3-6 --max_epoches=3 \
#         --patch_gen=4patch --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part=random
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-4patch-fgmid0.3-6 --max_epoches=3 \
#         --patch_gen=4patch --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part=mid
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-4patch-fgfront0.3-6 --max_epoches=3 \
#         --patch_gen=4patch --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part=front
# sleep 10s

# python train_cls_loc_jointly_new.py --session_name=e3-patch_weight0.05-all-4patch-fgback0.3-6 --max_epoches=3 \
#         --patch_gen=4patch --patch_select_cri=fgratio --patch_select_ratio=0.3 --patch_select_part=back
# sleep 10s