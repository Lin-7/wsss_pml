#!/bin/sh



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