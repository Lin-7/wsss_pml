#!/bin/sh

# python train_cls_loc_jointly_new.py --session_name=e5-patch_weight0.1-all-10patch_randomstart-fgmid0.4-seed7 --max_epoches=5 \
#         --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --patch_loss_weight=0.1

python train_cls_loc_jointly_new.py --session_name=e5-patch_weight0.05-all-10patch_randomstart-fgmid0.4-seed7 --max_epoches=5 \
        --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --patch_loss_weight=0.05
sleep 10s
python train_cls_loc_jointly_new.py --session_name=e5-patch_weight0.04-all-10patch_randomstart-fgmid0.4-seed7 --max_epoches=5 \
        --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --patch_loss_weight=0.04
sleep 10s
python train_cls_loc_jointly_new.py --session_name=e5-patch_weight0.06-all-10patch_randomstart-fgmid0.4-seed7 --max_epoches=5 \
        --patch_gen=randompatch --patch_select_cri=fgratio --patch_select_ratio=0.4 --patch_select_part=mid --patch_loss_weight=0.06
sleep 10s

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