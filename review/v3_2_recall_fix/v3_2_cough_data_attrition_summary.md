# v3.2 Cough Data Attrition Summary

- raw cough windows: `2729`
- reviewed cough events: `180`
- candidate cough rows: `154`
- final train addon cough rows: `35`

## Person-level highlights

- p01 cough entering final train addon: `35`
- p02 cough excluded as eval_sanity: `25`
- p03 cough excluded as unseen holdout: `59`
- 70cm/100cm cough excluded as distance extrap: `0`

## Folder attrition overview

- `100cm_front_p01_cough` raw_windows `160`, review_events `12`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `148`
- `100cm_front_p02_cough` raw_windows `160`, review_events `7`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `153`
- `100cm_front_p03_cough` raw_windows `160`, review_events `8`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `152`
- `40cm_front_p01_cough` raw_windows `180`, review_events `12`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `168`
- `40cm_front_p02_cough` raw_windows `160`, review_events `7`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `153`
- `40cm_front_p03_cough` raw_windows `160`, review_events `15`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `145`
- `40cm_left_45dig_p01_cough` raw_windows `160`, review_events `12`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `148`
- `40cm_left_45dig_p02_cough` raw_windows `160`, review_events `6`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `154`
- `40cm_left_45dig_p03_cough` raw_windows `160`, review_events `14`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `146`
- `40cm_right_45dig_p01_cough` raw_windows `159`, review_events `11`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `148`
- `40cm_right_45dig_p02_cough` raw_windows `158`, review_events `9`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `149`
- `40cm_right_45dig_p03_cough` raw_windows `158`, review_events `12`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `146`
- `70cm_front_p01_cough` raw_windows `160`, review_events `12`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `148`
- `70cm_front_p02_cough` raw_windows `160`, review_events `7`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `153`
- `70cm_front_p03_cough` raw_windows `158`, review_events `13`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `145`
- `轻咳，闷咳，连续咳_p01` raw_windows `160`, review_events `9`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `151`
- `轻咳，闷咳，连续咳_p02` raw_windows `78`, review_events `7`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `71`
- `轻咳，闷咳，连续咳_p03_atention` raw_windows `78`, review_events `7`, candidate_cough `0`, final_train_addon_cough `0`, holdout(sanity/unseen/distance) `0/0/0`, unreviewed_high_value_est `71`

## Recommended remediation

- Re-draw review clips from folders with the highest unreviewed cough window estimates first.
- Prioritize p01 train_candidate folders if the goal is to add more positive_recall_fix_candidate without breaking unseen/distance holds.
- Avoid consuming p03 unseen-person folders or 70cm/100cm distance-extrap folders into train unless the evaluation policy is explicitly changed.
- If more positives are needed without breaking holdouts, prioritize remaining unreviewed windows in p01 40cm / near-field folders and any p02 material only if it stays eval_sanity-only.
