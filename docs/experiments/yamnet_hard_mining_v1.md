# YAMNet hard mining v1

YAMNet is used only for pre-screening. This run does not modify formal labels and does not train a model.

## Summary

- Audio files scanned: 100
- Total windows: 1714
- Candidate CSV: `data\mining_candidates\yamnet_non_cough_100files\candidate_pool.csv`
- Errors CSV: `data\mining_candidates\yamnet_non_cough_100files\errors.csv`

## Decision counts

- hard_negative_review: 6
- baseline_suspicious_review: 179
- uncertain_review: 88
- pseudo_non_cough_candidate: 867
- exclude: 107
- unselected: 467

## Missing YAMNet class names

Respiratory sounds

## Hard negative review top 20

| sample_id | audio_file | start_time | end_time | yamnet_top1 | yamnet_top1_score | yamnet_cough_score | baseline_cough_prob | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100249_00001500_bdc4a5cc92 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100249.wav | 1.5 | 2.5 | Pig | 0.352756 | 8.3e-05 | 0.862215 | baseline_false_alarm_like_non_cough |
| 100249_00003000_f5eadffdd7 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100249.wav | 3.0 | 4.0 | Animal | 0.36268 | 8e-05 | 0.856125 | baseline_false_alarm_like_non_cough |
| 100021_00000000_0b0c61726d | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100021.wav | 0.0 | 1.0 | Glass | 0.361854 | 0.002585 | 0.84278 | baseline_false_alarm_like_non_cough |
| 100312_00000000_fe315af9a9 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100312.wav | 0.0 | 0.842812 | Burping, eructation | 0.501364 | 9.7e-05 | 0.80325 | baseline_false_alarm_like_non_cough |
| 100081_00000500_13164051e0 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100081.wav | 0.5 | 1.5 | Speech | 0.317648 | 0.001305 | 0.767643 | baseline_false_alarm_like_non_cough |
| 100021_00000500_a3b14e0289 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100021.wav | 0.5 | 1.5 | Silence | 0.500001 | 5.9e-05 | 0.754736 | baseline_false_alarm_like_non_cough |


## Baseline suspicious review top 20

| sample_id | audio_file | start_time | end_time | yamnet_top1 | yamnet_top1_score | yamnet_cough_score | baseline_cough_prob | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100476_00005500_4405254020 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 5.5 | 6.5 | Vehicle horn, car horn, honking | 0.407549 | 0.0 | 0.992779 | baseline_high_yamnet_low_cough_on_negative_source |
| 1005_00006000_9f4b13f986 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\1005.wav | 6.0 | 7.0 | Alarm | 0.220892 | 2e-06 | 0.992297 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00009500_b7f57943ee | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 9.5 | 10.5 | Music | 0.189768 | 0.0 | 0.991511 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00013000_5d3f933e8d | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 13.0 | 14.0 | Music | 0.479798 | 0.0 | 0.98742 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00011000_1ce66037c1 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 11.0 | 12.0 | Music | 0.597921 | 0.0 | 0.986666 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00013500_eb6fa06ace | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 13.5 | 14.5 | Music | 0.388864 | 1e-06 | 0.98611 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00011500_f1ca78802d | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 11.5 | 12.5 | Music | 0.574881 | 0.0 | 0.985443 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00003000_118f9104e3 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 3.0 | 4.0 | Music | 0.425099 | 0.0 | 0.983714 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00012500_49975cfb6e | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 12.5 | 13.5 | Music | 0.661888 | 0.0 | 0.978684 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00002500_5965021855 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 2.5 | 3.5 | Music | 0.273286 | 0.0 | 0.977546 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00010000_ad60973dee | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 10.0 | 11.0 | Music | 0.46764 | 0.0 | 0.97455 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00006000_c6a790fae1 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 6.0 | 7.0 | Music | 0.37651 | 0.0 | 0.971276 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00009000_6342e1e8e2 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 9.0 | 10.0 | Music | 0.293069 | 0.0 | 0.970967 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00010500_f4579749d4 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 10.5 | 11.5 | Music | 0.656872 | 0.0 | 0.97051 | baseline_high_yamnet_low_cough_on_negative_source |
| 100309_00000500_302b598838 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100309.wav | 0.5 | 1.5 | Music | 0.222852 | 0.003147 | 0.96938 | baseline_high_yamnet_low_cough_on_negative_source |
| 100476_00007000_fc4de7ef2a | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 7.0 | 8.0 | Music | 0.187962 | 6.7e-05 | 0.96783 | baseline_high_yamnet_low_cough_on_negative_source |
| 100465_00000000_549d8386a7 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100465.wav | 0.0 | 0.87375 | Silence | 0.518868 | 5e-06 | 0.964194 | baseline_high_yamnet_low_cough_on_negative_source |
| 1005_00009500_253f775139 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\1005.wav | 9.5 | 10.5 | Animal | 0.180556 | 0.00019 | 0.962683 | baseline_high_yamnet_low_cough_on_negative_source |
| 1005_00012500_0a20a11eb8 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\1005.wav | 12.5 | 13.5 | Livestock, farm animals, working animals | 0.294805 | 6.8e-05 | 0.955257 | baseline_high_yamnet_low_cough_on_negative_source |
| 1005_00004500_10923abf84 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\1005.wav | 4.5 | 5.5 | Music | 0.179772 | 7e-06 | 0.952411 | baseline_high_yamnet_low_cough_on_negative_source |


## Pseudo cough candidate top 20

_None._


## Uncertain review top 20

| sample_id | audio_file | start_time | end_time | yamnet_top1 | yamnet_top1_score | yamnet_cough_score | baseline_cough_prob | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100476_00017000_284a5c290b | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 17.0 | 18.0 | Music | 0.597984 | 1e-06 | 0.598931 | model_disagreement_or_near_threshold |
| 100317_00001500_fe3161d114 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100317.wav | 1.5 | 2.5 | Fart | 0.520585 | 8.6e-05 | 0.598549 | model_disagreement_or_near_threshold |
| 1005_00010000_42a2fa7c5e | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\1005.wav | 10.0 | 11.0 | Speech | 0.319603 | 1.1e-05 | 0.594978 | model_disagreement_or_near_threshold |
| 100357_00010000_c2747d061c | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100357.wav | 10.0 | 11.0 | Music | 0.526896 | 0.0 | 0.594894 | model_disagreement_or_near_threshold |
| 100069_00004500_d8233589d8 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100069.wav | 4.5 | 5.5 | Fart | 0.715452 | 0.00072 | 0.587643 | model_disagreement_or_near_threshold |
| 100078_00000500_78a25d98ad | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100078.wav | 0.5 | 1.5 | Fart | 0.413207 | 0.0 | 0.587074 | model_disagreement_or_near_threshold |
| 100357_00003500_949d6e8e9a | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100357.wav | 3.5 | 4.5 | Music | 0.389465 | 0.0 | 0.586399 | model_disagreement_or_near_threshold |
| 100265_00017000_15b1e5085c | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100265.wav | 17.0 | 18.0 | Steam | 0.538688 | 0.0 | 0.583066 | model_disagreement_or_near_threshold |
| 100144_00000000_5272d658b9 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100144.wav | 0.0 | 1.0 | Fireworks | 0.723589 | 0.0 | 0.579193 | model_disagreement_or_near_threshold |
| 100308_00000000_d0af3cc7ef | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100308.wav | 0.0 | 1.0 | Fart | 0.517675 | 2.1e-05 | 0.576215 | model_disagreement_or_near_threshold |
| 100357_00008000_ed9d7f87ca | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100357.wav | 8.0 | 9.0 | Sidetone | 0.348205 | 0.0 | 0.574172 | model_disagreement_or_near_threshold |
| 100459_00011500_bb00cda6eb | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100459.wav | 11.5 | 12.5 | Busy signal | 0.281391 | 0.0 | 0.574065 | model_disagreement_or_near_threshold |
| 100317_00002000_ed4af0a886 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100317.wav | 2.0 | 3.0 | Fart | 0.375739 | 0.044041 | 0.573072 | model_disagreement_or_near_threshold |
| 100357_00006500_767fcc52db | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100357.wav | 6.5 | 7.5 | Music | 0.42276 | 0.0 | 0.572606 | model_disagreement_or_near_threshold |
| 100476_00004500_de202c4c28 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 4.5 | 5.5 | Music | 0.247815 | 0.0 | 0.571072 | model_disagreement_or_near_threshold |
| 100357_00005500_f3a566e31e | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100357.wav | 5.5 | 6.5 | Music | 0.487445 | 0.0 | 0.570497 | model_disagreement_or_near_threshold |
| 100249_00000500_3a0b82f695 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100249.wav | 0.5 | 1.5 | Speech | 0.122653 | 0.0 | 0.569618 | model_disagreement_or_near_threshold |
| 100462_00001000_1a3df2c1ca | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100462.wav | 1.0 | 2.0 | Music | 0.712549 | 0.0 | 0.566263 | model_disagreement_or_near_threshold |
| 100476_00000500_6ef11c64bc | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100476.wav | 0.5 | 1.5 | Vehicle horn, car horn, honking | 0.291794 | 0.0 | 0.563523 | model_disagreement_or_near_threshold |
| 100462_00006000_9fbf4bce91 | D:\cough_model_train\DATA\non_cough\FSD50K_Negative_clean\100462.wav | 6.0 | 7.0 | Music | 0.709648 | 0.0 | 0.561075 | model_disagreement_or_near_threshold |


## Review recommendation

Review `hard_negative_review` first, then `baseline_suspicious_review`, then `pseudo_cough_candidate`, then `uncertain_review`. `pseudo_non_cough_candidate` is low priority and should not be promoted without spot checks.
