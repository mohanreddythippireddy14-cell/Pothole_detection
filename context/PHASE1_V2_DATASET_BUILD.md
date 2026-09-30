# Phase 1 v2 — merged mega-dataset build

## Source accounting
- andrewmvd: 665 photos -> 0 kept as-is (rest superseded by chitholian's higher-res duplicates)
- chitholian: 665 photos, 665 replacing an andrewmvd duplicate (higher native resolution: 720x720 vs ~400x300), 0 genuinely new
- rajdalsaniya: 2083 files -> 671 unique base photos kept (Roboflow export-time augmentation converged back to 1 rep/photo to avoid k-fold leakage), perceptual-hash-verified as 0/60-sampled-overlap with andrewmvd/chitholian
- banilkumar: 650 background (pothole-free) photos included; 650 classification-only 'Pothole' photos excluded (no bounding boxes, doesn't fit schema)

## Result: 1986 images (1336 annotated + 650 background) vs. original Phase 1: 665 images, 0 background

## Converged geometry features (resolution-independent; raw pixel width/height DROPPED
## from all cross-source statistics since sources have different native resolutions)

source                 chitholian_replaces_andrewmvd  rajdalsaniya
relative_width  count                    1739.000000   1358.000000
                mean                        0.251353      0.327783
                std                         0.198160      0.229372
                min                         0.022222      0.008514
                25%                         0.106943      0.150484
                50%                         0.191537      0.261647
                75%                         0.324151      0.446262
                max                         0.998148      1.000000
relative_height count                    1739.000000   1358.000000
                mean                        0.166441      0.322641
                std                         0.153463      0.230805
                min                         0.005556      0.019698
                25%                         0.061111      0.147565
                50%                         0.119444      0.256056
                75%                         0.210000      0.427742
                max                         0.946667      1.000000
relative_area   count                    1739.000000   1358.000000
                mean                        0.066409      0.147906
                std                         0.116528      0.194358
                min                         0.000177      0.001163
                25%                         0.007156      0.021920
                50%                         0.021411      0.067221
                75%                         0.066098      0.181934
                max                         0.852298      1.000000
aspect_ratio    count                    1739.000000   1358.000000
                mean                        1.854809      1.132717
                std                         1.103468      0.601950
                min                         0.299499      0.047923
                25%                         1.188464      0.750559
                50%                         1.601694      0.995021
                75%                         2.209084      1.347366
                max                        17.259140      7.158791
