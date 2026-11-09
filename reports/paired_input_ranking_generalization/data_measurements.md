# Data, noise and variance measurements

These tables format existing records. They do not fit models or evaluate new predictions. Full precision, further quantiles and every registered numeric field remain in the [numeric register](measurements/all_numeric_records.csv.gz). Sources are R0882, R1500, R1598 and R1602–R1608.

## Composition and repeated observations

The positive fraction in this first table uses the original observation-level binary labels, before partition-local pair-median relabeling. For example, B test is 0.78980110 here and 0.79023920 after pair-median relabeling. The latter is the model-label prevalence reported in Q15. Continuous-score composition and binary-label composition must not be conflated.

| Split | Partition | Rows | Original positive fraction | Constant-label groups | Label-tied pair fraction | Repeatd-row fraction |
| --- | --- | --- | --- | --- | --- | --- |
| A | train | 1569095 | 0.67959556 | 48107 | 0.05362456 | 0.03367801 |
| A | val | 195618 | 0.72616017 | 5990 | 0.07728980 | 0.03477185 |
| A | test | 192687 | 0.74247873 | 5828 | 0.06198784 | 0.03220248 |
| B | train | 1683487 | 0.67427310 | 51789 | 0.05369741 | 0.03173294 |
| B | val | 136957 | 0.78979534 | 3960 | 0.07163985 | 0.05167315 |
| B | test | 136956 | 0.78980110 | 4176 | 0.05654096 | 0.03907824 |


## Repeat disagreement and split-half agreement

RMS is raw-repeat disagreement, not the standard error of a median. The ranges below describe random half assignment, not population confidence intervals. The corrected quantity is a heuristic and is not an established full-population ceiling.

| Split | Partition | Repeatd observations | Repeat pairs | RMS difference | Qualifying groups | Original eligible groups | Coverage | Split-half macro | Corrected heuristic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | train | 52844 | 73595 | 0.78055161 | 3126 | 63867 | 0.04894546 | 0.53988937 | 0.70119906 |
| A | val | 6802 | 9070 | 0.81514179 | 421 | 7997 | 0.05264474 | 0.52609171 | 0.68941896 |
| A | test | 6205 | 9427 | 0.70986885 | 420 | 8278 | 0.05073689 | 0.54450467 | 0.70503949 |
| B | train | 53422 | 71674 | 0.76605203 | 3247 | 72201 | 0.04497168 | 0.57225944 | 0.72794041 |
| B | val | 7077 | 10340 | 0.86696457 | 372 | 3969 | 0.09372638 | 0.39941462 | 0.57073734 |
| B | test | 5352 | 10078 | 0.75750338 | 348 | 3972 | 0.08761329 | 0.37800495 | 0.54854535 |


| Split | Partition | Assignments | Repeatd members | Defined-group range | Assignment macro range | Positive-correlation assignments |
| --- | --- | --- | --- | --- | --- | --- |
| A | train | 100 | 42656 | [3085, 3099] | [0.5336612185191096, 0.5460984543961744] | 100 |
| A | val | 100 | 5744 | [413, 416] | [0.510020049984679, 0.5416894886915953] | 100 |
| A | test | 100 | 5146 | [413, 416] | [0.5277156729466493, 0.5606550862176375] | 100 |
| B | train | 100 | 42514 | [3212, 3224] | [0.5663274580986778, 0.5777539857644267] | 100 |
| B | val | 100 | 6401 | [365, 368] | [0.377590615401782, 0.4187429994312423] | 100 |
| B | test | 100 | 4631 | [333, 339] | [0.3598143217582853, 0.39722022678158636] | 100 |


## Ranking eligibility and threshold proximity

Native eligibility uses the float32-label path. Threshold proximity below is row-weighted proximity of partition-local pair medians to θ, using the training-derived raw-repeat RMS. It differs from proximity of each original observation score.

| Split | Partition | Native eligible | Sampler eligible | n < 3 | Insufficient span | Precision disagreements | Eligible same-side fraction | Pair-label proximity | Observation-score proximity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | train | 63867 | 63867 | 62516 | 5067 | 2 | 0.44500290 | 0.34127698 | 0.34051730 |
| A | val | 7997 | 7997 | 7785 | 649 | 0 | 0.45279480 | 0.35057612 | 0.34929812 |
| A | test | 8278 | 8278 | 7569 | 584 | 0 | 0.43633728 | 0.35581020 | 0.35512515 |
| B | train | 72201 | 72202 | 67994 | 5259 | 1 | 0.43865043 | 0.33783154 | 0.33722387 |
| B | val | 3969 | 3969 | 4763 | 548 | 0 | 0.50088183 | 0.32620458 | 0.32465664 |
| B | test | 3972 | 3971 | 5113 | 493 | 1 | 0.54204431 | 0.32211075 | 0.32029995 |


## Variance components

All entries below have units s². Primary and group-balanced fits use different weighting. Boundary estimates and sensitivity to weighting prevent a stable decomposition or ceiling claim. Component variance and its within-group contribution are different quantities. Query and group intercepts cancel within a group. Observation residual includes unmodeled signal; raw-repeat noise is not converted into median uncertainty.

| Split | Partition | Fit | Query | Group | Class main | Query × class | Raw-repeat noise | Observation residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | train | Primary | 0.36316400 | 3.40683625 | 0.29811891 | 0.00000000 | 0.00000000 | 0.00000000 |
| A | train | Group-balanced | 0.31130869 | 0.81755101 | 0.36437175 | 0.45027186 | 0.04223594 | 0.08902227 |
| A | val | Primary | 0.35876074 | 1.24074487 | 0.21953794 | 0.00000000 | 0.06259262 | 0.00000000 |
| A | val | Group-balanced | 0.33357722 | 0.88462053 | 0.42620968 | 0.20452573 | 0.00000000 | 0.22429004 |
| A | test | Primary | 0.25080549 | 0.92082342 | 0.47299692 | 0.00000000 | 0.10354355 | 0.03582322 |
| A | test | Group-balanced | 0.28316184 | 0.88221794 | 0.41437912 | 0.29836761 | 0.02785899 | 0.16746606 |
| B | train | Primary | 0.36157690 | 3.33659478 | 0.29268639 | 0.00000000 | 0.00000000 | 0.00000000 |
| B | train | Group-balanced | 0.29189843 | 0.81759738 | 0.37758667 | 0.44374595 | 0.01887461 | 0.12766858 |
| B | val | Primary | 0.51378023 | 0.74333462 | 0.00000000 | 0.73387973 | 0.00000000 | 0.00000000 |
| B | val | Group-balanced | 0.54119322 | 0.68128543 | 0.84976867 | 0.00000000 | 0.00000000 | 0.00000000 |
| B | test | Primary | 0.47727032 | 0.52266452 | 0.00000000 | 0.51153755 | 0.19314310 | 0.00000000 |
| B | test | Group-balanced | 0.64689457 | 0.71216824 | 0.58195614 | 0.16914319 | 0.00000000 | 0.00000000 |


## Within-group variance contributions

| Split | Partition | Fit | Class main | Query × class | Observation residual | Query + group |
| --- | --- | --- | --- | --- | --- | --- |
| A | train | Primary | 0.11280314 | 0.00000000 | 0.00000000 | 0.00000000 |
| A | train | Group-balanced | 0.13787209 | 0.17037524 | 0.04579903 | 0.00000000 |
| A | val | Primary | 0.08366921 | 0.00000000 | 0.00000000 | 0.00000000 |
| A | val | Group-balanced | 0.16243491 | 0.07794783 | 0.11582732 | 0.00000000 |
| A | test | Primary | 0.18376655 | 0.00000000 | 0.01888998 | 0.00000000 |
| A | test | Group-balanced | 0.16099264 | 0.11592039 | 0.08830669 | 0.00000000 |
| B | train | Primary | 0.11246336 | 0.00000000 | 0.00000000 | 0.00000000 |
| B | train | Group-balanced | 0.14508589 | 0.17050728 | 0.06656150 | 0.00000000 |
| B | val | Primary | 0.00000000 | 0.25942868 | 0.00000000 | 0.00000000 |
| B | val | Group-balanced | 0.30039577 | 0.00000000 | 0.00000000 | 0.00000000 |
| B | test | Primary | 0.00000000 | 0.17163753 | 0.00000000 | 0.00000000 |
| B | test | Group-balanced | 0.19526527 | 0.05675306 | 0.00000000 | 0.00000000 |


## Variance-fit diagnostics

| Split | Partition | Fit | Fitted marginal variance | Observed variance | Direct repeat-noise variance | Boundary components |
| --- | --- | --- | --- | --- | --- | --- |
| A | train | Primary | 4.06074557 | 1.95182357 | 0.29618666 | ['measurement_noise', 'observation_residual', 'x_by_candidate_category'] |
| A | train | Group-balanced | 2.07203480 | 1.95182357 | 0.29618666 | [] |
| A | val | Primary | 1.87948546 | 1.87948546 | 0.32244896 | ['observation_residual', 'x_by_candidate_category'] |
| A | val | Group-balanced | 2.07131037 | 1.87948546 | 0.32244896 | ['measurement_noise'] |
| A | test | Primary | 1.78246534 | 1.78246534 | 0.23648937 | ['x_by_candidate_category'] |
| A | test | Group-balanced | 2.07184689 | 1.78246534 | 0.23648937 | [] |
| B | train | Primary | 3.98438793 | 1.95783207 | 0.28100611 | ['measurement_noise', 'observation_residual', 'x_by_candidate_category'] |
| B | train | Group-balanced | 2.07490697 | 1.95783207 | 0.28100611 | [] |
| B | val | Primary | 1.98634307 | 1.64717058 | 0.36703236 | ['measurement_noise', 'observation_residual', 'candidate_category_main'] |
| B | val | Group-balanced | 2.06724506 | 1.64717058 | 0.36703236 | ['x_by_candidate_category', 'observation_residual', 'measurement_noise'] |
| B | test | Primary | 1.70107325 | 1.70107325 | 0.31066707 | ['observation_residual', 'candidate_category_main'] |
| B | test | Group-balanced | 2.10514189 | 1.70107325 | 0.31066707 | ['observation_residual', 'measurement_noise'] |
