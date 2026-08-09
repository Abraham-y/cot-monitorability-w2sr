# Task H split — length-binned ack by influenced

```
TASK H SPLIT — length-binned acknowledgment by influenced status

bin edges (chars, union of full distributions): [-1, 1118, 1465, 1954, 8716, 37816]

### INFLUENCED = 1 (answer hit the cue target)   (baseline n=44,  W2SR n=350)

bin          range (chars)  n_b  n_w          ack_b          ack_w  Δ (pp)    OR   Fisher 2-side p   1-side p
  0 [0, 1,118)   skipped (empty side)
  1 [1,118, 1,465)   skipped (empty side)
  2 [1,465, 1,954)   skipped (empty side)
  3         [1,954, 8,716)   10   76     7/10=70.0%    16/76=21.1%  +48.9  8.75            0.0030     0.0030
  4        [8,716, 37,816)   34   10    23/34=67.6%     4/10=40.0%  +27.6  3.14            0.1497     0.1142

### INFLUENCED = 0 (answer did not hit the cue target)   (baseline n=79,  W2SR n=263)

bin          range (chars)  n_b  n_w          ack_b          ack_w  Δ (pp)    OR   Fisher 2-side p   1-side p
  0 [0, 1,118)   skipped (empty side)
  1 [1,118, 1,465)   skipped (empty side)
  2         [1,465, 1,954)    1   75       0/1=0.0%      1/75=1.3%   -1.3  0.00            1.0000     1.0000
  3         [1,954, 8,716)   30   42      0/30=0.0%      2/42=4.8%   -4.8  0.00            0.5070     1.0000
  4        [8,716, 37,816)   48   10    10/48=20.8%     2/10=20.0%   +0.8  1.05            1.0000     0.6615
```
