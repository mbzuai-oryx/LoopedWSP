# Claim ledger (every number the write-up uses, with its evidence)

Setup: 15x15 mazes (my generator, maze.py) unless stated; test = make_dataset(100, 15, seed=99); 1,000 training mazes.
Architecture = LoopedWSP maze config (d=35, 4 heads d_k=8, 86-neuron ReLU MLP, no norm), ported in wsp.py.

## Port / setup
C1. The torch port with the real weights (weights/maze.npz from the LoopedWSP repo) solves 80/80 generated mazes (20 each at 11, 15, 21, 31) — verify.py output.
C2. trajectory_w() matches the written block's rounded dims 7..14 at every loop on 100/100 mazes (50 at 11, 50 at 15); scratch dims 15..34 are 0 after the run — check_traj.py.
C3. Layout: dims 7-9 distance mod 3, 10-13 pointers D,L,R,U, 14 path flag; pointers and path lag the wave by one loop — mapdims.py.

## Reproducing the failure
C4. No norm, random init: the state RMS grows from 0.098 (loop 1) to 47.8 (loop 16), 8.4e7 (loop 56) and 3.7e30 (loop 200), about x6 every 5 loops (results/diag.json init_state_rms_nonorm, sampled every 5 loops). A1 (BPTT, no norm, lr 1e-4) hits a non-finite loss at step 1, before any update (results/A1*.json diverged_at=1).
C5. With pre-RMSNorm the state RMS grows linearly (0.32 -> 85 over 200 loops) and a 1e-3 perturbation neither grows nor shrinks (diag.json).
C6. Paper-style BPTT (RMSNorm, per-loop state MSE + readout CE, 11x11): 0% of changing cells right and 0 mazes solved at steps 500 and 1000 (logs/A2n11.log).
C7. Teacher-forced one-step training with the readout CE attached (B1): one-step cell accuracy 99.06% = the copy baseline; 0% on changing cells; attention ~uniform (~0.004 ≈ 1/225 on each neighbour) (logs/B1.log, results/analyze_step.json run on B1).

## Diagnosis
C8. Gradient on attention scores at init is ~1e-8 with no consistent preference for neighbour offsets (results/diag4.json).
C9. Even with the written V/O/MLP downstream and uniform attention, the gradient per score is ~2-5e-7 and identical for all 4 heads (no direction preference between heads) (results/diag5.json).
C10. The readout CE blocks learning: oracle-gather routing, state-MSE only learns the step (95% changing-cell acc by 1,500 steps at lr 1e-3; 100% at lr 3e-3 by 1,500); with CE added it stays 0% at lr 1e-3 and 3e-3 over 1,500 steps (logs/dbg_*.log).
C11. With the readout detached (R1-R4, 5,000 steps, lr 2e-3): learned absolute positions 66% changing-cell acc; frozen written coordinates 0%; learnable relative bias 90%; oracle gather 98% (results/analyze_step.json from analyze_step.py; logs/R*.log).
C12. Frozen written coordinates saturate attention at init: mean max weight 0.616, 33% of rows >99% on one cell; learned tables: 0.004 (uniform) (sat.py output).
C13. Relbias heads: head0 0.54 on Down, head3 0.46 on Left, head2 ~0.2-0.26 on all 4 neighbours, head1 ~dead (analyze_step R3).
C14. Error breakdown: oracle gather (R4) has 0 wave and 0 pointer errors, 365/2,648 (14%) wrong path steps; relbias (R3) 0 wave, 28/9,859 pointer, 2,064/2,648 path; learned pos (R1) 487/9,859 wave, 4,426 pointer, 2,648/2,648 path (results/errors*.json, errors.py output).

## The written solution is a needle
C15. Written attention: logit gap between chosen key and runner-up = 100 (min and median), max |logit| 42,000 on 15x15, max softmax derivative 3.7e-44 (diag.json attn).
C16. Scaling Wq down: solved 100% down to gap 5, 0% at gap 3 (diag2.json).
C17. Relative weight noise on all weights: 1e-4 -> 100%; 1e-3 -> 35-100% (seeds); 3e-3 -> 2.5-5%; 1e-2 -> 0% (diag.json basin). By tensor group at 1e-3 only the position tables hurt (59%); others fine up to 1e-2 (diag2.json).
C18. One wrong distance-mod-3 bit at loop 6 -> 45% solved; one wrong pointer -> 100%; one stray path bit -> 0%; Gaussian state noise sigma 0.01 -> 100%, 0.05 -> 0% (diag3.json).
C19. Written weights: E_row/E_col up to 16,129 (=127^2) vs init std 0.02; Wq up to 565.7; 19,936 parameters (diag.json scale).

## Exactness and stability
C20. Oracle gather + changing-cell weighting (R5): 100% of changing cells right in one step, but free rollouts 0%; the block drifts the 7 unsupervised static numbers by 0.5 per loop -> 1e3 by loop 10, 1e31 by loop 60 (R5_trace.json).
C21. Restoring static+scratch numbers and rounding the search numbers after every loop (oracle discretizer), starting from the true loop-0 state, makes R5 solve 95/100 at 15x15, 89/100 at 21x21 and 28/30 (93%) at 31x31 (rollout_check.py prints counts; n=30 at 31x31). Without it: 0 at every size. Caveat: rows/cols >= 15 use position-table entries never trained.
C22. Adding keep-losses + input noise (R8) does not stop free-run drift: static dev 0.42 at loop 1, 2.9 at 10, 3e10 at 60; free 0%, snapped 56% at 15, 0% at 21 (R8_trace.json, rollout_check).
C23. Training on the block's own 8-loop unrolls (R11, warm start from R5, keep-losses, noise 0.05, 3,000 steps): one-step 100% on changing cells; snapped rollouts 96% at 15x15 and 72.5% at 21x21; free rollouts 1/100 at 15x15 (the only free success), 0/40 at 21x21 (logs/R11.log).

## Audit corrections (verification workflow wf_68137be6-344, adopted)
- C4: diag.json stores every loop (200 values), not every 5th. The x6-per-5-loops figure holds after loop 5 (first 5 loops ~x11). Diagnostic init is seed 1 in float64. A1's non-finite loss is float32 overflow of the squared state near loop 105 of 106.
- C7: B1's changing-cell and attention numbers are in results/analyze_step_written_B1.json (regenerated), not analyze_step.json. 99.06% is the copy baseline under train.py's per-chunk onestep metric; analyze_step's metric gives 99.28%.
- C10: REFUTED in part. No log backs '100% at lr 3e-3 without CE' (that run printed to stdout only), so the claim was dropped. Numbers are training-batch accuracies.
- C20: R5 has only 3% of mazes exact at every teacher-forced step, even though 100% of changing cells are right.
- C23: R11's 1/100 free success is by path-flag state; 0/100 by readout; 0/100 at 2T loops.
- Critique adopted: one-step failure is sufficient, not shown to be dominant; routing is a slow start, not a deadlock (R10 reaches 99.5%); the written-coords arm may be a scale artifact; working rollouts rely on hard-coded routing plus a reset/rounding oracle; the answer-loss effect may be loss balancing; single seed throughout.
- R10 (learned tables, keep losses + noise, 15k steps): 99.49% changing cells, 0% solved free and rounded (logs/R10.log).

## Part 2: discretizer inside the loop (train_disc.py)
Discretizer D: round search dims 7..14 to {0,1}, reset static 0..6 and scratch 15..34 to the embedding h0, after every loop (and at loop 0).
Losses on pre-rounding values; changing cells weighted; readout trained on detached state. pos none = E_row/E_col zero and frozen.
C24. Straight-through gradient through the rounding destabilizes training in all 3 settings tried:
     D1 (full BPTT, gather, pos none, lr 2e-3): non-finite loss at step 175 (logs/D1.log 'diverged 175'; no json written).
     D1b (BPTT truncated to last 16 loops, gather): 358 of the first 500 steps had a non-finite gradient norm and were skipped; loss ~90-94 at steps 300-500; 0% solved at step 500 (results/D1b_bptt16_gather.json).
     D4 (8-loop unrolls from true states, relbias, pos none): loss 0.2 at step 1050 -> 7.7e5 at step 2050 and after; 0% solved at steps 1000-3000 (results/D4_unroll_relbias.json, logs/D4.log). D4 started before the skip patch.
     D2 / D3 (full BPTT relbias / learned) were stopped before their first eval (step 500): no evidence either way.
C25. eval_disc.py reproduces R5 + rounding: 95/100 solved by path-flag state at 15x15 (matches rollout_check snapall), 20/100 by readout, 3/100 exact at every loop.
C26. Gradient stopped at the rounding (--no_st), 8-loop unrolls from true states at a random loop, lr 2e-3, seed 0 (logs/E*.log, results/E*.json; eval = free rollout of the model incl. discretizer from its own loop-0 state; 100 mazes at 15 and 21; 'state/readout/every-loop-exact'):
     E1 gather, no positions, 5k steps: 15x15 0.97/0.97/0.73 (same at 2T); 21x21 0.94/0.94/0.76; peak at step 2000: 1.00/1.00/0.98 at 15, 21 and 2T. 31x31 (eval_disc.py, 30 mazes): 28/28/22 of 30.
     E2 relbias, no positions, 8k steps: 15x15 0.82/0.82/0.58 (same at 2T); 21x21 0.79/0.79/0.58; 31x31: 1/1/0 of 30.
       probe_disc (50 mazes, 15x15, teacher-forced): 0 wrong changing cells in all fields; 1,027 false path flags and 62 false pointers among ~1.2M staying cells; 31/50 rollouts never wrong.
       leak.py: attention mass outside the 4 neighbours+self grows with size: head0 0.26/0.38/0.55, head2 0.42/0.53/0.67 at 15/21/31; heads 1,3 0.87-0.99 (results/leak_E2.json). Learned relative biases: neighbours ~+2..+4, far -2.5 (head 0).
     E3 softmax + learned position tables (the paper's architecture), 10k steps: 15x15 0.43/0.46/0.00 (same at 2T); 21x21 0/0/0 (rows/cols >= 15 untrained). 0 at steps 2k-8k. Probe at 6k: wave 32/4,929 changing wrong + 25,243 false; pointer 1,169/4,929; path 226/1,322.
     E4 relbias, rollouts from the model's own loop-0 state (no true-state starts): 0% at steps 500, 1000 and 1500 (stopped there, ~66 min).
     E3b = E3 continued from its step-10k weights, lr 1e-3, 10k more steps: 15x15 0.65 (2k), 0.66 (4k), 0.62 (6k), 0.73 (8k), 0.73/0.73/0.00 at 10k (same at 2T); 21x21 0/0/0 (logs/E3b.log).

## Part 3: pushing further (train_disc.py options --local, --sizes, --offset_aug, --focus)
C27. F3 (paper architecture warm-started from E3 at step 10k, random coordinate shifts in the 128x128 frame, lr 1e-3): 0% at 15x15 at steps 2k, 4k, 6k, 8k (logs/F3.log; stopped at 8k). Probe at 6k: pointer 4,922/4,929 changing cells wrong, wave 2,514/4,929.
C28. fails.py on F1 at step 5k (results/fails.json): of 20 failing 15x15 mazes, 19 first go wrong at loop dS+1, on the path flag: 15 at the cell one step after S (path_idx 1, flag not set) and 4 at another cell (a flag set off the path). The 20th (maze 22) first goes wrong at loop dS+15, a stray path flag off the path. E2, F1, F2 all plateau at exactly 0.80 at 15x15 (logs).
C29. Rare-loop oversampling (--focus 0.5, fine-tunes of 3k steps, lr 1e-3): G1 (from F1, local) 0.80/0.80/0.79 at 15, 0.69 at 21; G2 (from E2) 0.85/0.85/0.85 at 15, 0.80/0.80/0.80 at 21; G3 (from E3b, paper arch) 0.12 state / 0.79 readout at 15 (0.50/0.79 at 1k, 0.66/0.79 at 2k), 0 at 21 (logs/G*.log).
C30. Local window: F1 31x31 = 19/19/6 of 30 (eval_disc.py), vs E2 (no window) 1/30. F1 final 0.80 at 15, 0.69 at 21; F2 (window + mixed sizes 11/15/21) identical: 0.80 / 0.69 (logs/F1.log, F2.log).
C31. Ties: G1's 20 failing 15x15 test mazes all have >=1 tie on the path (a path cell with 2+ neighbours one step closer to the goal); 0 of the 80 solved mazes do; in 20/20 the first error is a path flag on the losing branch of a tie (results/fails_G1.json + tie check). 206/1000 training mazes have a tie on the path.
C32. Summed margin penalty on near-threshold bits (--hinge 1) + rare-loop focus 0.5, lr 5e-4, 3k-step fine-tunes:
     H1 (from G1: relbias + local window, no positions): 15x15 1.00/1.00/0.99 at 1k; 1.00/1.00/1.00 at 2k, also at 2T and at 21x21 (1.00/1.00/1.00). 31x31 on the step-2k snapshot (results/H1_step2000.pt): 30/30/30 of 30 (eval_disc.py).
     H2 (from G2: relbias, no window): 0.97/0.97/0.97 at 1k; 0.94/0.94/0.93 at 2k, 21x21 0.89 (logs/H2.log).
     H3 (from G3: paper arch, learned tables): readout 0.54 (1k), 0.72 (2k), 0.91 (3k, same at 2T); state 0.00 throughout; 21x21 0 (logs/H3.log).
     I2 (from G2, tie oversampling 0.6 + focus 0.2, no hinge): 0.86 at 1k (logs/I2.log).
C33. H1 step-2k snapshot at 61x61 (10 mazes, seed 97, eval_disc.py chunk 1): 10/10 solved by state and readout, 10/10 exact at every loop. H1 final (3k): 1.00/1.00/1.00 at 15, 2T and 21 (logs/H1.log). H2 final (3k): 0.99 at 15 and 2T, 0.91 at 21. I2 final (3k): 0.87 at 15, 0.82 at 21.

## Audit corrections, part 2 (workflow wf_86572982-b50, adopted)
- C24: D4's loss held at 0.20-0.25 to step 1500, then 1.2 (1550), 6.7e5 (1600), ~7.7e5 after. D1's config is not recorded (log only). Only D4 has a matched no-ST control (E2). The real contrast is gradient across loops vs. none.
- C26: E3 readout at 8k is 0.01. The E3 probe at 6k is unverifiable (no checkpoint survives).
- C27: the 6k F3 probe is not reproducible; the 8k probe gives pointer 4,923/4,929, wave 2,989/4,929, path 452/1,322 changing cells wrong.
- C28: results/fails_F1.json (exact F1 checkpoint unknown). The path_idx-1 errors set the flag one loop EARLY (model 1, target 0). E2 reaches 0.81-0.82 after 6k; only F1/F2 sit at exactly 0.80.
- C31: the training set used (seed 1015) has 169/1000 mazes with a tie on the path, not 206 (206 = seed 1000). Test tie counts: 20/100 (15), 31/100 (21), 11/30 (31) = exactly the windowed models' failures.
- C32: I2 used lr 1e-3, hinge 0.
- New, verified by me: H1 final on fresh mazes: 100/100 at 15x15 (seed 123) and 50/50 at 21x21 (seed 124), all exact at every loop; H1 final at 4T on the 15x15 test: 100/100; H2 final at 31x31: 1/30 (results/eval_disc.jsonl).
- H3 state 0% is mostly the goal-cell path flag never being set (critic's re-run on 20 mazes: only wrong cell in 14/20).
- Verified by me: H3 final, 100 test mazes: goal-cell path flag wrong in 100/100; the only wrong path flag in 91/100 (= its 91% readout).

## Part 4: from scratch, seeds and ablations (train_disc.py --cosine, 8k steps, lr 2e-3 cosine to 1e-4, 1 thread each)
Full recipe = --mode unroll --k 8 --no_st --attn relbias --pos none --local --focus 0.5 --hinge 1. Seeds change both init and the training mazes (seed 1000+seed+15).
C34. S0, S1, S2 (seeds 0,1,2): at step 8k, 1.00/1.00/1.00 at 15x15, 2T and 21x21 (logs/S*.log); at step 4k already 1.00 at 15, 2T, 21 for all three; at step 2k 0.98/0.96/0.95 at 15.
     31x31 (30 mazes, seed 97): 30/30/30 for each seed. Fresh 15x15 (100 mazes, seed 123): 100/100/100 for each (results/eval_disc.jsonl).
C35. Ablations, seed 0, same schedule: no margin (A_nohinge): 0.80 at 15 and 2T, 0.69 at 21, 19/30 at 31 = the tie rates of each test set. No first-step oversampling (A_nofocus): 1.00 at 15, 2T, 21; 30/30 at 31. No local window (A_nolocal): 1.00 at 15 and 2T, 0.89 state / 0.93 readout at 21, 0/30 at 31.
C36. 61x61 (10 mazes, seed 97, final checkpoints): S0, S1, S2 each 10/10/10 (results/eval_disc.jsonl).

## Audit corrections, part 4 (workflow wf_faf9dbf5-eea, adopted)
- C35 wording: A_nohinge fails exactly the tie mazes (20/100, 31/100, 11/30); its score equals the share of tie-free mazes, not the tie rate.
- Ablations are single-seed (seed 0); A_nofocus lagged at 2k (0.93 solved, 0.62 every-loop exact vs 0.98 for S0).
- Auditor's extra checks: S1 on 30 fresh 21x21 mazes (seed 555): 30/30/30; every loop up to 2T at 31x31 for S0-S2: no change after completion; min pre-rounding margin on 31x31 rollouts 0.115-0.147; seed 123 has 23/100 tie mazes; 0 train/test maze overlaps.
