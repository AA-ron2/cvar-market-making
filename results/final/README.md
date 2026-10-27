# Result tables of the thesis

The files here come from three sources:

- `main/`, `profiling/` and `diagnostics/`: the thesis's corrected pipeline on SURF Snellius (the profile stage at commit fe5f9fe, 2026-10-14 to 2026-10-15; the final stage at commit cf3096d, 2026-10-15);
- `inference_size_study/`: the simulation study of the primary test's size, which uses no market data (committed to the private repository on 2026-10-12);
- `forward_test/`: the forward test's analysis of 2026-10-24 (code at the private commit ad68e0f).

`release_assets.sha256` is new: the SHA-256 of the eight release assets. Every other file is byte-identical to its source except the five whose original SHA-256 is given below, which differ in local paths only or (for the forward test) also in the network specifics of embedded RUN_LOG text ([REDACTIONS.md](REDACTIONS.md)). Every row is one simulated trading day or coarser (a day, a day × fill seed × rollout path, a training path, a model, a contrast), or one bot-day for the forward test. `SHA256SUMS` lists every file here; `python scripts/analysis/verify_release.py` checks them. The models are release assets ([README](../../README.md#trained-models)).

Thesis references give the number and short title of the item in the thesis PDF (Chapter 3 "Data and calibration", Chapter 4 "Methods", Chapter 6 "Results", Appendix A "Supplementary statistical results"); the forward test is Appendix B, "Hummingbot forward validation (paper-trading deployment test)".

## `main/`: Final stage: B0 and the evaluation on Jan 7–29

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `b0_gamma_fixed.txt` | B0's tuned risk aversion γ_B0 | Section 6.2 "Retrospective out-of-sample comparison (Jan 7–29)" (γ_B0) | `34cbb018dd15a9112f8c6f4f2cbb2874bea4efefbc1de3ad7c1895b6abe226c4` | (same) |
| `b0_test_episodes.csv` | B0's 920 full-day paths (day × fill seed × rollout) | Table 6.2 "Retrospective out-of-sample results" (CVaR over 920 paths); Table 6.3 "Primary inference on the tail of daily drawdown" (path level) | `b6369a50d8087302444d4fdb03c45c13ae2b8ef2262c7bd334d56ebd7a8551e9` | (same) |
| `b0_test_results.csv` | B0 on the 23 test days: the metrics per seed-averaged day | Table 6.2 "Retrospective out-of-sample results"; Figure 6.2 "Daily maximum drawdown on the test days"; Table A.3 "Daily maximum drawdown on the test days" | `f3b3be9b9b72681692a40c6cf18b7544f8a692770023e0658468c03e6f673d10` | (same) |
| `b0_train_calibration_params.csv` | B0's per-day calibration on Jan 1–6 and its γ search range | Section 4.2 "B0 and B1" (the per-day tuning); Section 6.2 "Retrospective out-of-sample comparison (Jan 7–29)" (the admissible range) | `2f04c52830d710753f1967284a101bc4ac04a7ad285097de38728d3d0f32e968` | (same) |
| `b1_test_episodes.csv` | B1's 920 full-day paths (day × fill seed × rollout) | Table 6.2 "Retrospective out-of-sample results" (CVaR over 920 paths); Table 6.3 "Primary inference on the tail of daily drawdown" (path level) | `6b1177cc6028c8033cc9a75f09c08e8ca042abbd7f46ec853e233a792198a50a` | (same) |
| `b1_test_results.csv` | B1 on the 23 test days: the metrics per seed-averaged day | Table 6.2 "Retrospective out-of-sample results"; Figure 6.2 "Daily maximum drawdown on the test days"; Table A.3 "Daily maximum drawdown on the test days" | `97e5e548b5498c2e8c7828b0f75233ca7a1af9624b0fc09eaa2a8e31a65589b9` | (same) |
| `b1plus_test_episodes.csv` | B1⁺'s 920 full-day paths (day × fill seed × rollout) | Table 6.2 "Retrospective out-of-sample results" (CVaR over 920 paths); Table 6.3 "Primary inference on the tail of daily drawdown" (path level) | `1acda4bc9aa93ade5e206bd14e57f7c15f8f04d770c7a1ad8c92c924cc88be06` | (same) |
| `b1plus_test_results.csv` | B1⁺ on the 23 test days: the metrics per seed-averaged day | Table 6.2 "Retrospective out-of-sample results"; Figure 6.2 "Daily maximum drawdown on the test days"; Table A.3 "Daily maximum drawdown on the test days" | `a3b6347d44be757c23a15ff809e66ca20ae7a2512dbc489caa7d5a64969be7d1` | (same) |
| `b2_test_episodes.csv` | B2's 920 full-day paths (day × fill seed × rollout) | Table 6.2 "Retrospective out-of-sample results" (CVaR over 920 paths); Table 6.3 "Primary inference on the tail of daily drawdown" (path level) | `57df4d5751bc9c854121d41768165f2d3b8466b32398b28c082846d02aa2fcf6` | (same) |
| `b2_test_results.csv` | B2 on the 23 test days: the metrics per seed-averaged day | Table 6.2 "Retrospective out-of-sample results"; Figure 6.2 "Daily maximum drawdown on the test days"; Table A.3 "Daily maximum drawdown on the test days" | `8f7f48fa984881869604990e4c7b6457929f1645a752a53fda014d6c17d97a13` | (same) |
| `b3_test_episodes.csv` | B3's 920 full-day paths (day × fill seed × rollout) | Table 6.2 "Retrospective out-of-sample results" (CVaR over 920 paths); Table 6.3 "Primary inference on the tail of daily drawdown" (path level) | `bca83c075a5a31ef71984d29666af2b3b902a5f3cca64bbbd6951c68bbc02a67` | (same) |
| `b3_test_results.csv` | B3 on the 23 test days: the metrics per seed-averaged day | Table 6.2 "Retrospective out-of-sample results"; Figure 6.2 "Daily maximum drawdown on the test days"; Table A.3 "Daily maximum drawdown on the test days" | `0600c7d6fecac9a854f13d1c601ee0746306f153c5eb129c340d0a8352ef64cb` | (same) |
| `comparison_table.csv` | Per metric, the number of test days on which each of B0–B3 was best | Table 6.2 "Retrospective out-of-sample results" (days best) | `0e18ececbf126ff1121152eaea8c63f18d2095ce06dbddb997b67d61afbd0888` | (same) |
| `overall_summary_table.csv` | Per model, over the 23 test days: means and medians of the seven metrics, and the CVaR₀.₂ of the daily MaxDD over the days and over all 920 paths | Table 6.2 "Retrospective out-of-sample results" | `ecf0e93618013bc0d6d022e5f29d90754c1d8adf9b68c3b4f5c3057a34382b0e` | (same) |
| `paired_day_inference.csv` | The secondary family (paired day differences of seven metrics, Holm over 28) and B1⁺'s descriptive rows | Table A.2 "Secondary family of paired tests"; Figure 6.2 "Daily maximum drawdown on the test days" (b: the Hodges–Lehmann estimates) | `e4436a480b087136f22ca3af380af8c9c68d0ac23085046b7f8967bd470c448b` | (same) |
| `paired_day_inference_primary.csv` | The primary contrasts (B3 vs B1, B3 vs B2) on the CVaR₀.₂ of daily MaxDD, with the sensitivity and descriptive rows | Table 6.3 "Primary inference on the tail of daily drawdown" | `a8afc6ad3e41e15d5450fa9be285d644781c9aebc991facf6abd3c7aace68e05` | (same) |

## `profiling/`: Profile stage: calibration, selections, d, B3's training and the profile record

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `b2_alpha0.1_validation_results.csv` | The B2 candidate α = 0.1 on the validation days Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (B2 selection) | `367a30584bd6f77f17f0ca1caed46763f84b0ac601c1d89d39b15e967c4bce10` | (same) |
| `b2_alpha1.0_validation_results.csv` | The B2 candidate α = 1.0 on the validation days Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (B2 selection) | `6f5c5017d4c9bf7b195f9ad4586ff1e8002c0d17641858d692233c59a7d18e59` | (same) |
| `b2_alpha10.0_validation_results.csv` | The B2 candidate α = 10.0 on the validation days Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (B2 selection) | `3a4a31da3aa56adc0bfbaf51e935dc116e48bc947256bd26f725cbad0502f572` | (same) |
| `b2_alpha_sweep_summary.csv` | The B2 candidates' Jan 5–6 validation means per α | Section 6.1 "Training and selection (Jan 1–6)" (B2 selection) | `4ad8436d135de42daec6ef617d0cf934912217e2ad8d91b0a18e7fd1007d3405` | (same) |
| `b2_best_alpha.txt` | The selected α | Section 6.1 "Training and selection (Jan 1–6)" (B2 selection) | `d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6` | (same) |
| `b2_promotion.json` | The promotion of the selected α's final B2, with the SHA-256 of its files | provenance | `839aacfa33f3b1e1ba9430d812ee8f0d5638bc016704df885f396bdd1ddf98fe` | (same) |
| `b3_cvar_lagrangian_history.csv` | B3's ten updates: λ, ν, Ĥ, Ĥ − d and the per-update diagnostics | Table 6.1 "Movement of the fine-tunes away from B1" (B3's row); Figure 6.1 "B3's constraint statistic and multiplier during training" (η = 8) | `abf78774777b938c1af63e7f45b6f2fa618ea993fd0cf4e77f9a7307c3763d95` | (same) |
| `b3_dual/b3_eta0.5_cvar_lagrangian_history.csv` | The η = 0.5 run's B3 history (the same updates in the B3 history's columns) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `4dd36037f1c812a3224d5da54a66ccebe31377c2acb7e2fca74cb8a63911b6cf` | (same) |
| `b3_dual/b3_eta0.5_training_costs.csv` | The η = 0.5 run's training costs per update, day and path | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `2cd199029d8d66d935462983a7c75570791cf0e25499bc3ec8d359763c9bc154` | (same) |
| `b3_dual/b3_eta0.5_training_feasibility.json` | The η = 0.5 run's training feasibility | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `f70f5ad9d9473954df7594aa16fea35fde89d7220298259cab5f532ae1b26eb6` | (same) |
| `b3_dual/b3_eta2.0_cvar_lagrangian_history.csv` | The η = 2.0 run's B3 history (the same updates in the B3 history's columns) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `2942b32fe60a51d6c03c8b25a95e723f7984ab9d1bf1b07d9d90c8b084ed4f3b` | (same) |
| `b3_dual/b3_eta2.0_training_costs.csv` | The η = 2.0 run's training costs per update, day and path | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `2fbcca30448b3e95128ddff5493cef9e1740854145bc17172ef059e2c0f0ef34` | (same) |
| `b3_dual/b3_eta2.0_training_feasibility.json` | The η = 2.0 run's training feasibility | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `4de05a020f504df6535add2e51af390016a10fd09ffef434a460c36ad513fcae` | (same) |
| `b3_dual/b3_eta8.0_cvar_lagrangian_history.csv` | The η = 8.0 run's B3 history (the same updates in the B3 history's columns) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `abf78774777b938c1af63e7f45b6f2fa618ea993fd0cf4e77f9a7307c3763d95` | (same) |
| `b3_dual/b3_eta8.0_training_costs.csv` | The η = 8.0 run's training costs per update, day and path | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `49efda7913b941895f9d7ee1ce1211209a37550b0fdc93327fe9a32de8619463` | (same) |
| `b3_dual/b3_eta8.0_training_feasibility.json` | The η = 8.0 run's training feasibility | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `24d355b683b0da54464e7b3878faad6b0e383a7d4e24957401ad2396955346ba` | (same) |
| `b3_dual/history_eta0.5.csv` | The η = 0.5 run's ten updates (λ, ν, Ĥ, diagnostics) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `80aa99fa217bc1f98561a8006652283ae760d2677d9d5a58d4f167ee0af33c2c` | (same) |
| `b3_dual/history_eta2.0.csv` | The η = 2.0 run's ten updates (λ, ν, Ĥ, diagnostics) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `6b034df3076f2efb31f21c52341e40e63e2b06452cf293655b89f55af5690de1` | (same) |
| `b3_dual/history_eta8.0.csv` | The η = 8.0 run's ten updates (λ, ν, Ĥ, diagnostics) | Figure 6.1 "B3's constraint statistic and multiplier during training" | `a0f876cd673b48c68a08dc224cb32eaff3f1e903f60e9c942cf29b3f60444780` | (same) |
| `b3_dual/inputs.sha256` | The inputs of the selected η runs (B1, its normalizer, d, the calibration, the manifest) | provenance | `089f44081b4d43c766e8b5cf173a629d8b3941733ed54e82b2ea2902e7c0aaee` | (same) |
| `b3_dual/inputs_eta0.5.sha256` | The inputs the η = 0.5 task hashed at its start | provenance | `089f44081b4d43c766e8b5cf173a629d8b3941733ed54e82b2ea2902e7c0aaee` | (same) |
| `b3_dual/inputs_eta2.0.sha256` | The inputs the η = 2.0 task hashed at its start | provenance | `089f44081b4d43c766e8b5cf173a629d8b3941733ed54e82b2ea2902e7c0aaee` | (same) |
| `b3_dual/inputs_eta8.0.sha256` | The inputs the η = 8.0 task hashed at its start | provenance | `089f44081b4d43c766e8b5cf173a629d8b3941733ed54e82b2ea2902e7c0aaee` | (same) |
| `b3_dual/selected_lambda_lr.txt` | The selected η | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `510fcef8619a24e3ce4d8838394e3c00a262aa18419e4e723bfccf0ce6debaf8` | (same) |
| `b3_dual/selection.json` | The η selection (D-12a): the grid, the criteria and exclusions, the promoted B3's hashes | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `a33188a5a60ca5f198fa9cf0408cd0f76e81e362e642f59c75d14176519f779a` | `1fbc81728e8d6277e3c9be47de6c0117442707fcddb1c07c1f7a6e924f5320b9` |
| `b3_dual/summary.md` | The η page of the go/no-go review | Section 6.1 "Training and selection (Jan 1–6)" (η selection) | `069fd6633a3eb6810f834a6953064332c967e4434524bc02ffbdb5626d6e170f` | (same) |
| `b3_training_costs.csv` | B3's training costs per update, day and fill path | Section 6.1 "Training and selection (Jan 1–6)" (B3's feasibility) | `49efda7913b941895f9d7ee1ce1211209a37550b0fdc93327fe9a32de8619463` | (same) |
| `b3_training_feasibility.json` | B3's pooled CVaR₀.₂ over the last updates against d | Section 6.1 "Training and selection (Jan 1–6)" (B3's feasibility) | `24d355b683b0da54464e7b3878faad6b0e383a7d4e24957401ad2396955346ba` | (same) |
| `cvar_threshold_b3.json` | The estimator of d: both CVaRs, B1's σ, the pinned share, the seeds and every path's Z | Section 6.1 "Training and selection (Jan 1–6)" (d) | `a81e11016273714c059eddccab9e0c5da3d7942989daf9783d5e11d9d798ddf7` | `e5049aa49cef0f0d47151983ec4eaeea8c54636df9244f3e17aa25b76b36cbed` |
| `cvar_threshold_d.txt` | B3's threshold d, the stochastic CVaR₀.₂ and its ν | Section 6.1 "Training and selection (Jan 1–6)" (d) | `3782266b883c58ed061bb0b3b30831e61ba44b922680e774b37c9c6cbc332e41` | (same) |
| `formula_manifest.csv` | The 29 input files: date, role in the split, file name and SHA-256 | Section 3.1 "Order-book data"; Figure 3.1 "Data split" | `bd0341079f4bcf8b98b4b4b02832fad5151c44e70b3ceb93cb0f47386454df4b` | `79cdc69b36e5c7b494274aa23ee2587246c5c3f83571a34878a2a10f1db28059` |
| `formula_train_params.csv` | σ, A and κ per day, Jan 1–6 | Section 3.2 "Calibration" | `27737a2aeba35fefe0b55c230381cee7474da2d52246aef94d720a7ddd37e6c2` | (same) |
| `formula_train_params.json` | The frozen calibration on Jan 1–6: σ, A, κ, the fit and its settings | Section 3.2 "Calibration" | `2d0b4a916bcf52e8641fac0a6e34f93f088166ee87c4b98b4d24975f1f6426f8` | (same) |
| `profile_stage.commit` | The profile stage's commit | provenance | `69144396395a3018678212cc0a224c2acee9658a14ec96ed483f03897ebd2f70` | (same) |
| `profile_stage.sha256` | The profile record: the fifteen files the final stage checked | provenance | `f0b870b532e6727ee7b7728ab6e186460131a9b10aa7de8bf925f3f7edd629f8` | (same) |
| `profile_stage.submitted` | The profile stage's commit and submission time | provenance | `22ab0195a16b44ee195a8650226fe60311e3e535f565038848e22869e451d1af` | (same) |

## `diagnostics/`: Diagnostics: SHAP and the feature ablation (profile stage)

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `ablation/base_plus_vol_validation_results.csv` | The feature set "base_plus_vol" on Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (feature ablation) | `da0282754c4921b5d492b18e5e4a94c1e360e0ab35a22f1e72b22cb6808726d9` | (same) |
| `ablation/base_state_validation_results.csv` | The feature set "base_state" on Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (feature ablation) | `ae8c2e00e845882c9b407c4e018d5297521cdb43765628b2813df17beb9eef42` | (same) |
| `ablation/full_validation_results.csv` | The feature set "full" on Jan 5–6, per day | Section 6.1 "Training and selection (Jan 1–6)" (feature ablation) | `11c41c0a133042015a7eadfbf107373bfea921effb99d6ed11facaa912378d62` | (same) |
| `b1_shap_feature_importance.csv` | KernelSHAP: B1's mean \|SHAP\| per action and feature, first 150 decisions of Jan 1 | Section 6.1 "Training and selection (Jan 1–6)" (SHAP) | `28f17cae74eaa5902a4a50df6ac062b722b2a5e9f74fd59b0534286896ee89fb` | (same) |
| `formula_feature_ablation_results.csv` | The feature ablation: Jan 5–6 means per feature set | Section 6.1 "Training and selection (Jan 1–6)" (feature ablation) | `83685bc4904472f47a4c3a49a555283346593bb93a54a617686e7ca6cd5d8c6f` | (same) |

## `inference_size_study/`: The size study of the primary test

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `size_study.csv` | The size study of the primary test: rejection rates and coverage per null design | Table A.1 "Size of the primary tests" | `fa75bb7c132d0eda92c3cb7f7ce6323961db393969690fe9473e898a42336a41` | (same) |
| `size_study_metadata.json` | The size study's settings and seeds | Table A.1 "Size of the primary tests" | `0764dac26e6f84008d3ba8d0f76a42b882df0443e9118b97dc3f1f1ca903fcdc` | (same) |
| `size_study_table.tex` | The size study as a booktabs table | Table A.1 "Size of the primary tests" | `7edbf88e64f206b0b39817fb6d4446454b36ce5eab06ad2e354e4b54b3cdb576` | (same) |

## `forward_test/`: The forward test (Hummingbot paper trading, 2026-10-22 and 2026-10-23)

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `appendix_table.tex` | The forward test's appendix table (booktabs; B0 and B3 per day) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" | `1924b58890c3e5febcd045204d9333cc874b04d6468e5ca6ed2c3e43ce9353ac` | (same) |
| `data_description.csv` | Per bot-day: the snapshot file's rows, the mid's range, σ and the one-tick spread share | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis | `af8edd2c0d3f47338ec4a5c5a1009eb71d8a2b1f5c7b22a2a3946d690f68a526` | (same) |
| `feature_shift.csv` | Per bot-day and feature: B3's normalized observation, mean z and the shares beyond 3 or clipped | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis | `9df155578b8250472daee206dc591ee59cf8874b881ff55eb56e8a3c50ea07a5` | (same) |
| `fig_forward_inventory.pdf` | The inventory of B0 and B3 on each prospective day (PDF) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Figure B.1 "Inventory per bot and prospective day" | `a8131bdfd61ebf6833aaae7af1914810c070a02ee5c5e915422f77b0b731b693` | (same) |
| `fig_forward_inventory.png` | The same figure (PNG) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Figure B.1 "Inventory per bot and prospective day" | `ecb3b26cfc4868b04d7f49f31906d17577e533f67b8b8d30a7e52800305b8012` | (same) |
| `forward_test_tables.md` | Every table of the analysis in Markdown (paths as placeholders, its RUN_LOG section redacted) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 and the readings in the text come from these tables | `64028dc325cf793b7a02045569020f54d631a7641789360765125fb0ab86852d` | `2cec695866c84977d53efa19f9d57b99b26d5730453c4237c9f2addbe7249b89` |
| `pnl_split.csv` | Per bot-day: P_pre split into fill edge and inventory (exploratory) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": the text (the exploratory split of P_pre) | `e0663337571a314b49a36337f0008bf8d81423a0fcc22e7b7e6ef8d6c7e54715` | (same) |
| `qa_inventory.csv` | Q-A per bot-day: inventory, P&L and drawdown before the close | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (inventory, P_pre and MaxDD before the close) | `3e4fc2c3d45734f68802f3c63b27e3ef6ea3b6958d1caa5349ccde6cfc2d0140` | (same) |
| `qb_expected_fills.csv` | Q-B per bot-day: paper fills against the expected fills E | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (paper fills and E) | `19aa41e2d5e852567a0429e7e9e72a09881a0aedaf74d14c2e59e8b8403eda9a` | (same) |
| `qb_fill_classes.csv` | Q-B per bot-day and fill class (exploratory): edge, markouts, fills over E | Appendix B "Hummingbot forward validation (paper-trading deployment test)": the text (fills at the order's price; fills under a strict rule) | `393241d8c034f5351672a03fa06ccb5f16b32f5ae6ce5d4cc4098b9c94923d2a` | (same) |
| `qb_markouts.csv` | Q-B per bot-day: edge at posting and at the fill, markouts and retention | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (R(10 s)) and the text (edge and markouts) | `598b057fbada14d318d642d945f498be07777d30dc2c8b7cd4b8be28140c9582` | (same) |
| `qc_break_even.csv` | Q-C per bot-day: traded notional, break-even fees, paper fees and the close | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (the break-even fee before the close) | `813b7cbd2e0703f4ee034e4d50ba73c28f65916968ebaf94eca9173ffe571c13` | (same) |
| `qc_fee_curve.csv` | Q-C per bot-day and fee level: P&L net of a maker fee | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis | `fc4b1707a1cbf6538df85d487620f2f3152b45b4f2b338ff0ca5c012ea252703` | (same) |
| `qd_bot_counters.csv` | Q-D per bot-day: the recomputed counts beside the bot's own counters | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis | `83d37b2bc89aac43d9f78ee336fb685697782d744a415b4cfa58adabfec5556b` | (same) |
| `qd_exploratory.csv` | Q-D per bot-day (exploratory): trade fills at the price, book and clock events | Appendix B "Hummingbot forward validation (paper-trading deployment test)": the text (fills at the order's price) | `31963e9e2d99f5626150e5d45d1a6e9048aa4336d2f6182b0d643b7bed03ba00` | (same) |
| `qd_network_window.csv` | Q-D per bot-day (exploratory): clock checks and gaps inside and outside the mobile-hotspot window | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis (Figure B.1 shades the hotspot window) | `9dab9033ab305d475fc768bd31bb2175992d0d24b7f6df41281fe1119ac4c364` | (same) |
| `qd_operations.csv` | Q-D per bot-day: operation within the exchange rules | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (downtime) and the text (uptime, restarts, rejects) | `93925c0d086aeefcf2c63518db421a0ea0edfe3ee6657dc1d99eb1630b359553` | (same) |
| `results.json` | Every table, reading and input hash of the analysis in one file (paths as placeholders, its RUN_LOG text redacted) | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 and the readings in the text come from these tables | `36be1012695bf7f89252c23a530a66a20ae54f9e161e6fb3cf492fffe0c75c28` | `4aa75c7e2cbd701418f113fc3a756fddb23df4280304d5a036f4f4ff566fe4f9` |
| `s7_afterwards.csv` | §7 afterwards, per bot-day, limit and fee: when each loss limit would have triggered | Appendix B "Hummingbot forward validation (paper-trading deployment test)"; not shown in the thesis | `8d5120f574c62aea1cad55b4c57294a2c5b267a4be397282102a28aa5381c6a3` | (same) |
| `shadow_vs_paper.csv` | Per bot-day and metric: the shadow replay against paper trading, with the PIT | Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day" (the PIT rows) and the text (the shadow replay) | `829adfa072f1c2d2b86350babf53c6e600d535f9456017fe33462c003d226cbb` | (same) |

## Other files

| file | what it is | thesis | SHA-256 | original SHA-256 |
|---|---|---|---|---|
| `release_assets.sha256` | The SHA-256 of the eight release assets (the models and normalizers) | the trained models | `e75a2beae0109391e50587c24b7779e13c5efc3a3fa548c4beb511c07ffe10fa` | new |

Kept private: the job metadata of the run (`results/job_metadata/`), the per-day evaluation parts, the superseded profiling runs, every model other than the eight release assets, the logs, and the forward test's recorded market data and per-step inventory series.
