# Third-party notices

## mbt-gym

Parts of the `cvar_mm` package are adapted from mbt-gym (Jerome et al., 2023, "Mbt-gym: Reinforcement learning for
model-based limit order book trading", ICAIF '23), <https://github.com/JJJerome/mbt_gym>, commit 1e1aa38
(1e1aa38ca35c4fa472777e3574f9c907f89bb5b0):

- `cvar_mm/gym/index_names.py`: the six state and action index constants;
- `cvar_mm/rewards/reward_funcs.py`: the parameters and assignments of `CjMmCriterion`;
- `cvar_mm/gym/sb3_wrapper.py`: the `num_trajectories` and `n_steps` properties of
  `StableBaselinesTradingEnvironment`.

Each of these three files starts with a one-line notice that points here. The package also follows mbt-gym's
architecture and interfaces: the agent, model-dynamics, trading-environment and reward-function classes, the arrival,
fill-probability and midprice models, and the Stable Baselines 3 wrapper.

mbt-gym is distributed under the BSD 3-Clause License, reproduced here as it appears in mbt-gym's `LICENSE`:

```text
BSD 3-Clause License

Copyright (c) 2018, Joseph Jerome
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Hummingbot

The forward test (`hummingbot_adapter/`) runs on Hummingbot 2.17.0 (<https://github.com/hummingbot/hummingbot>,
commit 9af100d), which is distributed under the Apache License 2.0. Hummingbot is a runtime dependency, built locally
into a Docker image. No Hummingbot code is included in this repository.
