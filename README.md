# CogDiagHMM
Evaluate HMM on neural timeseries data

Case 1. When the current input as well as the current state are driving the transitions into the next state together. This is of interest to us. An incomplete or incorrect specification of either of them would lead to a wrong transition. The observations should be able to reflect the current state clearly. If not, the recorded observations are not complete. This is the case where we are using known behavior to *understand* recorded neural activity.

Case 2. When just the current input is sufficient to determine the transition into the next state. The notion of state is then redundant here. One example is `BlockData`.

Case 3. When just the current state is sufficient to determine the transition into next state. It's a self-driven driven behavior we are not interested in this work. Examples could be randomly varying emotional or fatigue states.

