# Synthetic Injection Methods for NILM

This note explains the three experiment methods in detail.

The goal is to start from synthetic appliance power and create NILM training samples.

A NILM training sample needs:

```text
X = aggregate smart meter window
y = target appliance window
```

But the generative model gives only:

```text
y_s = synthetic appliance power
```

So we need a way to construct a matching aggregate input.

---

## 0. Common Setup

### Real Training Data

Assume the real training data has `N_r` windows:

```text
D_r = {(X_r^1, y_r^1), (X_r^2, y_r^2), ..., (X_r^Nr, y_r^Nr)}
```

where:

```text
X_r = real aggregate smart meter window
y_r = real target appliance window
```

Example with 4 real windows:

```text
D_r = [R1, R2, R3, R4]
```

### Synthetic Appliance Data

The synthetic data is an appliance signal with an ON/OFF label:

```text
y_s(t), state_s(t)
```

where:

```text
state_s(t) = 1 means appliance ON
state_s(t) = 0 means appliance OFF
```

We cut it into windows of the same length as the real data.

If the real model uses 512 timesteps per window, then every synthetic sample must also use:

```text
window length = 512
```

So the synthetic windows are:

```text
S1, S2, S3, ..., Sm
```

Each synthetic window contains:

```text
y_s window
state_s window
```

A synthetic window is called an ON-event window if:

```text
any timestep in state_s window = 1
```

Mathematically:

```text
S_i is ON-event if max(state_s^i) = 1
```

If:

```text
max(state_s^i) = 0
```

then the window is OFF-only.

---

## 1. Aggregate Construction Used by All Methods

For every synthetic appliance window, we construct its aggregate using a real background window.

Formula:

```text
X_s = X_r - y_r + y_s
```

Meaning:

```text
real background = X_r - y_r
synthetic aggregate = real background + synthetic appliance
```

Example:

```text
X_r = [300, 310, 320, 315, 300]
y_r = [0,   0,   100, 100, 0]
```

Remove the real target appliance:

```text
background = X_r - y_r
           = [300, 310, 220, 215, 300]
```

Synthetic appliance:

```text
y_s = [0, 0, 500, 700, 0]
```

Constructed aggregate:

```text
X_s = background + y_s
    = [300, 310, 720, 915, 300]
```

Final synthetic training pair:

```text
(X_s, y_s)
```

This aggregate construction is fixed for D1, D2, and D3.

---

## 2. Injection Ratio

The injection ratio controls how many synthetic samples are added.

Let:

```text
N_r = number of real training windows
rho = injection ratio
N_s = number of synthetic samples to add
```

Formula:

```text
N_s = rho x N_r
```

Examples:

```text
N_r = 1000 real windows
```

Then:

```text
rho = 25%  -> N_s = 250 synthetic samples
rho = 50%  -> N_s = 500 synthetic samples
rho = 100% -> N_s = 1000 synthetic samples
rho = 200% -> N_s = 2000 synthetic samples
```

For implementation:

```text
N_s = round(rho x N_r)
```

---

## 3. D1: Full-Window Append

### Research Question

```text
What happens if we simply add the generated synthetic windows to the real training set?
```

This is the naive baseline.

### What Data It Uses

D1 uses the full synthetic window pool:

```text
Full pool = all synthetic windows
```

This includes:

```text
ON-event windows
OFF-only windows
weak windows
natural generated sequence windows
```

It does not filter by event.

### Algorithm

Given injection ratio `rho`:

1. Compute total synthetic samples:

```text
N_s = rho x N_r
```

2. Randomly sample `N_s` windows from the full synthetic window pool:

```text
S_full = sample(full_pool, N_s)
```

3. For each selected synthetic window, choose one real background window.

4. Construct:

```text
X_s = X_r - y_r + y_s
```

5. Append all constructed synthetic windows behind the real windows:

```text
D1 = [D_r, D_full]
```

### Example

Real windows:

```text
D_r = [R1, R2, R3, R4]
```

Synthetic full windows:

```text
Full pool = [S1_off, S2_on, S3_off, S4_on]
```

If `rho = 50%` and `N_r = 4`:

```text
N_s = 0.5 x 4 = 2
```

Sample:

```text
D_full = [S1_off, S4_on]
```

Final training order:

```text
D1 = [R1, R2, R3, R4, S1_off, S4_on]
```

### What It Tests

D1 tests whether synthetic data helps when it is used in the simplest way.

### Strength

```text
Simple and preserves the synthetic ON/OFF distribution.
```

### Risk

```text
If synthetic data is appended behind the real data, synthetic windows may form a block at the end.
If many synthetic windows are OFF or weak, the model may not learn much from them.
```

---

## 4. D2: ON-Event Insertion

### Research Question

```text
Can we use only useful synthetic ON events and place them into real OFF/background periods?
```

D2 is event-focused.

### What Data It Uses

D2 uses only synthetic ON events.

First, detect ON/OFF in synthetic appliance data.

Example synthetic signal:

```text
y_s     = [0, 0, 0, 800, 2200, 2100, 0, 0]
state_s = [0, 0, 0, 1,   1,    1,    0, 0]
```

The ON event is:

```text
[800, 2200, 2100]
```

So D2 extracts the event segment, not necessarily the whole 512-step window.

### Real OFF Background Requirement

We should insert the synthetic ON event only into real periods where the target appliance is OFF.

For real data:

```text
state_r(t) = 0 means target appliance OFF
state_r(t) = 1 means target appliance ON
```

A real segment can be used as background if:

```text
state_r(t) = 0 for all timesteps in the insertion interval
```

Mathematically, if synthetic event length is `L_event`, and candidate real start index is `a`, then the real gap is valid if:

```text
sum(state_r[a : a + L_event]) = 0
```

This means the target appliance is OFF during the whole inserted event.

### Algorithm

Given injection ratio `rho`:

1. Compute total synthetic samples:

```text
N_s = rho x N_r
```

2. Detect synthetic ON events:

```text
E = {e_1, e_2, ..., e_k}
```

Each event has a length:

```text
L_event
```

3. Detect real OFF gaps from real `on_off` labels.

A gap is a continuous segment where:

```text
state_r = 0
```

4. For each synthetic event:

- choose a real OFF gap long enough for the event
- choose an insertion position inside that gap
- insert the synthetic event into the real background

5. Construct the new aggregate during the event period:

```text
X_event = X_r - y_r + y_event
```

Because the real gap is OFF for the target appliance, usually:

```text
y_r = 0
```

So the construction becomes:

```text
X_event = X_r + y_event
```

6. After insertion, cut the constructed sequence into training windows.

### Example

Real background segment:

```text
X_r     = [300, 305, 310, 300, 295, 300, 310, 305]
y_r     = [0,   0,   0,   0,   0,   0,   0,   0]
state_r = [0,   0,   0,   0,   0,   0,   0,   0]
```

Synthetic ON event:

```text
y_event = [800, 2200, 2100]
```

Choose insertion start at index 2:

```text
before insertion:
X_r = [300, 305, 310, 300, 295, 300, 310, 305]
```

Inserted appliance event:

```text
y_inserted = [0, 0, 800, 2200, 2100, 0, 0, 0]
```

New aggregate:

```text
X_s = X_r + y_inserted
    = [300, 305, 1110, 2500, 2395, 300, 310, 305]
```

New target appliance label:

```text
y_s = y_inserted
    = [0, 0, 800, 2200, 2100, 0, 0, 0]
```

This creates one event-inserted training region.

### How to Make Events Evenly Distributed

If we insert all events into one part of the real sequence, the training data becomes unrealistic.

So we distribute insertion locations across the full real training timeline.

Suppose we need:

```text
N_s = 4 event-inserted samples
```

Real timeline length:

```text
T = 10000 timesteps
```

Divide the timeline into 4 bins:

```text
Bin 1: 0 to 2499
Bin 2: 2500 to 4999
Bin 3: 5000 to 7499
Bin 4: 7500 to 9999
```

Insert one event into each bin, using a valid OFF gap inside that bin.

Mathematically:

```text
bin_width = T / N_s
bin_i = [i x bin_width, (i+1) x bin_width)
```

For each bin, choose a valid OFF insertion start:

```text
a_i in bin_i
sum(state_r[a_i : a_i + L_event]) = 0
```

This ensures synthetic events are spread across the real sequence.

### What It Tests

D2 tests whether explicit event insertion improves rare appliance learning.

### Strength

```text
Adds useful ON events and avoids wasting samples on OFF-only windows.
```

### Risk

```text
May create too many ON events compared with the real distribution.
The model may over-predict appliance activity.
```

---

## 5. D3: Balanced Event Insertion

### Research Question

```text
Can we add useful ON events while still keeping some natural synthetic distribution?
```

D3 is the compromise between D1 and D2.

### What Balanced Means

Balanced means the synthetic portion is split into two parts:

```text
50% event-inserted samples
50% full-window synthetic samples
```

Let:

```text
N_s = total synthetic samples to add
N_event = number of event-inserted samples
N_full = number of full-window samples
```

Then:

```text
N_event = floor(N_s / 2)
N_full = N_s - N_event
```

and:

```text
N_s = N_event + N_full
```

### Example With Percentage Ratio

If:

```text
N_r = 1000 real training windows
rho = 100%
```

then:

```text
N_s = rho x N_r
    = 1.0 x 1000
    = 1000 synthetic samples
```

Balanced split:

```text
N_event = floor(1000 / 2) = 500
N_full = 1000 - 500 = 500
```

So D3 adds:

```text
500 event-inserted samples
500 full-window samples
```

Final dataset:

```text
D3 = D_r + D_event + D_full
```

### Algorithm

Given injection ratio `rho`:

1. Compute:

```text
N_s = rho x N_r
```

2. Split:

```text
N_event = floor(N_s / 2)
N_full = N_s - N_event
```

3. Create `N_event` event-inserted samples using D2.

4. Create `N_full` full-window samples using D1.

5. Combine them with the real training data.

### Example

Real windows:

```text
D_r = [R1, R2, R3, R4]
```

Assume:

```text
rho = 100%
N_r = 4
N_s = 4
```

Balanced:

```text
N_event = 2
N_full = 2
```

Event-inserted samples:

```text
D_event = [E1, E2]
```

Full-window samples:

```text
D_full = [F1, F2]
```

Final:

```text
D3 = [R1, R2, R3, R4, E1, E2, F1, F2]
```

If we want to avoid a block of event-inserted samples, we can order them evenly:

```text
D3_ordered = [R1, E1, R2, F1, R3, E2, R4, F2]
```

This order is only the training-window order. It is not a physical timestamp insertion unless we explicitly reconstruct the continuous timeline.

### How It Balances ON and OFF

D2 gives strong ON-event enrichment:

```text
many samples contain inserted events
```

D1 keeps full generated distribution:

```text
some samples are ON, some are OFF, some are weak
```

D3 combines both:

```text
half event-focused
half full-distribution
```

So the model sees:

```text
useful ON examples + more natural generated windows
```

### Strength

```text
Less aggressive than D2.
More useful than D1 if full synthetic data has many OFF windows.
```

### Risk

```text
The 50:50 split is a design choice. It may not be optimal for every appliance.
```

---

## 6. Summary Table

| Method | Synthetic source | Placement | Main purpose | Main risk |
|---|---|---|---|---|
| D1 Full-window append | Full synthetic windows | Append after real windows | Simple baseline, preserves generated distribution | May add many OFF or weak windows |
| D2 ON-event insertion | Synthetic ON events only | Insert into real OFF background | Enrich rare appliance events | May become too ON-heavy |
| D3 Balanced event insertion | 50% D2 + 50% D1 | Mix event-inserted and full-window samples | Compromise between usefulness and realism | 50:50 may not be optimal |

---

## 7. Suggested Paper Wording

Short version:

```text
D1 uses full synthetic appliance windows and appends them to the real training set. D2 extracts synthetic ON events and inserts them into real background periods where the target appliance is OFF. D3 is a balanced strategy: for a target injection ratio rho, half of the synthetic samples are event-inserted samples and the other half are full-window synthetic samples.
```

More technical version:

```text
For D3, the total number of synthetic samples is N_s = rho |D_r|. We set N_event = floor(N_s/2) and N_full = N_s - N_event. The event-inserted subset is generated by inserting synthetic ON events into real OFF-background intervals. The full-window subset is sampled from the full synthetic sequence. The final training set is D_r union D_event union D_full.
```
