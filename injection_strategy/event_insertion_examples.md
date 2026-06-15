# Event Insertion Examples

This file shows what D2 ON-event insertion actually does.

For D2, the generated synthetic event becomes the target appliance signal `Y_s` inside a full training window. The real background is estimated as:

```text
background = X_s - Y_s
```

The final aggregate is:

```text
X_s = background + Y_s
```

So the table columns mean:

- `original background`: the household background before inserting the target event.
- `inserted event`: the synthetic appliance event placed into the window.
- `final aggregate`: the aggregate after insertion.
- `on_off`: the inserted appliance state label.

The metadata also records the original synthetic event before insertion: source event ID, source start, source end, selected real background window, and insertion location.

## dishwasher

Example D2 synthetic window index: `195`

Original synthetic event ID: `10`

Original synthetic event location: row `26020` to `26111` in the synthetic sequence

Real background window used: `52`

Inserted ON event location inside the 512-step training window: `345` to `436`

Inserted event length: `92` timesteps

Inserted event energy: `55157.82`

| t | original background `X_r - Y_r` | inserted event `Y_s` | final aggregate `X_s` | on_off |
|---:|---:|---:|---:|---:|
| 342 | 343.25 | 0.00 | 343.25 | 0 |
| 343 | 347.80 | 0.00 | 347.80 | 0 |
| 344 | 350.78 | 0.00 | 350.78 | 0 |
| 345 | 348.44 | 24.55 | 372.99 | 1 |
| 346 | 341.40 | 111.32 | 452.72 | 1 |
| 347 | 399.50 | 127.02 | 526.52 | 1 |
| 348 | 423.88 | 127.06 | 550.93 | 1 |
| 349 | 424.10 | 127.96 | 552.06 | 1 |
| 350 | 427.70 | 126.79 | 554.49 | 1 |
| 351 | 430.44 | 126.79 | 557.24 | 1 |
| 352 | 429.60 | 126.83 | 556.43 | 1 |
| 353 | 427.70 | 125.66 | 553.36 | 1 |
| 354 | 421.89 | 125.97 | 547.85 | 1 |
| 355 | 415.56 | 126.91 | 542.46 | 1 |
| 356 | 412.44 | 125.74 | 538.18 | 1 |
| 357 | 356.00 | 125.74 | 481.74 | 1 |
| 358 | 325.70 | 125.78 | 451.48 | 1 |
| 359 | 322.33 | 125.48 | 447.81 | 1 |
| 360 | 322.62 | 124.31 | 446.93 | 1 |
| 361 | 322.40 | 74.05 | 396.45 | 1 |
| 362 | 322.60 | 34.33 | 356.93 | 1 |
| 363 | 325.38 | 555.96 | 881.34 | 1 |
| 364 | 322.67 | 2336.23 | 2658.89 | 1 |
| 365 | 325.60 | 2297.72 | 2623.32 | 1 |
| 366 | 335.22 | 2316.99 | 2652.21 | 1 |
| 367 | 335.67 | 2336.30 | 2671.97 | 1 |
| 368 | 337.00 | 2345.68 | 2682.68 | 1 |
| 369 | 338.33 | 2336.34 | 2674.67 | 1 |
| 370 | 337.33 | 2346.02 | 2683.35 | 1 |
| 371 | 338.60 | 2346.02 | 2684.62 | 1 |
| 372 | 339.38 | 2317.14 | 2656.52 | 1 |
| 373 | 339.00 | 2326.82 | 2665.82 | 1 |
| 374 | 330.80 | 2307.28 | 2638.08 | 1 |
| 375 | 325.50 | 1478.45 | 1803.95 | 1 |
| 376 | 324.90 | 120.81 | 445.71 | 1 |
| 377 | 332.80 | 120.85 | 453.65 | 1 |
| 378 | 337.62 | 119.68 | 457.30 | 1 |
| 379 | 336.80 | 122.09 | 458.89 | 1 |
| 380 | 335.33 | 123.03 | 458.36 | 1 |
| 381 | 338.00 | 125.74 | 463.74 | 1 |
| 382 | 337.50 | 125.78 | 463.28 | 1 |
| 383 | 335.67 | 127.92 | 463.59 | 1 |
| 384 | 339.67 | 125.55 | 465.22 | 1 |
| 385 | 341.40 | 124.65 | 466.05 | 1 |
| 386 | 334.90 | 123.48 | 458.38 | 1 |
| 387 | 325.00 | 125.63 | 450.63 | 1 |
| 388 | 328.88 | 124.72 | 453.60 | 1 |
| 389 | 340.56 | 124.46 | 465.01 | 1 |
| 390 | 339.70 | 125.70 | 465.40 | 1 |
| 391 | 339.33 | 125.74 | 465.07 | 1 |
| 392 | 339.10 | 125.44 | 464.54 | 1 |
| 393 | 338.60 | 126.68 | 465.28 | 1 |
| 394 | 338.44 | 125.51 | 463.96 | 1 |
| 395 | 340.50 | 127.06 | 467.56 | 1 |
| 396 | 338.50 | 125.55 | 464.05 | 1 |
| 397 | 337.14 | 126.79 | 463.94 | 1 |
| 398 | 329.50 | 125.63 | 455.13 | 1 |
| 399 | 328.50 | 124.42 | 452.92 | 1 |
| 400 | 343.00 | 125.66 | 468.66 | 1 |
| 401 | 341.00 | 124.50 | 465.50 | 1 |
| 402 | 343.10 | 124.53 | 467.63 | 1 |
| 403 | 339.78 | 124.53 | 464.31 | 1 |
| 404 | 553.56 | 123.07 | 676.62 | 1 |
| 405 | 990.50 | 124.61 | 1115.11 | 1 |
| 406 | 1574.90 | 124.35 | 1699.25 | 1 |
| 407 | 800.00 | 125.59 | 925.59 | 1 |
| 408 | 784.80 | 126.79 | 911.59 | 1 |
| 409 | 721.62 | 125.93 | 847.55 | 1 |
| 410 | 692.40 | 125.66 | 818.06 | 1 |
| 411 | 704.30 | 128.11 | 832.41 | 1 |
| 412 | 699.00 | 126.91 | 825.91 | 1 |
| 413 | 700.90 | 126.94 | 827.84 | 1 |
| 414 | 700.40 | 125.48 | 825.88 | 1 |
| 415 | 633.78 | 123.41 | 757.18 | 1 |
| 416 | 640.00 | 121.00 | 761.00 | 1 |
| 417 | 641.44 | 125.55 | 767.00 | 1 |
| 418 | 636.44 | 126.79 | 763.24 | 1 |
| 419 | 630.70 | 125.63 | 756.33 | 1 |
| 420 | 617.20 | 125.63 | 742.83 | 1 |
| 421 | 613.75 | 125.36 | 739.11 | 1 |
| 422 | 638.70 | 125.70 | 764.40 | 1 |
| 423 | 635.40 | 125.74 | 761.14 | 1 |
| 424 | 647.78 | 126.98 | 774.76 | 1 |
| 425 | 696.44 | 124.27 | 820.72 | 1 |
| 426 | 695.70 | 125.51 | 821.21 | 1 |
| 427 | 761.50 | 49.66 | 811.16 | 1 |
| 428 | 783.22 | 507.47 | 1290.70 | 1 |
| 429 | 776.11 | 2355.47 | 3131.58 | 1 |
| 430 | 772.38 | 2326.33 | 3098.70 | 1 |
| 431 | 744.80 | 2336.27 | 3081.07 | 1 |
| 432 | 1757.00 | 2336.30 | 4093.30 | 1 |
| 433 | 1994.67 | 2317.07 | 4311.73 | 1 |
| 434 | 2956.80 | 2317.10 | 5273.90 | 1 |
| 435 | 1536.25 | 2268.61 | 3804.86 | 1 |
| 436 | 689.40 | 2307.20 | 2996.60 | 1 |
| 437 | 704.40 | 0.00 | 704.40 | 0 |
| 438 | 694.25 | 0.00 | 694.25 | 0 |
| 439 | 697.40 | 0.00 | 697.40 | 0 |

Short interpretation:

- Before insertion, the background around this event is approximately `X_s - Y_s`.
- The synthetic `dishwasher` event is added only during the ON-labelled timesteps.
- Outside the event, `Y_s` is zero, so the final aggregate equals the background.

## fridge

Example D2 synthetic window index: `195`

Original synthetic event ID: `412`

Original synthetic event location: row `24762` to `24771` in the synthetic sequence

Real background window used: `52`

Inserted ON event location inside the 512-step training window: `442` to `451`

Inserted event length: `10` timesteps

Inserted event energy: `863.11`

| t | original background `X_r - Y_r` | inserted event `Y_s` | final aggregate `X_s` | on_off |
|---:|---:|---:|---:|---:|
| 439 | 632.40 | 0.00 | 632.40 | 0 |
| 440 | 654.59 | 0.00 | 654.59 | 0 |
| 441 | 638.29 | 0.00 | 638.29 | 0 |
| 442 | 628.80 | 94.56 | 723.36 | 1 |
| 443 | 621.11 | 87.23 | 708.34 | 1 |
| 444 | 633.71 | 87.54 | 721.25 | 1 |
| 445 | 663.47 | 86.89 | 750.36 | 1 |
| 446 | 664.33 | 86.57 | 750.90 | 1 |
| 447 | 666.80 | 85.93 | 752.73 | 1 |
| 448 | 666.44 | 85.28 | 751.73 | 1 |
| 449 | 648.56 | 84.64 | 733.19 | 1 |
| 450 | 678.40 | 83.03 | 761.43 | 1 |
| 451 | 678.90 | 81.44 | 760.34 | 1 |
| 452 | 706.20 | 0.00 | 706.20 | 0 |
| 453 | 752.62 | 0.00 | 752.62 | 0 |
| 454 | 750.60 | 0.00 | 750.60 | 0 |

Short interpretation:

- Before insertion, the background around this event is approximately `X_s - Y_s`.
- The synthetic `fridge` event is added only during the ON-labelled timesteps.
- Outside the event, `Y_s` is zero, so the final aggregate equals the background.

## kettle

Example D2 synthetic window index: `195`

Original synthetic event ID: `54`

Original synthetic event location: row `31086` to `31088` in the synthetic sequence

Real background window used: `52`

Inserted ON event location inside the 512-step training window: `434` to `436`

Inserted event length: `3` timesteps

Inserted event energy: `6312.02`

| t | original background `X_r - Y_r` | inserted event `Y_s` | final aggregate `X_s` | on_off |
|---:|---:|---:|---:|---:|
| 431 | 367.00 | 0.00 | 367.00 | 0 |
| 432 | 377.30 | 0.00 | 377.30 | 0 |
| 433 | 379.00 | 0.00 | 379.00 | 0 |
| 434 | 379.14 | 1698.20 | 2077.35 | 1 |
| 435 | 380.43 | 2330.50 | 2710.93 | 1 |
| 436 | 377.50 | 2283.32 | 2660.82 | 1 |
| 437 | 377.44 | 0.00 | 377.44 | 0 |
| 438 | 378.00 | 0.00 | 378.00 | 0 |
| 439 | 379.44 | 0.00 | 379.44 | 0 |

Short interpretation:

- Before insertion, the background around this event is approximately `X_s - Y_s`.
- The synthetic `kettle` event is added only during the ON-labelled timesteps.
- Outside the event, `Y_s` is zero, so the final aggregate equals the background.

## microwave

Example D2 synthetic window index: `195`

Original synthetic event ID: `53`

Original synthetic event location: row `26496` to `26496` in the synthetic sequence

Real background window used: `52`

Inserted ON event location inside the 512-step training window: `434` to `434`

Inserted event length: `1` timesteps

Inserted event energy: `1016.05`

| t | original background `X_r - Y_r` | inserted event `Y_s` | final aggregate `X_s` | on_off |
|---:|---:|---:|---:|---:|
| 431 | 353.90 | 0.00 | 353.90 | 0 |
| 432 | 355.90 | 0.00 | 355.90 | 0 |
| 433 | 357.57 | 0.00 | 357.57 | 0 |
| 434 | 583.00 | 1016.05 | 1599.05 | 1 |
| 435 | 650.50 | 0.00 | 650.50 | 0 |
| 436 | 648.10 | 0.00 | 648.10 | 0 |
| 437 | 645.30 | 0.00 | 645.30 | 0 |

Short interpretation:

- Before insertion, the background around this event is approximately `X_s - Y_s`.
- The synthetic `microwave` event is added only during the ON-labelled timesteps.
- Outside the event, `Y_s` is zero, so the final aggregate equals the background.

## washingmachine

Example D2 synthetic window index: `195`

Original synthetic event ID: `22`

Original synthetic event location: row `24114` to `24210` in the synthetic sequence

Real background window used: `52`

Inserted ON event location inside the 512-step training window: `353` to `449`

Inserted event length: `97` timesteps

Inserted event energy: `49986.38`

| t | original background `X_r - Y_r` | inserted event `Y_s` | final aggregate `X_s` | on_off |
|---:|---:|---:|---:|---:|
| 350 | 374.75 | 0.00 | 374.75 | 0 |
| 351 | 393.22 | 0.00 | 393.22 | 0 |
| 352 | 392.80 | 0.00 | 392.80 | 0 |
| 353 | 379.89 | 27.16 | 407.05 | 1 |
| 354 | 353.00 | 60.74 | 413.74 | 1 |
| 355 | 347.20 | 125.80 | 473.00 | 1 |
| 356 | 344.25 | 354.57 | 698.82 | 1 |
| 357 | 348.80 | 1611.78 | 1960.58 | 1 |
| 358 | 351.78 | 1947.59 | 2299.37 | 1 |
| 359 | 349.44 | 1947.59 | 2297.03 | 1 |
| 360 | 342.40 | 1947.59 | 2289.99 | 1 |
| 361 | 400.50 | 1955.98 | 2356.48 | 1 |
| 362 | 424.88 | 1955.98 | 2380.86 | 1 |
| 363 | 425.10 | 1955.97 | 2381.07 | 1 |
| 364 | 428.70 | 1955.98 | 2384.68 | 1 |
| 365 | 431.44 | 1947.59 | 2379.03 | 1 |
| 366 | 430.60 | 1939.18 | 2369.78 | 1 |
| 367 | 428.70 | 1947.59 | 2376.29 | 1 |
| 368 | 422.89 | 1939.18 | 2362.07 | 1 |
| 369 | 416.56 | 1947.56 | 2364.12 | 1 |
| 370 | 413.44 | 1964.36 | 2377.80 | 1 |
| 371 | 357.00 | 1846.81 | 2203.81 | 1 |
| 372 | 326.70 | 1653.73 | 1980.43 | 1 |
| 373 | 323.33 | 228.09 | 551.42 | 1 |
| 374 | 323.62 | 186.62 | 510.25 | 1 |
| 375 | 323.40 | 214.42 | 537.82 | 1 |
| 376 | 323.60 | 193.44 | 517.04 | 1 |
| 377 | 326.38 | 214.42 | 540.79 | 1 |
| 378 | 323.67 | 1620.13 | 1943.80 | 1 |
| 379 | 326.60 | 1191.97 | 1518.57 | 1 |
| 380 | 336.22 | 180.84 | 517.06 | 1 |
| 381 | 336.67 | 239.07 | 575.74 | 1 |
| 382 | 338.00 | 206.02 | 544.02 | 1 |
| 383 | 339.33 | 169.28 | 508.61 | 1 |
| 384 | 338.33 | 138.33 | 476.66 | 1 |
| 385 | 339.60 | 1754.45 | 2094.05 | 1 |
| 386 | 340.38 | 257.96 | 598.33 | 1 |
| 387 | 340.00 | 209.15 | 549.15 | 1 |
| 388 | 331.80 | 210.73 | 542.53 | 1 |
| 389 | 326.50 | 253.75 | 580.25 | 1 |
| 390 | 325.90 | 188.69 | 514.59 | 1 |
| 391 | 333.80 | 1393.42 | 1727.22 | 1 |
| 392 | 338.62 | 209.17 | 547.79 | 1 |
| 393 | 337.80 | 166.65 | 504.45 | 1 |
| 394 | 336.33 | 185.53 | 521.87 | 1 |
| 395 | 339.00 | 218.58 | 557.58 | 1 |
| 396 | 338.50 | 207.06 | 545.56 | 1 |
| 397 | 336.67 | 200.75 | 537.42 | 1 |
| 398 | 340.67 | 205.47 | 546.13 | 1 |
| 399 | 342.40 | 193.39 | 535.79 | 1 |
| 400 | 335.90 | 170.85 | 506.75 | 1 |
| 401 | 326.00 | 172.94 | 498.94 | 1 |
| 402 | 329.88 | 178.70 | 508.57 | 1 |
| 403 | 341.56 | 188.13 | 529.69 | 1 |
| 404 | 340.70 | 225.38 | 566.08 | 1 |
| 405 | 340.33 | 200.74 | 541.07 | 1 |
| 406 | 340.10 | 194.98 | 535.08 | 1 |
| 407 | 339.60 | 69.02 | 408.62 | 1 |
| 408 | 339.44 | 254.77 | 594.22 | 1 |
| 409 | 341.50 | 20.73 | 362.23 | 1 |
| 410 | 339.50 | 28.08 | 367.58 | 1 |
| 411 | 338.14 | 31.23 | 369.38 | 1 |
| 412 | 330.50 | 60.62 | 391.12 | 1 |
| 413 | 329.50 | 118.35 | 447.85 | 1 |
| 414 | 344.00 | 144.56 | 488.56 | 1 |
| 415 | 342.00 | 91.05 | 433.05 | 1 |
| 416 | 344.10 | 72.13 | 416.23 | 1 |
| 417 | 340.78 | 123.56 | 464.34 | 1 |
| 418 | 554.56 | 213.85 | 768.40 | 1 |
| 419 | 991.50 | 114.13 | 1105.63 | 1 |
| 420 | 1575.90 | 125.68 | 1701.58 | 1 |
| 421 | 801.00 | 390.14 | 1191.14 | 1 |
| 422 | 785.80 | 147.70 | 933.50 | 1 |
| 423 | 722.62 | 125.67 | 848.29 | 1 |
| 424 | 693.40 | 52.21 | 745.61 | 1 |
| 425 | 705.30 | 289.36 | 994.66 | 1 |
| 426 | 700.00 | 357.58 | 1057.58 | 1 |
| 427 | 701.90 | 94.18 | 796.08 | 1 |
| 428 | 701.40 | 66.87 | 768.27 | 1 |
| 429 | 634.78 | 109.90 | 744.68 | 1 |
| 430 | 641.00 | 163.43 | 804.43 | 1 |
| 431 | 642.44 | 102.57 | 745.02 | 1 |
| 432 | 637.44 | 136.13 | 773.57 | 1 |
| 433 | 631.70 | 352.32 | 984.02 | 1 |
| 434 | 618.20 | 201.74 | 819.94 | 1 |
| 435 | 614.75 | 102.57 | 717.32 | 1 |
| 436 | 639.70 | 105.69 | 745.39 | 1 |
| 437 | 636.40 | 46.93 | 683.33 | 1 |
| 438 | 648.78 | 100.46 | 749.24 | 1 |
| 439 | 697.44 | 368.07 | 1065.52 | 1 |
| 440 | 696.70 | 281.99 | 978.69 | 1 |
| 441 | 762.50 | 249.47 | 1011.97 | 1 |
| 442 | 784.22 | 172.84 | 957.06 | 1 |
| 443 | 777.11 | 197.51 | 974.62 | 1 |
| 444 | 773.38 | 354.40 | 1127.78 | 1 |
| 445 | 745.80 | 381.66 | 1127.46 | 1 |
| 446 | 1757.90 | 387.97 | 2145.87 | 1 |
| 447 | 1995.67 | 34.32 | 2029.99 | 1 |
| 448 | 2957.80 | 99.36 | 3057.16 | 1 |
| 449 | 1537.25 | 43.75 | 1581.00 | 1 |
| 450 | 690.40 | 0.00 | 690.40 | 0 |
| 451 | 705.40 | 0.00 | 705.40 | 0 |
| 452 | 695.25 | 0.00 | 695.25 | 0 |

Short interpretation:

- Before insertion, the background around this event is approximately `X_s - Y_s`.
- The synthetic `washingmachine` event is added only during the ON-labelled timesteps.
- Outside the event, `Y_s` is zero, so the final aggregate equals the background.
