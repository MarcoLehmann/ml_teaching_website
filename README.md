Simple D3 Visualizer: y = 0.4 x + 0.6 with Noisy Points

This minimal webpage shows a thin green reference line y = 0.4 x + 0.6 and 1–10 green datapoints sampled with Gaussian noise around that line within x ∈ [-5, 5].

Run
- Open index.html in your browser. No build step needed.

Controls
- Points: number of points (1–10)
- Sample: resamples the points (the reference line is fixed)
- a (blue slope): slider in [-2, 2]
- b (blue intercept): slider in [-2, 2]

Model vs Data
- The blue line y = a x + b is a model for the sampled green points.
- For each data x, the model prediction y_hat is computed and shown as a vertical red residual line from the point to the model.
- Each residual is completed to a square (area equals residual^2), drawn to the right of the vertical segment.

Tech
- D3.js v7 via CDN
- Vanilla HTML/CSS/JS

Notes
- The plotting range is fixed to x, y in [-5, 5].
- Noise uses a small fixed standard deviation to keep points near the line.

