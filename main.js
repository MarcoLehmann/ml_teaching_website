/**
 * ===========================================================================
 * STOCHASTIC GRADIENT DESCENT INTERACTIVE TUTORIAL
 * ===========================================================================
 * 
 * This application provides an interactive visualization of gradient-based
 * optimization methods for machine learning, specifically:
 * - Gradient Descent (GD)
 * - Stochastic Gradient Descent (SGD)
 * - Annealed Stochastic Gradient Descent (ASGD)
 * 
 * The tutorial uses a simple linear regression problem (y = ax + b) to
 * demonstrate how these optimization algorithms work, visualizing:
 * - Error landscapes (MSE as a function of parameters)
 * - Optimization trajectories
 * - Learning curves
 * - The effect of batch size and learning rate
 * 
 * Architecture:
 * 1. Global utilities (MSE, gradient computation, landscape rendering)
 * 2. Data generation and visualization
 * 3. Model visualization with residuals
 * 4. Error landscape visualization
 * 5. Gradient Descent implementation and visualization
 * 6. Stochastic Gradient Descent with mini-batches
 * 7. Annealed SGD with learning rate decay
 * 8. Learning curves for all algorithms
 * 
 * @author Marco Lehmann
 * @version 2.0 - Refactored for maintainability
 */

// ============================================================================
// GLOBAL CONFIGURATION
// ============================================================================

// True parameters for data generation
const TRUE_A = 0.4;
const TRUE_B = 0.3;

// Visualization domains
const xDomain = [-5, 5];
const sampleXMin = -4;
const sampleXMax = 4;
const yDomain = [-4, 4];

// Parameter ranges (used across multiple visualizations)
const PARAM_RANGES = {
  aMin: 0.1,
  aMax: 0.7,
  bMin: 0.0,
  bMax: 0.6
};

// Visualization dimensions and margins
const VIZ_DIMENSIONS = {
  // Standard plot dimensions
  standard: { width: 760, height: 520, margin: { top: 30, right: 80, bottom: 50, left: 60 } },
  // Data plot dimensions
  data: { width: 760, height: 400, margin: { top: 20, right: 20, bottom: 40, left: 50 } },
  // Model plot dimensions
  model: { width: 760, height: 520, margin: { top: 24, right: 24, bottom: 44, left: 56 } },
  // Landscape plot dimensions
  landscape: { width: 840, height: 520, margin: { top: 30, right: 100, bottom: 50, left: 60 } },
  // Small plot dimensions (grid layouts)
  small: { width: 360, height: 320, margin: { top: 30, right: 20, bottom: 40, left: 50 } },
  // Learning curve dimensions
  curve: { width: 360, height: 280, margin: { top: 30, right: 20, bottom: 40, left: 50 } },
  // Alpha plot dimensions
  alpha: { width: 360, height: 220, margin: { top: 24, right: 20, bottom: 36, left: 48 } }
};

// Colors
const COLORS = {
  text: '#ffffff',
  textMuted: '#c9d4e5',
  trajectory: '#ff00ff',
  trajectoryPoint: '#ffffff',
  gradient: '#ffff00',
  sgd: '#00aaff',
  asgd: '#ffaa00',
  modelLine: '#3a89ff',
  truthLine: '#2ecc71',
  residual: '#ff4d4d',
  point: '#2ecc71'
};

// Animation timing (single value for all animations)
const ANIMATION_INTERVAL = 50;

// Data state
let noiseStdDev = 0.5;
let currentDataPoints = [];
let randNorm = d3.randomNormal.source(d3.randomLcg(Math.random()))(0, noiseStdDev);

// ============================================================================
// GLOBAL UTILITIES
// ============================================================================

/**
 * Safely retrieves a DOM element by ID, with error handling.
 * @param {string} id - Element ID
 * @returns {HTMLElement|null} The element or null if not found
 */
function getElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    console.warn(`Element with id "${id}" not found`);
  }
  return element;
}

/**
 * Updates a display element with a formatted value.
 * @param {string} id - Element ID to update
 * @param {number|string} value - Value to display
 * @param {string|Function} format - Format function or format string ('fixed', 'int', 'float')
 */
function updateDisplay(id, value, format = 'fixed') {
  const element = getElement(id);
  if (!element) return;
  
  let formatted;
  if (typeof format === 'function') {
    formatted = format(value);
  } else if (format === 'int') {
    formatted = String(parseInt(value, 10));
  } else if (format === 'float') {
    formatted = parseFloat(value).toString();
  } else if (typeof format === 'number') {
    formatted = parseFloat(value).toFixed(format);
  } else {
    formatted = parseFloat(value).toFixed(2);
  }
  
  element.textContent = formatted;
}

/**
 * Binds a slider to update its display value and optionally call a callback.
 * @param {string} sliderId - Slider input element ID
 * @param {string} displayId - Display element ID (optional)
 * @param {Object} options - Configuration options
 * @param {string|number|Function} options.format - Format for display ('fixed', number of decimals, or function)
 * @param {Function} options.onChange - Optional callback function(value)
 * @param {Function} options.onInput - Optional callback for input event (different from onChange)
 */
function bindSlider(sliderId, displayId = null, options = {}) {
  const slider = getElement(sliderId);
  if (!slider) return;
  
  const { format = 'fixed', onChange = null, onInput = null } = options;
  
  slider.addEventListener('input', () => {
    const value = parseFloat(slider.value);
    
    // Update display if provided
    if (displayId) {
      updateDisplay(displayId, value, format);
    }
    
    // Call onInput callback if provided
    if (onInput) {
      onInput(value);
    }
  });
  
  // Call onChange callback if provided (for change event, not input)
  if (onChange) {
    slider.addEventListener('change', () => {
      onChange(parseFloat(slider.value));
    });
  }
}

/**
 * Computes the Mean Squared Error (MSE) for a linear model.
 * MSE = (1/2N) * Σ(y_i - (a*x_i + b))²
 * 
 * @param {Array<{x: number, y: number}>} points - Data points
 * @param {number} a - Slope parameter
 * @param {number} b - Intercept parameter
 * @returns {number} The computed MSE
 */
function computeMSE(points, a, b) {
  if (!points || points.length === 0) return 0;
  if (!isFinite(a) || !isFinite(b)) return Infinity;
  
  let sum = 0;
  for (const d of points) {
    if (!isFinite(d.x) || !isFinite(d.y)) continue;
    const err = d.y - (a * d.x + b);
    sum += err * err;
  }
  const result = sum / (2 * points.length);
  return isFinite(result) ? result : Infinity;
}

/**
 * Renders a data plot with optional model line, truth line, residuals, and squares.
 * This is a reusable parametrized function to avoid code duplication.
 * 
 * @param {Object} config - Configuration object
 * @param {d3.Selection} config.svg - SVG selection to render into
 * @param {number} config.width - Total SVG width
 * @param {number} config.height - Total SVG height
 * @param {Object} config.margin - Margin object {top, right, bottom, left}
 * @param {Array<{x: number, y: number}>} config.dataPoints - Data points to display
 * @param {number} [config.modelA] - Model parameter a (slope), if null no model line shown
 * @param {number} [config.modelB] - Model parameter b (intercept), if null no model line shown
 * @param {boolean} [config.showTruthLine=false] - Show the green dashed truth line
 * @param {boolean} [config.showModelLine=false] - Show the blue model line
 * @param {boolean} [config.showResiduals=false] - Show red vertical residual lines
 * @param {boolean} [config.showSquares=false] - Show red squares
 * @param {boolean} [config.showMSE=false] - Show MSE text
 * @param {string} [config.title] - Optional title text
 * @param {number} [config.axisTicks=10] - Number of axis ticks
 * @param {boolean} [config.showGrid=true] - Show grid lines
 * @param {boolean} [config.showZeroLines=false] - Show emphasized zero lines
 * @returns {Object} Object containing {xScale, yScale, g} for further customization
 */
function renderDataPlot(config) {
  const {
    svg,
    width,
    height,
    margin,
    dataPoints,
    modelA = null,
    modelB = null,
    showTruthLine = false,
    showModelLine = false,
    showResiduals = false,
    showSquares = false,
    showMSE = false,
    title = null,
    axisTicks = 10,
    showGrid = true,
    showZeroLines = false
  } = config;

  const maxW = width - margin.left - margin.right;
  const maxH = height - margin.top - margin.bottom;

  svg.selectAll('*').remove();
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // Calculate data ranges
  const xRange = xDomain[1] - xDomain[0]; // 10 units
  const yRange = yDomain[1] - yDomain[0]; // 8 units

  // Calculate pixels per unit for each axis if we used all available space
  const pixelsPerUnitX = maxW / xRange;
  const pixelsPerUnitY = maxH / yRange;

  // Use the smaller scale to ensure equal scaling (both axes fit)
  const pixelsPerUnit = Math.min(pixelsPerUnitX, pixelsPerUnitY);

  // Calculate actual plot dimensions with equal scaling
  const w = xRange * pixelsPerUnit;
  const h = yRange * pixelsPerUnit;

  // Center the plot in available space
  const xOffset = (maxW - w) / 2;
  const yOffset = (maxH - h) / 2;

  // Adjust group transform to center the plot
  g.attr('transform', `translate(${margin.left + xOffset},${margin.top + yOffset})`);

  const xScale = d3.scaleLinear().domain(xDomain).range([0, w]);
  const yScale = d3.scaleLinear().domain(yDomain).range([h, 0]);

  // Grid
  if (showGrid) {
    g.append('g').attr('class', 'grid').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(axisTicks).tickSize(-h).tickFormat(''));
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(yScale).ticks(axisTicks).tickSize(-w).tickFormat(''));
  }

  // Emphasized zero lines
  if (showZeroLines) {
    g.append('line')
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', '#ffffff').attr('stroke-width', 2.5).attr('opacity', 0.8);
    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', yScale(0)).attr('y2', yScale(0))
      .attr('stroke', '#ffffff').attr('stroke-width', 2.5).attr('opacity', 0.8);
  } else {
    // Regular zero lines
    g.append('line').attr('class', 'zero-line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', yScale(0)).attr('y2', yScale(0));
    g.append('line').attr('class', 'zero-line')
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', 0).attr('y2', h);
  }

  // Axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(axisTicks > 5 ? 10 : 5));
  g.append('g').attr('class', 'axis').call(d3.axisLeft(yScale).ticks(axisTicks > 5 ? 10 : 5));

  // Axis labels
  const labelSize = axisTicks > 5 ? '14px' : '14px';
  g.append('text').attr('x', w / 2).attr('y', h + 35).attr('text-anchor', 'middle')
    .attr('fill', '#c9d4e5').attr('font-size', labelSize).text('x');
  g.append('text').attr('x', -35).attr('y', h / 2).attr('text-anchor', 'middle')
    .attr('fill', '#c9d4e5').attr('font-size', labelSize).text('y');

  const lineGen = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));

  // Truth line (green dashed)
  if (showTruthLine) {
    const truthLine = [
      { x: xDomain[0], y: TRUE_A * xDomain[0] + TRUE_B },
      { x: xDomain[1], y: TRUE_A * xDomain[1] + TRUE_B }
    ];
    g.append('path').datum(truthLine).attr('d', lineGen).attr('class', 'truth-line');
  }

  // Residuals and squares (must be drawn before model line to appear behind)
  if ((showResiduals || showSquares) && modelA !== null && modelB !== null) {
    const residuals = dataPoints.map(d => {
      const yHat = modelA * d.x + modelB;
      return { x: d.x, yData: d.y, yHat, y0: Math.min(yHat, d.y), y1: Math.max(yHat, d.y) };
    });

    if (showSquares) {
      const squares = residuals.map(d => {
        const size = Math.abs(d.yData - d.yHat);
        const yTop = d.y0 + size;
        return { x: d.x, y0: d.y0, size, yTop };
      });
      g.selectAll('rect.residual-square').data(squares).join('rect')
        .attr('class', 'residual-square')
        .attr('x', d => xScale(d.x))
        .attr('y', d => yScale(d.yTop))
        .attr('width', d => Math.abs(xScale(d.x + d.size) - xScale(d.x)))
        .attr('height', d => Math.abs(yScale(d.y0) - yScale(d.yTop)));
    }

    if (showResiduals) {
      g.selectAll('line.residual').data(residuals).join('line')
        .attr('class', 'residual')
        .attr('x1', d => xScale(d.x))
        .attr('x2', d => xScale(d.x))
        .attr('y1', d => yScale(d.yHat))
        .attr('y2', d => yScale(d.yData));
    }
  }

  // Model line (blue)
  if (showModelLine && modelA !== null && modelB !== null) {
    const modelLine = [
      { x: xDomain[0], y: modelA * xDomain[0] + modelB },
      { x: xDomain[1], y: modelA * xDomain[1] + modelB }
    ];
    g.append('path').datum(modelLine).attr('d', lineGen).attr('class', 'line');
  }

  // Data points
  g.selectAll('circle.point').data(dataPoints).join('circle')
    .attr('class', 'point')
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y))
    .attr('r', 6);

  // Title
  if (title) {
    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '12px').text(title);
  }

  // MSE
  if (showMSE && modelA !== null && modelB !== null) {
    const mse = computeMSE(dataPoints, modelA, modelB);
    g.append('text').attr('x', 10).attr('y', 20).attr('fill', '#ffffff').attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(`MSE: ${mse.toFixed(4)}`);
  }

  return { xScale, yScale, g };
}

/**
 * Computes the gradient of the MSE loss function for all data points.
 * Used in standard Gradient Descent.
 * 
 * @param {Array<{x: number, y: number}>} points - All data points
 * @param {number} a - Current slope parameter
 * @param {number} b - Current intercept parameter
 * @returns {{da: number, db: number}} Gradient components
 */
function computeGradient(points, a, b) {
  if (!points || points.length === 0) return { da: 0, db: 0 };
  if (!isFinite(a) || !isFinite(b)) return { da: 0, db: 0 };
  
  let sumDa = 0, sumDb = 0;
  for (const d of points) {
    if (!isFinite(d.x) || !isFinite(d.y)) continue;
    const err = d.y - (a * d.x + b);
    sumDa += err * (-d.x);
    sumDb += err * (-1);
  }
  const da = sumDa / points.length;
  const db = sumDb / points.length;
  return { 
    da: isFinite(da) ? da : 0, 
    db: isFinite(db) ? db : 0 
  };
}

/**
 * Computes the gradient using a randomly sampled mini-batch of data.
 * Used in Stochastic Gradient Descent (SGD) and Annealed SGD.
 * 
 * @param {Array<{x: number, y: number}>} points - All available data points
 * @param {number} a - Current slope parameter
 * @param {number} b - Current intercept parameter
 * @param {number} batchSize - Number of random samples to use
 * @returns {{da: number, db: number}} Approximate gradient components
 */
function computeGradientBatch(points, a, b, batchSize) {
  if (!points || points.length === 0) return { da: 0, db: 0 };
  if (!isFinite(a) || !isFinite(b) || !isFinite(batchSize) || batchSize <= 0) {
    return { da: 0, db: 0 };
  }
  
  const n = Math.min(batchSize, points.length);
  const indices = d3.shuffle(d3.range(points.length)).slice(0, n);
  let sumDa = 0, sumDb = 0;
  for (const idx of indices) {
    const d = points[idx];
    if (!isFinite(d.x) || !isFinite(d.y)) continue;
    const err = d.y - (a * d.x + b);
    sumDa += err * (-d.x);
    sumDb += err * (-1);
  }
  const da = sumDa / n;
  const db = sumDb / n;
  return { 
    da: isFinite(da) ? da : 0, 
    db: isFinite(db) ? db : 0 
  };
}

/**
 * Renders an error landscape heatmap with contour lines and axes.
 * This is the core visualization for showing how MSE varies with parameters a and b.
 * 
 * @param {d3.Selection} g - D3 selection for the SVG group element
 * @param {number} w - Width of the plot area
 * @param {number} h - Height of the plot area
 * @param {Array<{x: number, y: number}>} dataPoints - Data to compute errors for
 * @param {number} aMin - Minimum value for parameter a
 * @param {number} aMax - Maximum value for parameter a
 * @param {number} bMin - Minimum value for parameter b
 * @param {number} bMax - Maximum value for parameter b
 * @param {number} [meshPoints=25] - Resolution of the error grid
 * @returns {{xOffset: number, yOffset: number, plotSize: number, aScale: Function, bScale: Function, minErr: number, maxErr: number}}
 *          Layout and scale information for drawing overlays
 */
function renderErrorLandscape(g, w, h, dataPoints, aMin, aMax, bMin, bMax, meshPoints = 20) {
  // Make plot area square
  const plotSize = Math.min(w, h);
  const xOffset = (w - plotSize) / 2;
  const yOffset = (h - plotSize) / 2;
  const aScale = d3.scaleLinear().domain([aMin, aMax]).range([0, plotSize]);
  const bScale = d3.scaleLinear().domain([bMin, bMax]).range([plotSize, 0]);

  // Compute error grid
  const aArr = d3.range(aMin, aMax, (aMax - aMin) / meshPoints);
  const bArr = d3.range(bMin, bMax, (bMax - bMin) / meshPoints);
  const errors = [];
  let minErr = Infinity, maxErr = -Infinity;
  for (const b of bArr) {
    const row = [];
    for (const a of aArr) {
      const e = computeMSE(dataPoints, a, b);
      row.push(e);
      minErr = Math.min(minErr, e);
      maxErr = Math.max(maxErr, e);
    }
    errors.push(row);
  }

  // Render heatmap
  const colorScale = d3.scaleSequential(d3.interpolateReds).domain([minErr, maxErr]);
  for (let i = 0; i < bArr.length; i++) {
    for (let j = 0; j < aArr.length; j++) {
      g.append('rect')
        .attr('x', xOffset + aScale(aArr[j]) - (plotSize / aArr.length) / 2)
        .attr('y', yOffset + bScale(bArr[i]) - (plotSize / bArr.length) / 2)
        .attr('width', plotSize / aArr.length)
        .attr('height', plotSize / bArr.length)
        .attr('fill', colorScale(errors[i][j]))
        .attr('opacity', 0.7);
    }
  }

  // Add contour lines for better readability
  const aScaleOffset = d3.scaleLinear().domain([aMin, aMax]).range([xOffset, xOffset + plotSize]);
  const bScaleOffset = d3.scaleLinear().domain([bMin, bMax]).range([yOffset + plotSize, yOffset]);
  drawContourLines(g, errors, aArr, bArr, aScaleOffset, bScaleOffset, minErr, maxErr, 10);

  // Add axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${plotSize + yOffset})`)
    .call(d3.axisBottom(aScale).ticks(6));
  g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${yOffset})`)
    .call(d3.axisLeft(bScale).ticks(6));
  g.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
    .attr('fill', '#c9d4e5').text('a');
  g.append('text').attr('x', -30).attr('y', h / 2).attr('text-anchor', 'middle')
    .attr('fill', '#c9d4e5').text('b');

  // Return useful data for caller to draw trajectory or other overlays
  return { xOffset, yOffset, plotSize, aScale, bScale, minErr, maxErr };
}

/**
 * Draws the optimization trajectory and current position on an error landscape.
 * 
 * @param {d3.Selection} g - D3 selection for the SVG group element
 * @param {Array<{a: number, b: number}>} trajectory - History of parameter values
 * @param {{a: number, b: number}} current - Current parameter values
 * @param {{xOffset: number, yOffset: number, aScale: Function, bScale: Function}} landscapeData - Layout info from renderErrorLandscape
 */
function drawTrajectoryOnLandscape(g, trajectory, current, landscapeData) {
  const { xOffset, yOffset, aScale, bScale } = landscapeData;
  
  // Draw path if we have history
  if (trajectory.length > 1) {
    const lineGen = d3.line()
      .x(d => xOffset + aScale(d.a))
      .y(d => yOffset + bScale(d.b));
    g.append('path')
      .attr('d', lineGen(trajectory))
      .attr('stroke', COLORS.trajectory)
      .attr('stroke-width', 3)
      .attr('fill', 'none');

    // Draw trajectory points
    g.selectAll('circle.traj').data(trajectory).join('circle')
      .attr('class', 'traj')
      .attr('cx', d => xOffset + aScale(d.a))
      .attr('cy', d => yOffset + bScale(d.b))
      .attr('r', 2)
      .attr('fill', COLORS.trajectoryPoint);
  }

  // Draw current position
  g.append('circle')
    .attr('cx', xOffset + aScale(current.a))
    .attr('cy', yOffset + bScale(current.b))
    .attr('r', 6)
    .attr('fill', COLORS.trajectory)
    .attr('stroke', COLORS.trajectoryPoint)
    .attr('stroke-width', 2);
}

// ============================================================================
// Section 1: The Data
// ============================================================================
function initDataViz() {
  const svg = d3.select('#viz-data');
  const dims = VIZ_DIMENSIONS.data;
  const width = dims.width;
  const height = dims.height;
  const margin = dims.margin;

  function generateData(n, sigma) {
    if (!isFinite(n) || n < 1) n = 1;
    if (!isFinite(sigma) || sigma < 0) sigma = 0;
    
    const rng = d3.randomNormal.source(d3.randomLcg(Math.random()))(0, sigma);
    if (n === 1) return [{ x: 0, y: TRUE_A * 0 + TRUE_B + rng() }];
    // Sample n points uniformly in the range [sampleXMin, sampleXMax]
    const step = (sampleXMax - sampleXMin) / (n - 1);
    return Array.from({ length: n }, (_, i) => {
      const x = sampleXMin + i * step;
      return { x, y: TRUE_A * x + TRUE_B + rng() };
    });
  }

  function render() {
    const dataPointsEl = getElement('dataPoints');
    const noiseSigmaEl = getElement('noiseSigma');
    if (!dataPointsEl || !noiseSigmaEl) return;
    
    const n = parseInt(dataPointsEl.value, 10);
    noiseStdDev = parseFloat(noiseSigmaEl.value);
    currentDataPoints = generateData(n, noiseStdDev);
    
    updateDisplay('dataPointsValue', n, 'int');
    updateDisplay('noiseSigmaValue', noiseStdDev, 1);

    renderDataPlot({
      svg,
      width,
      height,
      margin,
      dataPoints: currentDataPoints,
      showTruthLine: true,
      showZeroLines: true,
      axisTicks: 10,
      showGrid: true
    });
  }

  bindSlider('dataPoints', 'dataPointsValue', { format: 'int', onInput: render });
  bindSlider('noiseSigma', 'noiseSigmaValue', { 
    format: 1, 
    onInput: (value) => {
      render();
      window.dispatchEvent(new Event('dataUpdated'));
    }
  });
  
  const sampleBtn = getElement('sampleDataBtn');
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      render();
      window.dispatchEvent(new Event('dataUpdated'));
    });
  }
  
  render();
}

// ============================================================================
// Section 6b: Annealed Stochastic Gradient Descent (ASGD)
// ============================================================================
function initAnnealedSGDViz() {
  const svg = d3.select('#viz-asgd');
  const dims = VIZ_DIMENSIONS.standard;
  const width = dims.width;
  const height = dims.height;
  const margin = dims.margin;
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  // Helper to generate random starting position within valid ranges
  function randomStartPosition() {
    return {
      a: aMin + Math.random() * (aMax - aMin),
      b: bMin + Math.random() * (bMax - bMin)
    };
  }

  function render() {
    if (currentDataPoints.length === 0) return;
    g.selectAll('*').remove();

    // Render landscape and get scale info
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, 25);
    
    // Draw trajectory and current point
    drawTrajectoryOnLandscape(g, asgdTrajectory, asgdCurrent, landscapeData);
  }

  function step() {
    const batchSize = parseInt(document.getElementById('asgdBatch').value, 10);
    const { da, db } = computeGradientBatch(currentDataPoints, asgdCurrent.a, asgdCurrent.b, batchSize);

    const alpha = asgdHistory.length === 0
      ? parseFloat(document.getElementById('asgdAlpha0').value)
      : asgdHistory[asgdHistory.length - 1].alpha;

    asgdCurrent.a -= alpha * da;
    asgdCurrent.b -= alpha * db;
    asgdTrajectory.push({ a: asgdCurrent.a, b: asgdCurrent.b });

    const mse = computeMSE(currentDataPoints, asgdCurrent.a, asgdCurrent.b);

    const anneal = parseFloat(document.getElementById('asgdAnneal').value);
    const lower = parseFloat(document.getElementById('asgdLower').value);
    const nextAlpha = Math.max(alpha * anneal, lower);

    asgdHistory.push({ mse, a: asgdCurrent.a, b: asgdCurrent.b, alpha: nextAlpha });
    window.dispatchEvent(new Event('asgdUpdated'));

    document.getElementById('asgdMseValue').textContent = mse.toFixed(4);
    document.getElementById('asgdCurrentA').textContent = asgdCurrent.a.toFixed(3);
    document.getElementById('asgdCurrentB').textContent = asgdCurrent.b.toFixed(3);
    document.getElementById('asgdAlphaNow').textContent = nextAlpha.toFixed(4);
    render();
  }

  (function initAlphaPlot() {
    const svgAlpha = d3.select('#viz-asgd-alpha');
    const wA = 360, hA = 220;
    const marginA = { top: 24, right: 20, bottom: 36, left: 48 };
    svgAlpha.attr('viewBox', `0 0 ${wA} ${hA}`);
    const gA = svgAlpha.append('g').attr('transform', `translate(${marginA.left},${marginA.top})`);
    const w = wA - marginA.left - marginA.right;
    const h = hA - marginA.top - marginA.bottom;

    function renderAlpha() {
      gA.selectAll('*').remove();
      if (asgdHistory.length === 0) {
        gA.append('text').attr('x', w / 2).attr('y', h / 2)
          .attr('text-anchor', 'middle').attr('fill', '#c9d4e5').attr('font-size', '12px')
          .text('Run ASGD to see α decay');
        return;
      }
      const x = d3.scaleLinear().domain([0, asgdHistory.length]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(asgdHistory, d => d.alpha) || 1]).range([h, 0]);

      gA.append('g').attr('class', 'grid').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).ticks(5).tickSize(-h).tickFormat(''));
      gA.append('g').attr('class', 'grid')
        .call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''));

      gA.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).ticks(5));
      gA.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5));
      gA.append('text').attr('x', w / 2).attr('y', -8).attr('text-anchor', 'middle')
        .attr('fill', '#ffffff').attr('font-size', '12px').text('Learning Rate (α) over Iterations');
      gA.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle')
        .attr('fill', '#c9d4e5').attr('font-size', '11px').text('Iteration');
      gA.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle');
      gA.append('text').attr('x', -h / 2).attr('y', -36).attr('text-anchor', 'middle')
        .attr('fill', '#c9d4e5').attr('font-size', '11px')
        .attr('transform', `rotate(-90, -${h / 2}, -36)`).text('α');

      const line = d3.line().x((d, i) => x(i)).y(d => y(d.alpha));
      gA.append('path').datum(asgdHistory)
        .attr('d', line).attr('stroke', '#ffaa00').attr('stroke-width', 2).attr('fill', 'none');
    }

    window.addEventListener('asgdUpdated', renderAlpha);
    renderAlpha();
  })();

  document.getElementById('asgdStepBtn').addEventListener('click', step);
  document.getElementById('asgdRunBtn').addEventListener('click', () => {
    if (asgdRunning) { asgdRunning = false; return; }
    asgdRunning = true;
    const iters = parseInt(document.getElementById('asgdIterations').value, 10);
    let count = 0;
    const interval = setInterval(() => {
      if (count >= iters || !asgdRunning) {
        clearInterval(interval);
        asgdRunning = false;
        return;
      }
      step();
      count++;
    }, ANIMATION_INTERVAL);
  });

  document.getElementById('asgdAlpha0').addEventListener('input', () => {
    document.getElementById('asgdAlpha0Value').textContent = parseFloat(document.getElementById('asgdAlpha0').value).toFixed(3);
  });
  document.getElementById('asgdAnneal').addEventListener('input', () => {
    document.getElementById('asgdAnnealValue').textContent = parseFloat(document.getElementById('asgdAnneal').value).toFixed(3);
  });
  document.getElementById('asgdLower').addEventListener('input', () => {
    document.getElementById('asgdLowerValue').textContent = parseFloat(document.getElementById('asgdLower').value).toFixed(4);
  });
  document.getElementById('asgdA0').addEventListener('input', () => {
    document.getElementById('asgdA0Value').textContent = parseFloat(document.getElementById('asgdA0').value).toFixed(2);
    if (!asgdRunning) {
      asgdCurrent.a = parseFloat(document.getElementById('asgdA0').value);
      asgdTrajectory = [{ a: asgdCurrent.a, b: asgdCurrent.b }];
      render();
    }
  });
  document.getElementById('asgdB0').addEventListener('input', () => {
    document.getElementById('asgdB0Value').textContent = parseFloat(document.getElementById('asgdB0').value).toFixed(2);
    if (!asgdRunning) {
      asgdCurrent.b = parseFloat(document.getElementById('asgdB0').value);
      asgdTrajectory = [{ a: asgdCurrent.a, b: asgdCurrent.b }];
      render();
    }
  });

  document.getElementById('asgdResetBtn').addEventListener('click', () => {
    asgdRunning = false;
    // Generate new random starting position
    const startPos = randomStartPosition();
    asgdCurrent.a = startPos.a;
    asgdCurrent.b = startPos.b;
    // Update sliders to match
    document.getElementById('asgdA0').value = startPos.a;
    document.getElementById('asgdB0').value = startPos.b;
    document.getElementById('asgdA0Value').textContent = startPos.a.toFixed(2);
    document.getElementById('asgdB0Value').textContent = startPos.b.toFixed(2);
    asgdTrajectory = [{ a: asgdCurrent.a, b: asgdCurrent.b }];
    asgdHistory = [];
    const alpha0 = parseFloat(document.getElementById('asgdAlpha0').value);
    asgdHistory.push({ mse: computeMSE(currentDataPoints, asgdCurrent.a, asgdCurrent.b), a: asgdCurrent.a, b: asgdCurrent.b, alpha: alpha0 });
    window.dispatchEvent(new Event('asgdUpdated'));
    document.getElementById('asgdAlphaNow').textContent = alpha0.toFixed(4);
    render();
  });

  document.getElementById('asgdBatch').addEventListener('input', () => {
    document.getElementById('asgdBatchValue').textContent = String(parseInt(document.getElementById('asgdBatch').value, 10));
    if (!asgdRunning) render();
  });
  document.getElementById('asgdIterations').addEventListener('input', () => {
    document.getElementById('asgdIterValue').textContent = String(parseInt(document.getElementById('asgdIterations').value, 10));
  });

  window.addEventListener('dataUpdated', () => {
    asgdCurrent.a = parseFloat(document.getElementById('asgdA0').value);
    asgdCurrent.b = parseFloat(document.getElementById('asgdB0').value);
    asgdTrajectory = [{ a: asgdCurrent.a, b: asgdCurrent.b }];
    asgdHistory = [];
    const alpha0 = parseFloat(document.getElementById('asgdAlpha0').value);
    asgdHistory.push({ mse: computeMSE(currentDataPoints, asgdCurrent.a, asgdCurrent.b), a: asgdCurrent.a, b: asgdCurrent.b, alpha: alpha0 });
    document.getElementById('asgdAlphaNow').textContent = alpha0.toFixed(4);
    render();
  });

  asgdCurrent.a = parseFloat(document.getElementById('asgdA0').value);
  asgdCurrent.b = parseFloat(document.getElementById('asgdB0').value);
  asgdTrajectory = [{ a: asgdCurrent.a, b: asgdCurrent.b }];
  const alpha0 = parseFloat(document.getElementById('asgdAlpha0').value);
  asgdHistory = [{ mse: computeMSE(currentDataPoints, asgdCurrent.a, asgdCurrent.b), a: asgdCurrent.a, b: asgdCurrent.b, alpha: alpha0 }];
  document.getElementById('asgdAlphaNow').textContent = alpha0.toFixed(4);
  render(); // Initial render after deferred init
}

// ============================================================================
// Section 2: The Model
// ============================================================================
function initModelViz() {
  const svg = d3.select('#viz-model');
  const dims = VIZ_DIMENSIONS.model;
  const width = dims.width;
  const height = dims.height;
  const margin = dims.margin;

  function render() {
    const modelAEl = getElement('modelA');
    const modelBEl = getElement('modelB');
    if (!modelAEl || !modelBEl) return;
    
    const a = parseFloat(modelAEl.value);
    const b = parseFloat(modelBEl.value);
    
    if (!isFinite(a) || !isFinite(b)) return;
    
    updateDisplay('modelAValue', a, 2);
    updateDisplay('modelBValue', b, 2);

    renderDataPlot({
      svg,
      width,
      height,
      margin,
      dataPoints: currentDataPoints,
      modelA: a,
      modelB: b,
      showModelLine: true,
      showResiduals: false,
      showSquares: false,
      showMSE: false,
      axisTicks: 10,
      showGrid: true,
      showZeroLines: false
    });
  }

  bindSlider('modelA', 'modelAValue', { format: 2, onInput: render });
  bindSlider('modelB', 'modelBValue', { format: 2, onInput: render });
  window.addEventListener('dataUpdated', render);
  render();
}

// ============================================================================
// Section 3: Interactive MSE Minimization
// ============================================================================
function initModelMinimizationViz() {
  const svg = d3.select('#viz-model-minimization');
  const width = 760, height = 520;
  const margin = { top: 24, right: 24, bottom: 44, left: 56 };

  function render() {
    const a = parseFloat(document.getElementById('modelAMin').value);
    const b = parseFloat(document.getElementById('modelBMin').value);
    const showResidualsSquares = document.getElementById('showResidualsSquaresMinimization').checked;
    document.getElementById('modelAMinValue').textContent = Number(a).toFixed(2);
    document.getElementById('modelBMinValue').textContent = Number(b).toFixed(2);

    renderDataPlot({
      svg,
      width,
      height,
      margin,
      dataPoints: currentDataPoints,
      modelA: a,
      modelB: b,
      showModelLine: true,
      showResiduals: showResidualsSquares,
      showSquares: showResidualsSquares,
      showMSE: showResidualsSquares,
      axisTicks: 10,
      showGrid: true,
      showZeroLines: false
    });
  }

  document.getElementById('modelAMin').addEventListener('input', () => {
    const val = parseFloat(document.getElementById('modelAMin').value);
    document.getElementById('modelAMinValue').textContent = val.toFixed(2);
    render();
  });
  document.getElementById('modelBMin').addEventListener('input', () => {
    const val = parseFloat(document.getElementById('modelBMin').value);
    document.getElementById('modelBMinValue').textContent = val.toFixed(2);
    render();
  });
  document.getElementById('showResidualsSquaresMinimization').addEventListener('change', render);
  window.addEventListener('dataUpdated', render);
  render();
}

// ============================================================================
// Section 4: Error Landscape
// ============================================================================
function initLandscapeViz() {
  const svg = d3.select('#viz-landscape');
  const width = 840, height = 520;
  const margin = { top: 30, right: 100, bottom: 50, left: 60 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const meshPoints = 25;
  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  function render() {
    if (currentDataPoints.length === 0) return;

    g.selectAll('*').remove();

    // Use the reusable renderErrorLandscape function
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, meshPoints);
    const { xOffset, yOffset, plotSize, aScale, bScale, minErr, maxErr } = landscapeData;
    
    // Adjust opacity for this visualization (0.8 instead of default 0.7)
    g.selectAll('rect').attr('opacity', 0.8);
    
    // Update axis ticks to 8 (instead of default 6)
    g.selectAll('g.axis').remove();
    g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${plotSize + yOffset})`)
      .call(d3.axisBottom(aScale).ticks(8));
    g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${yOffset})`)
      .call(d3.axisLeft(bScale).ticks(8));
    
    // Re-add axis labels (remove existing ones first)
    g.selectAll('text').each(function() {
      const text = d3.select(this).text();
      if (text === 'a' || text === 'b') {
        d3.select(this).remove();
      }
    });
    g.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').text('a');
    g.append('text').attr('x', -30).attr('y', h / 2).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').text('b');
    
    // Create colorScale for colorbar
    const colorScale = d3.scaleSequential(d3.interpolateReds).domain([minErr, maxErr]);

    // Title
    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '14px').text('Error Landscape: MSE as a function of a and b');

    // Colorbar
    const colorbarW = 20, colorbarH = h;
    const colorbarG = svg.append('g').attr('transform', `translate(${width - margin.right + 20},${margin.top})`);
    const colorGradient = colorbarG.append('defs').append('linearGradient')
      .attr('id', 'colorGrad').attr('x1', '0%').attr('y1', '100%').attr('x2', '0%').attr('y2', '0%');
    const nStops = 10;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const val = minErr + t * (maxErr - minErr);
      colorGradient.append('stop')
        .attr('offset', `${i * 10}%`)
        .attr('stop-color', colorScale(val));
    }
    colorbarG.append('rect')
      .attr('x', 0).attr('y', 0).attr('width', colorbarW).attr('height', colorbarH)
      .style('fill', 'url(#colorGrad)');
    
    // Custom 5 labels: min, max, and 3 evenly spaced in between
    const tickVals = [
      minErr,
      minErr + (maxErr - minErr) * 0.25,
      minErr + (maxErr - minErr) * 0.5,
      minErr + (maxErr - minErr) * 0.75,
      maxErr
    ];
    const colorScale2 = d3.scaleLinear().domain([minErr, maxErr]).range([colorbarH, 0]);
    const axisG = colorbarG.append('g').attr('transform', `translate(${colorbarW}, 0)`);
    axisG.call(d3.axisRight(colorScale2).tickValues(tickVals).tickFormat(d3.format('.2f')))
      .selectAll('text')
      .attr('fill', '#ffffff')
      .attr('font-size', '12px')
      .attr('font-weight', '500');
    axisG.selectAll('line').attr('stroke', '#ffffff').attr('opacity', 0.6);
    axisG.selectAll('path').attr('stroke', '#ffffff').attr('opacity', 0.6);

    // Mouse hover
    svg.on('mousemove', function(event) {
      const [mx, my] = d3.pointer(event, g.node());
      if (mx >= xOffset && mx <= xOffset + plotSize && my >= yOffset && my <= yOffset + plotSize) {
        const a = aScale.invert(mx - xOffset);
        const b = bScale.invert(my - yOffset);
        const mse = computeMSE(currentDataPoints, a, b);
        document.getElementById('landscapeMouseA').textContent = a.toFixed(3);
        document.getElementById('landscapeMouseB').textContent = b.toFixed(3);
        document.getElementById('landscapeMouseMSE').textContent = mse.toFixed(4);
      }
    });
    svg.on('mouseleave', () => {
      document.getElementById('landscapeMouseA').textContent = '—';
      document.getElementById('landscapeMouseB').textContent = '—';
      document.getElementById('landscapeMouseMSE').textContent = '—';
    });
  }

  window.addEventListener('dataUpdated', render);
  render(); // Initial render after deferred init
}

// ============================================================================
// Section 4b: Single GD Step Visualization
// ============================================================================
function initSingleStepViz() {
  const width = 360, height = 320;
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svgBefore = d3.select('#viz-step-before');
  const svgLandscape = d3.select('#viz-step-landscape');
  const svgAfter = d3.select('#viz-step-after');

  svgBefore.attr('viewBox', `0 0 ${width} ${height}`);
  svgLandscape.attr('viewBox', `0 0 ${width} ${height}`);
  svgAfter.attr('viewBox', `0 0 ${width} ${height}`);

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  function renderLandscape(a0, b0, a1, b1, alpha) {
    svgLandscape.selectAll('*').remove();
    
    // Arrow marker (add to defs FIRST before any elements that use it)
    let defs = svgLandscape.select('defs');
    if (defs.empty()) {
      defs = svgLandscape.append('defs');
    }
    defs.selectAll('#arrow-step').remove();
    defs.append('marker')
      .attr('id', 'arrow-step').attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#ffff00');
    
    const g = svgLandscape.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Use the reusable renderErrorLandscape function
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, 20);
    const { xOffset, yOffset, plotSize, aScale, bScale } = landscapeData;
    
    // Recompute gradient using the same function as "Evaluating the Gradient" section
    // This ensures consistency: both sections use computeGradient(currentDataPoints, a, b)
    const { da, db } = computeGradient(currentDataPoints, a0, b0);
    
    // Compute the gradient descent step: new position = old position - learning_rate * gradient
    // This is the actual step taken in gradient descent, scaled by alpha (learning rate)
    const computedA1 = a0 - alpha * da;
    const computedB1 = b0 - alpha * db;
    // Use computed values instead of passed-in values to ensure consistency
    const finalA1 = computedA1;
    const finalB1 = computedB1;

    // Arrow from (a0,b0) to (finalA1,finalB1) - using recomputed values
    const x0 = xOffset + aScale(a0), y0 = yOffset + bScale(b0);
    const x1 = xOffset + aScale(finalA1), y1 = yOffset + bScale(finalB1);
    g.append('line')
      .attr('x1', x0).attr('y1', y0)
      .attr('x2', x1).attr('y2', y1)
      .attr('stroke', '#ffff00').attr('stroke-width', 3)
      .attr('marker-end', 'url(#arrow-step)');

    // Points
    g.append('circle').attr('cx', x0).attr('cy', y0).attr('r', 5)
      .attr('fill', '#ff00ff').attr('stroke', '#fff').attr('stroke-width', 2);
    g.append('circle').attr('cx', x1).attr('cy', y1).attr('r', 5)
      .attr('fill', '#00ff00').attr('stroke', '#fff').attr('stroke-width', 2);

    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '12px').text('Gradient Step on Landscape');
    
    // Return computed values for consistency
    return { a1: finalA1, b1: finalB1 };
  }

  function render() {
    if (currentDataPoints.length === 0) return;
    const a0 = parseFloat(document.getElementById('stepA0').value);
    const b0 = parseFloat(document.getElementById('stepB0').value);
    const alpha = parseFloat(document.getElementById('stepAlpha').value);

    document.getElementById('stepA0Value').textContent = a0.toFixed(2);
    document.getElementById('stepB0Value').textContent = b0.toFixed(2);
    document.getElementById('stepAlphaValue').textContent = alpha.toFixed(2);

    renderDataPlot({
      svg: svgBefore,
      width,
      height,
      margin,
      dataPoints: currentDataPoints,
      modelA: a0,
      modelB: b0,
      showTruthLine: true,
      showModelLine: true,
      showMSE: true,
      title: 'Before: Model at iteration t',
      axisTicks: 5,
      showGrid: false,
      showZeroLines: false
    });

    // Compute gradient and new position - pass to renderLandscape which will recompute for consistency
    const { da, db } = computeGradient(currentDataPoints, a0, b0);
    const a1 = a0 - alpha * da;
    const b1 = b0 - alpha * db;
    const result = renderLandscape(a0, b0, a1, b1, alpha);
    // Use values returned from renderLandscape to ensure consistency
    const finalA1 = result.a1;
    const finalB1 = result.b1;

    renderDataPlot({
      svg: svgAfter,
      width,
      height,
      margin,
      dataPoints: currentDataPoints,
      modelA: finalA1,
      modelB: finalB1,
      showTruthLine: true,
      showModelLine: true,
      showMSE: true,
      title: 'After: Model at iteration t+1',
      axisTicks: 5,
      showGrid: false,
      showZeroLines: false
    });
  }

  document.getElementById('stepA0').addEventListener('input', () => {
    document.getElementById('stepA0Value').textContent = parseFloat(document.getElementById('stepA0').value).toFixed(2);
    render();
  });
  document.getElementById('stepB0').addEventListener('input', () => {
    document.getElementById('stepB0Value').textContent = parseFloat(document.getElementById('stepB0').value).toFixed(2);
    render();
  });
  document.getElementById('stepAlpha').addEventListener('input', () => {
    document.getElementById('stepAlphaValue').textContent = parseFloat(document.getElementById('stepAlpha').value).toFixed(2);
    render();
  });
  window.addEventListener('dataUpdated', render);
  render(); // Initial render after deferred init
}

// ============================================================================
// Section 4c: Mini-Batch Landscapes Visualization
// ============================================================================
function initBatchLandscapesViz() {
  const width = 360, height = 320;
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  function renderBatchLandscape(svg, batchPoints, a0, b0, batchNum, batchIndices) {
    svg.selectAll('*').remove();
    
    // Arrow marker (add to defs FIRST)
    const markerId = 'arrow-batch-' + batchNum;
    let defs = svg.select('defs');
    if (defs.empty()) {
      defs = svg.append('defs');
    }
    if (defs.select('#' + markerId).empty()) {
      defs.append('marker')
        .attr('id', markerId)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', '#ffff00');
    }
    
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Make plot area square
    const plotSize = Math.min(w, h);
    const xOffset = (w - plotSize) / 2;
    const yOffset = (h - plotSize) / 2;
    const aScale = d3.scaleLinear().domain([aMin, aMax]).range([0, plotSize]);
    const bScale = d3.scaleLinear().domain([bMin, bMax]).range([plotSize, 0]);

    // Compute error landscape for this batch - use batchPoints directly
    const meshPts = 15;
    const aArr = d3.range(aMin, aMax, (aMax - aMin) / meshPts);
    const bArr = d3.range(bMin, bMax, (bMax - bMin) / meshPts);
    let minErr = Infinity, maxErr = -Infinity;
    const errors = [];
    for (const b of bArr) {
      const row = [];
      for (const a of aArr) {
        const e = computeMSE(batchPoints, a, b);
        row.push(e);
        minErr = Math.min(minErr, e);
        maxErr = Math.max(maxErr, e);
      }
      errors.push(row);
    }

    const colorScale = d3.scaleSequential(d3.interpolateReds).domain([minErr, maxErr]);
    for (let i = 0; i < bArr.length; i++) {
      for (let j = 0; j < aArr.length; j++) {
        g.append('rect')
          .attr('x', xOffset + aScale(aArr[j]) - (plotSize / aArr.length) / 2)
          .attr('y', yOffset + bScale(bArr[i]) - (plotSize / bArr.length) / 2)
          .attr('width', plotSize / aArr.length)
          .attr('height', plotSize / bArr.length)
          .attr('fill', colorScale(errors[i][j]))
          .attr('opacity', 0.7);
      }
    }

    // Add contour lines - using batchPoints MSE, adjust scales with offset
    const aScaleOffset = d3.scaleLinear().domain([aMin, aMax]).range([xOffset, xOffset + plotSize]);
    const bScaleOffset = d3.scaleLinear().domain([bMin, bMax]).range([yOffset + plotSize, yOffset]);
    drawContourLines(g, errors, aArr, bArr, aScaleOffset, bScaleOffset, minErr, maxErr, 8);

    g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${plotSize + yOffset})`)
      .call(d3.axisBottom(aScale).ticks(5));
    g.append('g').attr('class', 'axis').attr('transform', `translate(${xOffset},${yOffset})`)
      .call(d3.axisLeft(bScale).ticks(5));

    // Compute gradient for this batch (batchPoints is already a subset, so compute on all of them)
    const { da, db } = computeGradient(batchPoints, a0, b0);
    const arrowLen = 0.08;
    const x0 = xOffset + aScale(a0), y0 = yOffset + bScale(b0);
    const x1 = xOffset + aScale(a0 - arrowLen * da), y1 = yOffset + bScale(b0 - arrowLen * db);

    g.append('line')
      .attr('x1', x0).attr('y1', y0)
      .attr('x2', x1).attr('y2', y1)
      .attr('stroke', '#ffff00').attr('stroke-width', 2.5)
      .attr('marker-end', 'url(#' + markerId + ')');

    g.append('circle').attr('cx', x0).attr('cy', y0).attr('r', 5)
      .attr('fill', '#ff00ff').attr('stroke', '#fff').attr('stroke-width', 2);

    // Title with batch indices
    const idxText = batchIndices.join(', ');
    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '12px')
      .text(`Batch ${batchNum}: J=[${idxText}]`);
  }

  let batch1Indices = [];
  let batch2Indices = [];
  let batch3Indices = [];

  function render() {
    if (currentDataPoints.length === 0) return;
    const a0 = parseFloat(document.getElementById('batchA0').value);
    const b0 = parseFloat(document.getElementById('batchB0').value);
    const batchSize = parseInt(document.getElementById('batchSizeViz').value, 10);

    document.getElementById('batchA0Value').textContent = a0.toFixed(2);
    document.getElementById('batchB0Value').textContent = b0.toFixed(2);
    document.getElementById('batchSizeValue').textContent = String(batchSize);

    // Generate 3 random batches - only regenerate indices on button click or data change
    const N = currentDataPoints.length;
    const batch1 = batch1Indices.map(i => currentDataPoints[i]);
    const batch2 = batch2Indices.map(i => currentDataPoints[i]);
    const batch3 = batch3Indices.map(i => currentDataPoints[i]);

    renderBatchLandscape(d3.select('#viz-batch-1'), batch1, a0, b0, 1, batch1Indices);
    renderBatchLandscape(d3.select('#viz-batch-2'), batch2, a0, b0, 2, batch2Indices);
    renderBatchLandscape(d3.select('#viz-batch-3'), batch3, a0, b0, 3, batch3Indices);
  }

  function updateBatchSizeSlider() {
    const N = currentDataPoints.length;
    const slider = document.getElementById('batchSizeViz');
    const currentValue = parseInt(slider.value, 10);
    
    // Update max to N
    slider.setAttribute('max', N);
    
    // Ensure current value doesn't exceed N
    if (currentValue > N) {
      slider.value = N;
      document.getElementById('batchSizeValue').textContent = String(N);
    }
  }

  function regenerateBatches() {
    const N = currentDataPoints.length;
    const batchSize = parseInt(document.getElementById('batchSizeViz').value, 10);
    batch1Indices = d3.shuffle(d3.range(N).slice()).slice(0, Math.min(batchSize, N));
    batch2Indices = d3.shuffle(d3.range(N).slice()).slice(0, Math.min(batchSize, N));
    batch3Indices = d3.shuffle(d3.range(N).slice()).slice(0, Math.min(batchSize, N));
    render();
  }

  document.getElementById('batchA0').addEventListener('input', () => {
    document.getElementById('batchA0Value').textContent = parseFloat(document.getElementById('batchA0').value).toFixed(2);
    render();
  });
  document.getElementById('batchB0').addEventListener('input', () => {
    document.getElementById('batchB0Value').textContent = parseFloat(document.getElementById('batchB0').value).toFixed(2);
    render();
  });
  document.getElementById('batchSizeViz').addEventListener('input', () => {
    document.getElementById('batchSizeValue').textContent = String(parseInt(document.getElementById('batchSizeViz').value, 10));
    regenerateBatches();
  });
  document.getElementById('batchUpdateBtn').addEventListener('click', regenerateBatches);
  window.addEventListener('dataUpdated', () => {
    updateBatchSizeSlider();
    regenerateBatches();
  });
  updateBatchSizeSlider();
  regenerateBatches(); // Initial render after deferred init
}

// ============================================================================
// Section 5: Gradient Descent - State Management
// ============================================================================
// State for Gradient Descent visualization
let gdTrajectory = [];
let gdCurrent = { a: 0.5, b: 0.5 };
let gdRunning = false;

function initGDViz() {
  const svg = d3.select('#viz-gd');
  const width = 760, height = 520;
  const margin = { top: 30, right: 80, bottom: 50, left: 60 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;
  
  // Helper to generate random starting position within valid ranges
  function randomStartPosition() {
    return {
      a: aMin + Math.random() * (aMax - aMin),
      b: bMin + Math.random() * (bMax - bMin)
    };
  }

  function render() {
    if (currentDataPoints.length === 0) return;
    g.selectAll('*').remove();

    // Render landscape and get scale info
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, 25);
    
    // Draw trajectory and current point
    drawTrajectoryOnLandscape(g, gdTrajectory, gdCurrent, landscapeData);
  }

  function step() {
    const alpha = parseFloat(document.getElementById('gdAlpha').value);
    const { da, db } = computeGradient(currentDataPoints, gdCurrent.a, gdCurrent.b);
    gdCurrent.a -= alpha * da;
    gdCurrent.b -= alpha * db;
    gdTrajectory.push({ a: gdCurrent.a, b: gdCurrent.b });

    const mse = computeMSE(currentDataPoints, gdCurrent.a, gdCurrent.b);

    gdHistory.push({ mse, a: gdCurrent.a, b: gdCurrent.b });
    window.dispatchEvent(new Event('gdUpdated'));

    document.getElementById('gdMseValue').textContent = mse.toFixed(4);
    document.getElementById('gdCurrentA').textContent = gdCurrent.a.toFixed(3);
    document.getElementById('gdCurrentB').textContent = gdCurrent.b.toFixed(3);
    render();
  }

  document.getElementById('gdStepBtn').addEventListener('click', step);
  document.getElementById('gdRunBtn').addEventListener('click', () => {
    if (gdRunning) {
      gdRunning = false;
      return;
    }
    gdRunning = true;
    const iters = parseInt(document.getElementById('gdIterations').value, 10);
    let count = 0;
    const interval = setInterval(() => {
      if (count >= iters || !gdRunning) {
        clearInterval(interval);
        gdRunning = false;
        return;
      }
      step();
      count++;
    }, ANIMATION_INTERVAL);
  });

  document.getElementById('gdAlpha').addEventListener('input', () => {
    document.getElementById('gdAlphaValue').textContent = parseFloat(document.getElementById('gdAlpha').value).toFixed(3);
  });
  document.getElementById('gdIterations').addEventListener('input', () => {
    document.getElementById('gdIterValue').textContent = String(parseInt(document.getElementById('gdIterations').value, 10));
  });
  document.getElementById('gdA0').addEventListener('input', () => {
    document.getElementById('gdA0Value').textContent = parseFloat(document.getElementById('gdA0').value).toFixed(2);
    if (!gdRunning) {
      gdCurrent.a = parseFloat(document.getElementById('gdA0').value);
      gdTrajectory = [{ a: gdCurrent.a, b: gdCurrent.b }];
      render();
    }
  });
  document.getElementById('gdB0').addEventListener('input', () => {
    document.getElementById('gdB0Value').textContent = parseFloat(document.getElementById('gdB0').value).toFixed(2);
    if (!gdRunning) {
      gdCurrent.b = parseFloat(document.getElementById('gdB0').value);
      gdTrajectory = [{ a: gdCurrent.a, b: gdCurrent.b }];
      render();
    }
  });

  document.getElementById('gdResetBtn').addEventListener('click', () => {
    gdRunning = false;
    // Generate new random starting position
    const startPos = randomStartPosition();
    gdCurrent.a = startPos.a;
    gdCurrent.b = startPos.b;
    // Update sliders to match
    document.getElementById('gdA0').value = startPos.a;
    document.getElementById('gdB0').value = startPos.b;
    document.getElementById('gdA0Value').textContent = startPos.a.toFixed(2);
    document.getElementById('gdB0Value').textContent = startPos.b.toFixed(2);
    gdTrajectory = [{ a: gdCurrent.a, b: gdCurrent.b }];
    gdHistory = [];
    window.dispatchEvent(new Event('gdUpdated'));
    render();
  });

  window.addEventListener('dataUpdated', () => {
    gdCurrent.a = parseFloat(document.getElementById('gdA0').value);
    gdCurrent.b = parseFloat(document.getElementById('gdB0').value);
    gdTrajectory = [{ a: gdCurrent.a, b: gdCurrent.b }];
    gdHistory = [];
    window.dispatchEvent(new Event('gdUpdated'));
    render();
  });
  
  // Initialize with random starting position
  const initialPos = randomStartPosition();
  gdCurrent.a = initialPos.a;
  gdCurrent.b = initialPos.b;
  document.getElementById('gdA0').value = initialPos.a;
  document.getElementById('gdB0').value = initialPos.b;
  document.getElementById('gdA0Value').textContent = initialPos.a.toFixed(2);
  document.getElementById('gdB0Value').textContent = initialPos.b.toFixed(2);
  gdTrajectory = [{ a: gdCurrent.a, b: gdCurrent.b }];
  render(); // Initial render after deferred init
}

// ============================================================================
// Section 6: Stochastic Gradient Descent - State Management
// ============================================================================
// State for SGD visualization
let sgdTrajectory = [];
let sgdCurrent = { a: 0.4, b: 0.5 };
let sgdRunning = false;

function initSGDViz() {
  const svg = d3.select('#viz-sgd');
  const width = 760, height = 520;
  const margin = { top: 30, right: 80, bottom: 50, left: 60 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  // Helper to generate random starting position within valid ranges
  function randomStartPosition() {
    return {
      a: aMin + Math.random() * (aMax - aMin),
      b: bMin + Math.random() * (bMax - bMin)
    };
  }

  function render() {
    if (currentDataPoints.length === 0) return;
    g.selectAll('*').remove();
    
    // Render landscape and get scale info
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, 25);
    
    // Draw trajectory and current point
    drawTrajectoryOnLandscape(g, sgdTrajectory, sgdCurrent, landscapeData);
  }

  function step() {
    const alpha = parseFloat(document.getElementById('sgdAlpha').value);
    const batchSize = parseInt(document.getElementById('sgdBatch').value, 10);
    const { da, db } = computeGradientBatch(currentDataPoints, sgdCurrent.a, sgdCurrent.b, batchSize);
    sgdCurrent.a -= alpha * da;
    sgdCurrent.b -= alpha * db;
    sgdTrajectory.push({ a: sgdCurrent.a, b: sgdCurrent.b });

    const mse = computeMSE(currentDataPoints, sgdCurrent.a, sgdCurrent.b);

    sgdHistory.push({ mse, a: sgdCurrent.a, b: sgdCurrent.b });
    window.dispatchEvent(new Event('sgdUpdated'));

    document.getElementById('sgdMseValue').textContent = mse.toFixed(4);
    document.getElementById('sgdCurrentA').textContent = sgdCurrent.a.toFixed(3);
    document.getElementById('sgdCurrentB').textContent = sgdCurrent.b.toFixed(3);
    render();
  }

  document.getElementById('sgdStepBtn').addEventListener('click', step);
  document.getElementById('sgdRunBtn').addEventListener('click', () => {
    if (sgdRunning) {
      sgdRunning = false;
      return;
    }
    sgdRunning = true;
    const iters = parseInt(document.getElementById('sgdIterations').value, 10);
    let count = 0;
    const interval = setInterval(() => {
      if (count >= iters || !sgdRunning) {
        clearInterval(interval);
        sgdRunning = false;
        return;
      }
      step();
      count++;
    }, ANIMATION_INTERVAL);
  });
  document.getElementById('sgdAlpha').addEventListener('input', () => {
    document.getElementById('sgdAlphaValue').textContent = parseFloat(document.getElementById('sgdAlpha').value).toFixed(3);
  });
  document.getElementById('sgdBatch').addEventListener('input', () => {
    document.getElementById('sgdBatchValue').textContent = String(parseInt(document.getElementById('sgdBatch').value, 10));
  });
  document.getElementById('sgdIterations').addEventListener('input', () => {
    document.getElementById('sgdIterValue').textContent = String(parseInt(document.getElementById('sgdIterations').value, 10));
  });
  document.getElementById('sgdA0').addEventListener('input', () => {
    document.getElementById('sgdA0Value').textContent = parseFloat(document.getElementById('sgdA0').value).toFixed(2);
    if (!sgdRunning) {
      sgdCurrent.a = parseFloat(document.getElementById('sgdA0').value);
      sgdTrajectory = [{ a: sgdCurrent.a, b: sgdCurrent.b }];
      render();
    }
  });
  document.getElementById('sgdB0').addEventListener('input', () => {
    document.getElementById('sgdB0Value').textContent = parseFloat(document.getElementById('sgdB0').value).toFixed(2);
    if (!sgdRunning) {
      sgdCurrent.b = parseFloat(document.getElementById('sgdB0').value);
      sgdTrajectory = [{ a: sgdCurrent.a, b: sgdCurrent.b }];
      render();
    }
  });

  document.getElementById('sgdResetBtn').addEventListener('click', () => {
    sgdRunning = false;
    // Generate new random starting position
    const startPos = randomStartPosition();
    sgdCurrent.a = startPos.a;
    sgdCurrent.b = startPos.b;
    // Update sliders to match
    document.getElementById('sgdA0').value = startPos.a;
    document.getElementById('sgdB0').value = startPos.b;
    document.getElementById('sgdA0Value').textContent = startPos.a.toFixed(2);
    document.getElementById('sgdB0Value').textContent = startPos.b.toFixed(2);
    sgdTrajectory = [{ a: sgdCurrent.a, b: sgdCurrent.b }];
    sgdHistory = [];
    window.dispatchEvent(new Event('sgdUpdated'));
    render();
  });

  window.addEventListener('dataUpdated', () => {
    sgdCurrent.a = parseFloat(document.getElementById('sgdA0').value);
    sgdCurrent.b = parseFloat(document.getElementById('sgdB0').value);
    sgdTrajectory = [{ a: sgdCurrent.a, b: sgdCurrent.b }];
    sgdHistory = [];
    window.dispatchEvent(new Event('sgdUpdated'));
    render();
  });
  sgdCurrent.a = parseFloat(document.getElementById('sgdA0').value);
  sgdCurrent.b = parseFloat(document.getElementById('sgdB0').value);
  sgdTrajectory = [{ a: sgdCurrent.a, b: sgdCurrent.b }];
  render(); // Initial render after deferred init
}

// ============================================================================
// Section 7: Learning Curves - State and Reusable Visualization
// ============================================================================
// History tracking for all algorithms
let gdHistory = [];
let sgdHistory = [];
let asgdHistory = [];

// ASGD state (moved here for logical grouping with history)
let asgdTrajectory = [];
let asgdCurrent = { a: 0.25, b: 0.35 };
let asgdRunning = false;

// Reusable function to create learning curves visualization
function createLearningCurvesViz(svgIds, historyGetter, eventName, algorithmName) {
  const width = 360, height = 280;
  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svgMSE = d3.select(svgIds.mse);
  const svgA = d3.select(svgIds.a);
  const svgB = d3.select(svgIds.b);

  svgMSE.attr('viewBox', `0 0 ${width} ${height}`);
  svgA.attr('viewBox', `0 0 ${width} ${height}`);
  svgB.attr('viewBox', `0 0 ${width} ${height}`);

  const gMSE = svgMSE.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const gA = svgA.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const gB = svgB.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  function renderPlot(g, data, yAccessor, yLabel, title, yBounds = null) {
    g.selectAll('*').remove();

    if (!data || data.length === 0) {
      g.append('text').attr('x', w / 2).attr('y', h / 2)
        .attr('text-anchor', 'middle').attr('fill', '#c9d4e5').attr('font-size', '12px')
        .text(`Run ${algorithmName} to see curves`);
      return;
    }

    const maxLen = data.length;
    const xScale = d3.scaleLinear().domain([0, maxLen]).range([0, w]);

    // Use fixed bounds if provided, otherwise compute from data
    let yMin, yMax;
    if (yBounds) {
      yMin = yBounds.min;
      yMax = yBounds.max;
    } else {
      const allVals = data.map(yAccessor);
      yMin = d3.min(allVals) || 0;
      yMax = d3.max(allVals) || 1;
    }
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

    g.append('g').attr('class', 'grid').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(-h).tickFormat(''));
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-w).tickFormat(''));

    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(yScale).ticks(5));
    g.append('text').attr('x', w / 2).attr('y', h + 35).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').attr('font-size', '11px').text('Iteration');
    g.append('text').attr('x', -h / 2).attr('y', -35).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').attr('font-size', '11px')
      .attr('transform', `rotate(-90, -${h / 2}, -35)`).text(yLabel);
    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '13px').text(title);

    const lineGen = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(yAccessor(d)));

    const color = algorithmName === 'GD' ? '#ff00ff' : algorithmName === 'SGD' ? '#00aaff' : '#ffaa00';
    
    g.append('path')
      .datum(data)
      .attr('d', lineGen)
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('fill', 'none');

    // Legend
    g.append('line').attr('x1', 10).attr('x2', 35).attr('y1', 5).attr('y2', 5)
      .attr('stroke', color).attr('stroke-width', 2);
    g.append('text').attr('x', 40).attr('y', 9).attr('fill', '#c9d4e5').attr('font-size', '10px')
      .text(algorithmName);
  }

  function renderCurves() {
    const historyData = historyGetter();
    
    // MSE: fixed min at 0, max from data
    const mseMax = historyData.length > 0 ? d3.max(historyData, d => d.mse) || 1 : 1;
    renderPlot(gMSE, historyData, d => d.mse, 'MSE', 'Mean Squared Error', { min: 0, max: mseMax });
    
    // Parameter a: fixed bounds from PARAM_RANGES
    renderPlot(gA, historyData, d => d.a, 'a', 'Parameter a (slope)', { min: PARAM_RANGES.aMin, max: PARAM_RANGES.aMax });
    
    // Parameter b: fixed bounds from PARAM_RANGES
    renderPlot(gB, historyData, d => d.b, 'b', 'Parameter b (intercept)', { min: PARAM_RANGES.bMin, max: PARAM_RANGES.bMax });
  }

  window.addEventListener(eventName, renderCurves);
  renderCurves();
}

function initGDCurvesViz() {
  createLearningCurvesViz(
    { mse: '#viz-curves-gd-mse', a: '#viz-curves-gd-a', b: '#viz-curves-gd-b' },
    () => gdHistory,
    'gdUpdated',
    'GD'
  );
}

function initSGDCurvesViz() {
  createLearningCurvesViz(
    { mse: '#viz-curves-sgd-mse', a: '#viz-curves-sgd-a', b: '#viz-curves-sgd-b' },
    () => sgdHistory,
    'sgdUpdated',
    'SGD'
  );
}

function initASGDCurvesViz() {
  createLearningCurvesViz(
    { mse: '#viz-curves-asgd-mse', a: '#viz-curves-asgd-a', b: '#viz-curves-asgd-b' },
    () => asgdHistory,
    'asgdUpdated',
    'ASGD'
  );
}

// ============================================================================
// CONTOUR LINE RENDERING UTILITY
// ============================================================================

/**
 * Draws contour lines on an error landscape using marching squares algorithm.
 * Applies sqrt transformation for visually equidistant contour levels.
 * 
 * @param {d3.Selection} g - D3 selection for the SVG group element
 * @param {Array<Array<number>>} errors - 2D array of error values
 * @param {Array<number>} aArr - Array of a parameter values (x-axis)
 * @param {Array<number>} bArr - Array of b parameter values (y-axis)
 * @param {Function} aScale - D3 scale for a parameter
 * @param {Function} bScale - D3 scale for b parameter
 * @param {number} minErr - Minimum error value in the landscape
 * @param {number} maxErr - Maximum error value in the landscape
 * @param {number} [numLevels=20] - Number of contour levels to draw
 */
function drawContourLines(g, errors, aArr, bArr, aScale, bScale, minErr, maxErr, numLevels = 10) {
  const r = maxErr - minErr;
  const sqrtTransformed = errors.map(row => row.map(val => Math.sqrt((val - minErr + 1e-6) / r)));
  
  for (let level = 0; level < numLevels; level++) {
    const targetSqrt = Math.sqrt(level / (numLevels - 1));
    const paths = [];
    
    for (let i = 0; i < bArr.length - 1; i++) {
      for (let j = 0; j < aArr.length - 1; j++) {
        const v00 = sqrtTransformed[i][j];
        const v10 = sqrtTransformed[i][j + 1];
        const v01 = sqrtTransformed[i + 1][j];
        const v11 = sqrtTransformed[i + 1][j + 1];
        
        const edges = [];
        
        // Check edges for intersections
        // Top edge
        if ((v00 < targetSqrt && v10 >= targetSqrt) || (v00 >= targetSqrt && v10 < targetSqrt)) {
          const t = (targetSqrt - v00) / (v10 - v00);
          edges.push([aScale(aArr[j] + t * (aArr[j + 1] - aArr[j])), bScale(bArr[i])]);
        }
        // Right edge
        if ((v10 < targetSqrt && v11 >= targetSqrt) || (v10 >= targetSqrt && v11 < targetSqrt)) {
          const t = (targetSqrt - v10) / (v11 - v10);
          edges.push([aScale(aArr[j + 1]), bScale(bArr[i] + t * (bArr[i + 1] - bArr[i]))]);
        }
        // Bottom edge
        if ((v01 < targetSqrt && v11 >= targetSqrt) || (v01 >= targetSqrt && v11 < targetSqrt)) {
          const t = (targetSqrt - v01) / (v11 - v01);
          edges.push([aScale(aArr[j] + t * (aArr[j + 1] - aArr[j])), bScale(bArr[i + 1])]);
        }
        // Left edge
        if ((v00 < targetSqrt && v01 >= targetSqrt) || (v00 >= targetSqrt && v01 < targetSqrt)) {
          const t = (targetSqrt - v00) / (v01 - v00);
          edges.push([aScale(aArr[j]), bScale(bArr[i] + t * (bArr[i + 1] - bArr[i]))]);
        }
        
        if (edges.length === 2) {
          paths.push(edges);
        }
      }
    }
    
    paths.forEach(edgePair => {
      g.append('line')
        .attr('x1', edgePair[0][0])
        .attr('y1', edgePair[0][1])
        .attr('x2', edgePair[1][0])
        .attr('y2', edgePair[1][1])
        .attr('stroke', '#000000')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.6);
    });
  }
}

// ============================================================================
// Complex Loss Landscape (Local vs Global Minimum example)
// ============================================================================
function initComplexLossViz() {
  const svg = d3.select('#viz-complex-loss');
  const width = 840, height = 520;
  const margin = { top: 30, right: 100, bottom: 50, left: 60 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const p1Min = -5, p1Max = 7;
  const p2Min = -7, p2Max = 2;
  const meshPoints = 40;

  function computeLoss(p1, p2) {
    return 0.05 * Math.pow(p1, 4) - 2 * Math.pow(p1, 2) + p1 * p2 + 4 * p2 + Math.pow(p2, 2) + 51;
  }

  function render() {
    g.selectAll('*').remove();

    const p1Arr = d3.range(p1Min, p1Max, (p1Max - p1Min) / meshPoints);
    const p2Arr = d3.range(p2Min, p2Max, (p2Max - p2Min) / meshPoints);

    const losses = [];
    let minLoss = Infinity, maxLoss = -Infinity;
    for (const p2 of p2Arr) {
      const row = [];
      for (const p1 of p1Arr) {
        const loss = computeLoss(p1, p2);
        row.push(loss);
        minLoss = Math.min(minLoss, loss);
        maxLoss = Math.max(maxLoss, loss);
      }
      losses.push(row);
    }

    const p1Scale = d3.scaleLinear().domain([p1Min, p1Max]).range([0, w]);
    const p2Scale = d3.scaleLinear().domain([p2Min, p2Max]).range([h, 0]);

    // Grayscale colormap
    const colorScale = d3.scaleSequential(d3.interpolateGreys).domain([minLoss, maxLoss]);

    // Draw filled contours
    for (let i = 0; i < p2Arr.length; i++) {
      for (let j = 0; j < p1Arr.length; j++) {
        g.append('rect')
          .attr('x', p1Scale(p1Arr[j]) - (w / p1Arr.length) / 2)
          .attr('y', p2Scale(p2Arr[i]) - (h / p2Arr.length) / 2)
          .attr('width', w / p1Arr.length)
          .attr('height', h / p2Arr.length)
          .attr('fill', colorScale(losses[i][j]))
          .attr('opacity', 0.9);
      }
    }

    // Add contour lines
    drawContourLines(g, losses, p1Arr, p2Arr, p1Scale, p2Scale, minLoss, maxLoss, 10);

    // Grid
    g.append('g').attr('class', 'grid').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(p1Scale).ticks(10).tickSize(-h).tickFormat(''));
    g.append('g').attr('class', 'grid')
      .call(d3.axisLeft(p2Scale).ticks(10).tickSize(-w).tickFormat(''));

    // Axes
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(p1Scale).ticks(6));
    
    // Set y-ticks as in notebook: range(-6,4,2) -> [-6, -4, -2, 0, 2]
    const yTicks = [-6, -4, -2, 0, 2];
    g.append('g').attr('class', 'axis').call(d3.axisLeft(p2Scale).tickValues(yTicks));

    g.append('text').attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').text('p1');
    g.append('text').attr('x', -30).attr('y', h / 2).attr('text-anchor', 'middle')
      .attr('fill', '#c9d4e5').text('p2');

    // Title
    g.append('text').attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle')
      .attr('fill', '#ffffff').attr('font-size', '14px').text('Loss Landscape');

    // Colorbar
    const colorbarW = 20, colorbarH = h;
    const colorbarG = svg.append('g').attr('transform', `translate(${width - margin.right + 20},${margin.top})`);
    const colorGradient = colorbarG.append('defs').append('linearGradient')
      .attr('id', 'colorGradComplex').attr('x1', '0%').attr('y1', '100%').attr('x2', '0%').attr('y2', '0%');
    const nStops = 10;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const val = minLoss + t * (maxLoss - minLoss);
      colorGradient.append('stop')
        .attr('offset', `${i * 10}%`)
        .attr('stop-color', colorScale(val));
    }
    colorbarG.append('rect')
      .attr('x', 0).attr('y', 0).attr('width', colorbarW).attr('height', colorbarH)
      .style('fill', 'url(#colorGradComplex)');
  }

  render(); // Initial render after deferred init
}

// ============================================================================
// Section: Gradient Evaluation
// ============================================================================
function initGradientEvalViz() {
  const svg = d3.select('#viz-gradient-eval');
  const width = 760, height = 520;
  const margin = { top: 30, right: 80, bottom: 50, left: 60 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const { aMin, aMax, bMin, bMax } = PARAM_RANGES;

  function render() {
    if (currentDataPoints.length === 0) return;
    g.selectAll('*').remove();

    // Use the reusable renderErrorLandscape function
    const landscapeData = renderErrorLandscape(g, w, h, currentDataPoints, aMin, aMax, bMin, bMax, 20);
    const { xOffset, yOffset, plotSize, aScale, bScale } = landscapeData;

    const a = parseFloat(document.getElementById('gradEvalA').value);
    const b = parseFloat(document.getElementById('gradEvalB').value);
    // Use the same gradient computation as "Gradient Step on Landscape" section
    // Both sections use computeGradient(currentDataPoints, a, b) for consistency
    const { da, db } = computeGradient(currentDataPoints, a, b);
    
    // Show raw gradient (not negated) - this is the gradient direction, scaled by arrowLen for visualization

    document.getElementById('gradEvalAValue').textContent = isNaN(a) ? '0.00' : a.toFixed(2);
    document.getElementById('gradEvalBValue').textContent = isNaN(b) ? '0.00' : b.toFixed(2);
    document.getElementById('gradEvalDa').textContent = (isNaN(da) || !isFinite(da)) ? '0.000' : da.toFixed(3);
    document.getElementById('gradEvalDb').textContent = (isNaN(db) || !isFinite(db)) ? '0.000' : db.toFixed(3);

    // Draw gradient arrow (showing raw gradient direction)
    let defs = svg.select('defs');
    if (defs.empty()) {
      defs = svg.append('defs');
    }
    defs.selectAll('#arrow-grad-eval').remove();
    defs.append('marker')
      .attr('id', 'arrow-grad-eval')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#ffff00');

    if (!isNaN(a) && !isNaN(b) && isFinite(a) && isFinite(b) && 
        !isNaN(da) && !isNaN(db) && isFinite(da) && isFinite(db)) {
      const x0 = xOffset + aScale(a);
      const y0 = yOffset + bScale(b);
      const arrowLen = 1.0; // Use alpha = 1 (no scaling) to show actual gradient magnitude
      // Raw gradient points in direction of increase, so we show it as-is
      const x1 = xOffset + aScale(a + arrowLen * da);
      const y1 = yOffset + bScale(b + arrowLen * db);

      g.append('line')
        .attr('x1', x0).attr('y1', y0)
        .attr('x2', x1).attr('y2', y1)
        .attr('stroke', '#ffff00').attr('stroke-width', 3)
        .attr('marker-end', 'url(#arrow-grad-eval)');

      g.append('circle').attr('cx', x0).attr('cy', y0).attr('r', 6)
        .attr('fill', '#ff00ff').attr('stroke', '#fff').attr('stroke-width', 2);
    }
  }

  document.getElementById('gradEvalA').addEventListener('input', render);
  document.getElementById('gradEvalB').addEventListener('input', render);
  window.addEventListener('dataUpdated', render);
  render(); // Initial render after deferred init
}

// ============================================================================
// Initialize all sections
// ============================================================================
function init() {
  // Initialize light sections immediately (no expensive computations)
  initDataViz();
  initModelViz();
  initModelMinimizationViz();
  
  // Defer heavy computations to allow page to render
  // Use single deferred call instead of multiple setTimeout chains
  setTimeout(() => {
    initLandscapeViz();
    initGradientEvalViz();
    initSingleStepViz();
    initBatchLandscapesViz();
    initGDViz();
    initSGDViz();
    initAnnealedSGDViz();
    initGDCurvesViz();
    initSGDCurvesViz();
    initASGDCurvesViz();
    initComplexLossViz();
  }, 50);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
