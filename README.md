# 📈 Non-Linear Regression Analysis
*A course project for Non-Linear Regression (Aug 2025 – Nov 2025)*  

This repository contains a full implementation and analysis of **Non-Linear Least Squares (NLLS)** methods, focusing on parameter estimation using the **Gauss–Newton algorithm**. The project includes deriving iterative update formulas, implementing optimization routines from scratch, analyzing convergence behaviour, and evaluating model fit using error metrics.

---

## 🧠 Project Overview

Most real-world relationships between variables are **non-linear**, and classical linear regression assumptions do not hold.  
This project explores:

- Estimating parameters when the model is non-linear in parameters  
- How iterative methods like **Gauss–Newton** converge  
- How initial guesses affect convergence  
- How to analyze residual errors  
- Numerical issues like divergence and stability  

---

## 🚀 Features Implemented

### ✔️ 1. Gauss–Newton Algorithm (From Scratch)
- Analytical Jacobian computation  
- Parameter update rule:  
  ```math
  \theta_{k+1} = \theta_k - (J^\top J)^{-1} J^\top r
  ```
- RSS-based convergence criteria  
- Handling divergence and failed convergence  

### ✔️ 2. Residual Sum of Squares (RSS) Minimization
- Tracking RSS at every iteration  
- Tabulated convergence history  
- Visualization-ready structure  

### ✔️ 3. Iteration History Table
For every iteration:
- Parameter estimates  
- Residual error vector  
- RSS  
- Step size  

### ✔️ 4. Error Metrics
- Mean Squared Error (MSE)  
- Variance estimate ```math \hat{\sigma}^2 ```  
- Optional confidence intervals  

---

## 📊 Results

The Gauss–Newton implementation successfully converges for all tested models.  
You can include (after generating them):

- Parameter estimate tables  
- RSS vs. iteration plots  
- Residual plots  
- Model fitting visualizations  


---
