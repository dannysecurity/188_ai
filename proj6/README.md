# Project 6

This repository contains my completed implementation of Project 6 from UC Berkeley’s CS 188: Introduction to Artificial Intelligence. 
All model logic is implemented in `models.py`, using a neural network mini-library (`nn.py`) provided by the course.

---

## 🧠 Project Overview

In this project, I built several supervised learning models using a custom neural network backend. These include linear models (like the perceptron), fully connected neural networks for regression and classification, and recurrent networks for handling sequential data.

I used parameterized layers (via `nn.Parameter`) and composed differentiable operations (e.g., `nn.Linear`, `nn.Add`, `nn.ReLU`) to construct models of increasing complexity. All training was implemented manually using batched stochastic gradient descent with explicit calls to `nn.gradients` and `parameter.update`.

---

## ⚙️ Design Highlights

- **Modular training loop** with batched input, stopping based on convergence or validation metrics.
- **Custom architecture design** for regression, classification, and sequence models—tuned layer sizes, learning rates, and activation strategies.
- **Recurrent neural network** built from first principles to handle variable-length word inputs, encoding sequences into fixed-length representations.
- **Softmax and square loss functions** implemented for different tasks, using the provided `nn.SoftmaxLoss` and `nn.SquareLoss`.
