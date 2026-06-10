---
layout: post
title: "Hello, World"
date: 2026-06-10
tags: [meta]
mathjax: true
---

Starting a blog is one of those things that keeps getting pushed to "later". Well, here it is.

The plan is simple: write about things I find interesting. That will probably include software engineering, machine learning, and the occasional detour into whatever rabbit hole I've fallen into that week.

## Why bother?

Writing forces clarity. Explaining something to an imaginary reader reveals all the gaps in your own understanding. That's useful.

It's also a record. In five years it might be interesting — or embarrassing — to look back and see what I thought mattered.

## A note on notation

A lot of posts here will involve math. For example, the softmax function turns a vector of raw scores into a probability distribution:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Or gradient descent, which updates parameters $\theta$ by stepping in the direction of steepest descent of the loss $\mathcal{L}$:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}(\theta)$$

where $\eta$ is the learning rate. Inline math works too: the cross-entropy loss for a single example is $-\sum_k y_k \log \hat{y}_k$.

## What to expect

No promises on frequency. Posts will be long when the topic deserves it, and short when it doesn't.

Here's a quick taste of what code will look like:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
```

That's about it. Let's see how this goes.
