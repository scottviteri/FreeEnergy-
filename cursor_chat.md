# Project document review
_Exported on 2/11/2026 at 12:51:23 PST from Cursor (2.2.14)_

---

**User**

Can you please read through project.pdf

---

**Cursor**

I’ll locate `project.pdf` in the workspace first, then extract its text so I can read through it and report back clearly.

I found the file at `/home/scottviteri/Projects/FreeEnergy++/project.pdf`; next I’m extracting it to plain text so I can review and summarize it for you.

`pypdf` isn’t installed, so I’m checking for a native PDF text utility next (`pdftotext`) and will use that path if available.

Text extraction succeeded. I’m reading the extracted content now and will give you a concise walkthrough of what’s in the PDF.

Read through `project.pdf`. It lays out a research proposal for **RL-based curriculum selection** in language model training.

Key points:
- Core idea: instead of random training data sampling, the model **chooses which examples to train on** and gets reward from held-out improvement.
- Method: transformer produces a latent `z`, policy samples selection vector `h`, picks candidate `x` via softmax over embeddings, trains on `x`, and reward is improvement on held-out set. Uses a SAC-style objective with a Q-head.
- Evaluation focuses on **sample efficiency** (perplexity and downstream-task performance vs number of examples seen), compared against random/loss/uncertainty/competence curricula.
- Includes analysis plans for learned curriculum structure, exploration dynamics, and Q-function interpretability.
- Connects to your prior Markovian Transformers work and frames this as a step toward broader goals around agency/homeostasis in learning systems.
- Timeline/resources: about **6 months**, with initial cluster work plus roughly **$10K** for larger-scale cloud experiments.
- Expected outcomes: empirical result (positive or negative), open-source implementation, curriculum analysis, and foundation for longer-term work.

If you want, I can also give:
1) a critical review (strengths, risks, missing controls), or  
2) concrete edits to tighten this for a grant/application audience.

---

**User**

I would like to make a few edits to project.tex
First is that the variance should be learned, which means we want log sigma^2 = W_gamma z
Since S and D are streamed in, they should get subscripts
Maybe note for (7) that we will need to remember the previous log prob before updating, so we don't need to keep theta and theta' in memory
The Q function should be also a projection of an activation vector, but should be an activation vector over the last token of x in context (Q_theta(x))
Make it clear that theta and theta' are being updated in the inner loop, and S and D are updated in the outer loop -- each datapoint in S or D is conceptually a full context window of text for a transformer
Make it clear that we are training a policy gradient step, and using the Q function to compute the advantage
  So the step should be Advantage * Grad_theta pi_theta(x) 


---

**Cursor**

I’ll open `project.tex`, find the algorithm/equation sections, and apply your requested updates directly to notation and training-loop details. Then I’ll quickly sanity-check for lint/build issues.

I’ve identified all the spots to update. I’m now patching `project.tex` to (1) make variance learned via `W_\gamma z`, (2) use streamed-set subscripts, and (3) rewrite the update equations to explicitly show inner/outer loops plus the advantage-weighted policy gradient step.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Q-network $Q_\theta(x)$ computed from the last-token activation of $x$ in context
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    Let $u_k$ be the transformer activation at the last token of $x_k$ in context, and define
    \begin{align}
        q_k &= Q_{\theta_k}(x_k) = W_Q u_k \\
        \hat{Q}_{\theta_k}(x_k, h_k) &= h_k^\top q_k \\
        L_Q &= \left(\hat{Q}_{\theta_k}(x_k, h_k) - r_k\right)^2.
    \end{align}
    
    \item \textbf{Policy-gradient update (advantage-weighted):}
    \begin{align}
        A_k &= r_k - \hat{Q}_{\theta_k}(x_k, h_k) \\
        \nabla_{\theta_k} J_\pi &= A_k \nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, we compute a Q activation from the selected context window $x$: if $u_x$ is the last-token transformer activation, then $Q_\theta(x) = W_Q u_x$ and $\hat{Q}_\theta(x,h) = h^\top Q_\theta(x)$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Updated `project.tex` with all the edits you requested.

What I changed:
- Learned variance is now explicitly state-dependent:
  - `\log \sigma^2 = W_\gamma z`
  - Added a variance head in Architecture (`W_\gamma z`) instead of state-independent variance text.
- Streamed sets now use time subscripts:
  - `S_t`, `D_t` in Data Streaming and algorithm equations.
- Added the memory note for reward equation:
  - After `r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)`, I added text saying to cache `\log p_{\theta_k}(D_t)` pre-update so you don’t need both parameter versions in memory.
- Q-function is now defined from the last-token activation of selected context:
  - `u_k` = last-token activation of `x_k` in context,
  - `q_k = Q_{\theta_k}(x_k) = W_Q u_k`,
  - `\hat{Q}_{\theta_k}(x_k,h_k) = h_k^\top q_k`.
- Clarified inner vs outer loops:
  - Outer loop updates streamed sets `S_t, D_t`.
  - Inner loop updates model parameters (`\theta_k \to \theta_{k+1}`).
  - Also added that each element of `S_t`/`D_t` is a full transformer context window.
- Policy update is now explicit policy gradient using advantage from Q:
  - `A_k = r_k - \hat{Q}_{\theta_k}(x_k,h_k)`
  - `\nabla_{\theta_k} J_\pi = A_k \nabla_{\theta_k}\log \pi_{\theta_k}(x_k|h_k) + \beta \nabla_{\theta_k}\mathcal{H}[\pi_{\theta_k}]`

Also updated the Analysis section reference from `Q(z,h)` to `Q_\theta(x)` for consistency.

If you want, I can do one more pass to tighten notation so policy parameters and LM parameters are either fully shared or explicitly split (`\theta_\pi`, `\theta_{\text{LM}}`, `\theta_Q`) to avoid ambiguity.

---

**User**

W_Q should just output a number -- no need to include h_k, since h_k is what is used to produce x_k in the first place

Something is wrong with the advantage here. Can you help me out here?

---

**Cursor**

Good catch — let me think through this carefully.

**The problem with `A_k = r_k - \hat{Q}(x_k, h_k)`:** If Q is a good estimate of E[r|x], then `r - Q(x)` has zero mean *conditioned on x*. Using it as the advantage in a policy gradient yields an estimator whose expected gradient is **zero** — it subtracts an action-dependent quantity, which changes the expected gradient (to nothing).

**The fix:** In a standard actor-critic, Q(x) itself is the return estimate used in the policy gradient. The proper advantage requires a *state-dependent, action-independent* baseline V. The cleanest formulation:

- **Q** estimates the reward: `L_Q = (Q_θ(x_k) - r_k)²`  
- **Policy gradient** uses Q directly as the advantage signal: `∇J = Q_θ(x_k) · ∇ log π_θ(x_k|h_k)`  

(Optionally, for variance reduction, subtract a baseline `V_k = Σ_x π(x|h_k) Q_θ(x)`, but that requires evaluating Q on all candidates.)

Let me apply both fixes now — making Q a scalar output of `W_Q u_k` with no `h_k`, and fixing the policy gradient.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Q-network $Q_\theta(x)$ computed from the last-token activation of $x$ in context
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    Let $u_k$ be the transformer activation at the last token of $x_k$ in context.
    The Q-function maps this to a scalar reward estimate:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear in the Q-function: $h_k$ already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    The Q-function serves as the advantage signal in a standard actor-critic policy gradient:
    \begin{equation}
        \nabla_{\theta_k} J_\pi = Q_{\theta_k}(x_k)\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{equation}
    For additional variance reduction one could subtract a state-dependent baseline $V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x)$, replacing $Q_{\theta_k}(x_k)$ with the advantage $A_k = Q_{\theta_k}(x_k) - V_k$. This requires evaluating $Q$ on every candidate in $S_t$, which is feasible since $W_Q$ is a single linear projection.
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, we compute a Q activation from the selected context window $x$: if $u_x$ is the last-token transformer activation, then $Q_\theta(x) = W_Q u_x$ and $\hat{Q}_\theta(x,h) = h^\top Q_\theta(x)$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now update the Architecture section to match the new scalar Q definition:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Q-network $Q_\theta(x)$ computed from the last-token activation of $x$ in context
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    Let $u_k$ be the transformer activation at the last token of $x_k$ in context.
    The Q-function maps this to a scalar reward estimate:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear in the Q-function: $h_k$ already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    The Q-function serves as the advantage signal in a standard actor-critic policy gradient:
    \begin{equation}
        \nabla_{\theta_k} J_\pi = Q_{\theta_k}(x_k)\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{equation}
    For additional variance reduction one could subtract a state-dependent baseline $V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x)$, replacing $Q_{\theta_k}(x_k)$ with the advantage $A_k = Q_{\theta_k}(x_k) - V_k$. This requires evaluating $Q$ on every candidate in $S_t$, which is feasible since $W_Q$ is a single linear projection.
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a single linear projection of the last-token transformer activation of the selected context window $x$: $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$. Because $h$ already determined the choice of $x$, the Q-function need only assess $x$ itself.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Also update the Setup bullet for the Q-network to be consistent:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    Let $u_k$ be the transformer activation at the last token of $x_k$ in context.
    The Q-function maps this to a scalar reward estimate:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear in the Q-function: $h_k$ already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    The Q-function serves as the advantage signal in a standard actor-critic policy gradient:
    \begin{equation}
        \nabla_{\theta_k} J_\pi = Q_{\theta_k}(x_k)\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{equation}
    For additional variance reduction one could subtract a state-dependent baseline $V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x)$, replacing $Q_{\theta_k}(x_k)$ with the advantage $A_k = Q_{\theta_k}(x_k) - V_k$. This requires evaluating $Q$ on every candidate in $S_t$, which is feasible since $W_Q$ is a single linear projection.
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a single linear projection of the last-token transformer activation of the selected context window $x$: $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$. Because $h$ already determined the choice of $x$, the Q-function need only assess $x$ itself.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Let me verify the final file reads cleanly:

Everything looks consistent. Let me clean up the extracted text file I created earlier:

Here's a summary of what changed:

### 1. Q outputs a scalar — no `h_k` dependency
`Q_{\theta_k}(x_k) = W_Q u_k \in \mathbb{R}`, where `u_k` is the last-token activation. Since `h_k` already determined the choice of `x_k`, Q only needs to assess the example itself.

### 2. Fixed the advantage / policy gradient
The old formulation had `A_k = r_k - \hat{Q}(x_k, h_k)`, which is the **Q-prediction error**. That's problematic because if Q is a good estimate of E[r|x], then `r - Q(x)` has zero mean *conditioned on the action* — so multiplying it by `∇ log π` gives an expected gradient of **zero**.

The fix: Q itself is the return estimate in a standard actor-critic:

$$\nabla_\theta J = Q_\theta(x_k)\;\nabla_\theta \log \pi_\theta(x_k \mid h_k) + \beta\,\nabla_\theta \mathcal{H}[\pi_\theta]$$

I also added a note that for variance reduction you can optionally subtract a proper **state-dependent, action-independent** baseline `V_k = Σ_x π(x|h_k) Q_θ(x)` to get a true advantage `A_k = Q(x_k) - V_k`. This is feasible since `W_Q` is just a linear projection over the candidate set.

### 3. Architecture section updated
Removed the bilinear `h^\top Q_\theta(x)` definition, now simply: `Q_\theta(x) = W_Q u_x ∈ ℝ`.

---

**User**

So is this not the same as normal Q functions? This seems more like a baseline

In this setup, the current weights are the state

---

**Cursor**

You're absolutely right, and my language was sloppy. Let me clarify the conceptual picture:

- **State** = current weights θ_k (implicitly encoded because the transformer parameterized by θ_k produces u_k)
- **Action** = which example x_k to train on
- **Q_θ(x_k) = W_Q u_k** is a proper **state-action value function** Q(s, a) — the state dependence comes through θ_k governing how u_k is computed, and the action is x_k

So Q_θ(x_k) is a standard Q-function, not an advantage. Using it raw in the policy gradient works but has higher variance. The proper advantage is A = Q(s,a) − V(s), where V is the state-dependent baseline. Let me fix the terminology in the paper.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is therefore a standard state-action value function $Q(s,a)$, implemented as a linear projection of the last-token activation:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by averaging $Q$ over the candidate set under the current policy:
    \begin{equation}
        V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x).
    \end{equation}
    This is feasible because $W_Q$ is a single linear projection applied to the (already computed or cheaply obtainable) last-token activations of each candidate.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a single linear projection of the last-token transformer activation of the selected context window $x$: $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$. Because $h$ already determined the choice of $x$, the Q-function need only assess $x$ itself.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now let me also update the Architecture section to make the state/action framing explicit there too:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is therefore a standard state-action value function $Q(s,a)$, implemented as a linear projection of the last-token activation:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by averaging $Q$ over the candidate set under the current policy:
    \begin{equation}
        V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x).
    \end{equation}
    This is feasible because $W_Q$ is a single linear projection applied to the (already computed or cheaply obtainable) last-token activations of each candidate.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V = \sum_x \pi(x|h)\,Q_\theta(x)$ yields the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Let me verify the final result reads coherently:

That reads cleanly. Here's what changed and why:

**Q is a standard Q-function, not an advantage.** The state is θ_k (encoded implicitly — the transformer parameterized by θ_k produces u_k), and the action is x_k. So `Q_θ(x_k) = W_Q u_k` is a proper Q(s, a).

**V is the baseline, A is the advantage.** The state-value baseline averages Q over the action distribution:

$$V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\, Q_{\theta_k}(x)$$

Then the advantage is the standard form:

$$A_k = Q_{\theta_k}(x_k) - V_k$$

And the policy gradient uses the advantage:

$$\nabla J = A_k \nabla \log \pi_{\theta_k}(x_k \mid h_k) + \beta\, \nabla \mathcal{H}[\pi]$$

Computing V_k over all candidates in S_t is feasible since `W_Q` is just a single linear projection of last-token activations.

---

**User**

But if L_Q is part of the loss, then Q is just learning to a be a reward, as opposed to a discounted future expected reward given the policy and starting state and action 

---

**Cursor**

You're completely right. As written, `L_Q = (Q(x_k) - r_k)^2` just trains Q to predict the immediate reward — it's a reward predictor, not a proper Q-function. A real Q-function should estimate the **discounted future expected return** from taking action x_k in state θ_k and then following the policy:

$$Q(s_k, a_k) = \mathbb{E}\!\left[r_k + \gamma r_{k+1} + \gamma^2 r_{k+2} + \cdots\right]$$

The fix is to use a proper Bellman target: `y_k = r_k + γ V_{k+1}`. Let me update the paper.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$; discount factor $\gamma$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is therefore a standard state-action value function $Q(s,a)$, implemented as a linear projection of the last-token activation:
    \begin{align}
        Q_{\theta_k}(x_k) &= W_Q\, u_k \;\in\; \mathbb{R} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - r_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by averaging $Q$ over the candidate set under the current policy:
    \begin{equation}
        V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x).
    \end{equation}
    This is feasible because $W_Q$ is a single linear projection applied to the (already computed or cheaply obtainable) last-token activations of each candidate.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V = \sum_x \pi(x|h)\,Q_\theta(x)$ yields the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now fix the Q-function update to use a Bellman target:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Entropy coefficient $\beta$; discount factor $\gamma$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is a standard state-action value function $Q(s,a)$, estimating the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    It is implemented as a linear projection of the last-token activation:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    We train $Q$ via the Bellman equation.  After the language-modeling update produces $\theta_{k+1}$, we compute the next-state value:
    \begin{equation}
        V_{k+1} = \sum_{x \in S_t} \pi_{\theta_{k+1}}(x \mid h_{k+1})\,Q_{\bar{\theta}}(x),
    \end{equation}
    where $\bar{\theta}$ denotes a slow-moving (target-network) copy of the Q-weights for stability.
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\, V_{k+1} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - y_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear in $Q$: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by averaging $Q$ over the candidate set under the current policy:
    \begin{equation}
        V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x).
    \end{equation}
    This is feasible because $W_Q$ is a single linear projection applied to the (already computed or cheaply obtainable) last-token activations of each candidate.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V = \sum_x \pi(x|h)\,Q_\theta(x)$ yields the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now let me verify the full algorithm section reads coherently end to end:

Looks good. The whole algorithm section is now internally consistent. Here's the conceptual fix:

**Before:** `L_Q = (Q_θ(x_k) - r_k)²` — this just trains Q to predict the *immediate* reward. That's a reward model, not a Q-function.

**After:** Q is trained via the **Bellman equation**:

$$y_k = r_k + \gamma\, V_{k+1}, \qquad L_Q = (Q_{\theta_k}(x_k) - y_k)^2$$

where the next-state value bootstraps through the policy:

$$V_{k+1} = \sum_{x \in S_t} \pi_{\theta_{k+1}}(x \mid h_{k+1})\, Q_{\bar\theta}(x)$$

This way Q learns the **discounted future expected return** — not just "how much reward does x_k give right now," but "how much cumulative reward flows from selecting x_k and then following the policy from θ_{k+1} onward." This captures strategic choices: an example that gives low immediate reward but positions the model well for future learning will still get high Q.

I also added γ to the Setup section, and used a target-network copy $\bar\theta$ in the bootstrap for training stability.

---

**User**

Let's make it more clear that we are sampling several next data points for x so we can compute the expectation to calculate V
Also remove the stability stuff (theta hat). Maybe instead you would like to be explicit about detaching theta, but not the weights of W_Q in the L_Q part of the objective?
I don't think we need the entropy part of the objective

---

**Cursor**

Now let me also check the Setup and Architecture sections for `\beta` references to clean up:

Good — two spots to clean up: the Setup bullet (remove β, keep γ) and the policy gradient (remove entropy term). Let me also check the Architecture section:

Now I'll apply all three changes: (1) rewrite Q-function update to sample multiple candidates and use `\operatorname{sg}` for detach, remove target network; (2) remove entropy from policy gradient; (3) remove β from setup.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is a standard state-action value function $Q(s,a)$, estimating the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    It is implemented as a linear projection of the last-token activation:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    We train $Q$ via the Bellman equation.  After the language-modeling update produces $\theta_{k+1}$, we compute the next-state value:
    \begin{equation}
        V_{k+1} = \sum_{x \in S_t} \pi_{\theta_{k+1}}(x \mid h_{k+1})\,Q_{\bar{\theta}}(x),
    \end{equation}
    where $\bar{\theta}$ denotes a slow-moving (target-network) copy of the Q-weights for stability.
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\, V_{k+1} \\
        L_Q &= \left(Q_{\theta_k}(x_k) - y_k\right)^2.
    \end{align}
    Note that $h_k$ does not appear in $Q$: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by averaging $Q$ over the candidate set under the current policy:
    \begin{equation}
        V_k = \sum_{x \in S_t} \pi_{\theta_k}(x \mid h_k)\,Q_{\theta_k}(x).
    \end{equation}
    This is feasible because $W_Q$ is a single linear projection applied to the (already computed or cheaply obtainable) last-token activations of each candidate.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k)
        + \beta\, \nabla_{\theta_k} \mathcal{H}\!\left[\pi_{\theta_k}(\cdot \mid h_k)\right].
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V = \sum_x \pi(x|h)\,Q_\theta(x)$ yields the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is a standard state-action value function $Q(s,a)$, estimating the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    It is implemented as a linear projection of the last-token activation:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    We train $Q$ via the Bellman equation.  After the language-modeling update produces $\theta_{k+1}$, we estimate the next-state value by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\, \operatorname{sg}(V_{k+1}) \\
        L_Q &= \left(Q_{\theta_k}(x_k) - y_k\right)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient (detach): we do not back-propagate through $\theta_{k+1}$ or the sampled $Q$ values in the target, but we \emph{do} back-propagate through $W_Q$ on the left-hand side $Q_{\theta_k}(x_k)$.
    Note that $h_k$ does not appear in $Q$: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V = \sum_x \pi(x|h)\,Q_\theta(x)$ yields the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now let me verify the final state of the algorithm section:

The Architecture section still references the old exact-sum form for V. Let me update that too:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$
    \item Off-the-shelf embedding model $\phi$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is a standard state-action value function $Q(s,a)$, estimating the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    It is implemented as a linear projection of the last-token activation:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    We train $Q$ via the Bellman equation.  After the language-modeling update produces $\theta_{k+1}$, we estimate the next-state value by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\, \operatorname{sg}(V_{k+1}) \\
        L_Q &= \left(Q_{\theta_k}(x_k) - y_k\right)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient (detach): we do not back-propagate through $\theta_{k+1}$ or the sampled $Q$ values in the target, but we \emph{do} back-propagate through $W_Q$ on the left-hand side $Q_{\theta_k}(x_k)$.
    Note that $h_k$ does not appear in $Q$: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Here's a summary of the three changes:

1. **V is now estimated by sampling, not an exact sum.** Both in the Q-function update (V_{k+1}) and the policy-gradient baseline (V_k), we sample M candidates from π and average their Q values:
$$V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}), \quad x^{(j)} \sim \pi_{\theta_k}(\cdot \mid h_k)$$

2. **Target network removed; explicit stop-gradient instead.** The Bellman target uses `sg(·)` (stop-gradient/detach) on V_{k+1} — we don't backprop through θ_{k+1} or the sampled Q values in the target, but we *do* backprop through W_Q on the left-hand side `Q_θ_k(x_k)`.

3. **Entropy term removed.** The policy gradient is now simply:
$$\nabla J = A_k \nabla \log \pi_{\theta_k}(x_k \mid h_k)$$
No β, no entropy bonus. (β also removed from the Setup section.)

---

**User**

Is there anything else in this document that is confusing, or would stop you from writing an implementation?

---

**Cursor**



Let me re-read the full document carefully with fresh eyes, looking for ambiguities, gaps, or implementation blockers.

Here are the things I'd want clarified before writing an implementation. I've grouped them by severity:

---

### Must-resolve (would block or break an implementation)

1. **Which parameters get which gradients?** θ is shared across the LM, the policy (via z → μ, σ), and the Q-head (via u → W_Q). In a single inner step we compute three different losses/gradients:
   - LM update: `θ += α ∇ log p_θ(x_k)` (step 3)
   - Q loss: `L_Q = (Q_θ(x_k) - y_k)²` (step 5)
   - Policy gradient: `A_k ∇ log π_θ(x_k|h_k)` (step 6)
   
   Are these applied sequentially to the same θ within one inner step (i.e., three successive gradient updates)? Or are the Q and policy gradients accumulated and applied as one update after the LM step? And critically — does the LM gradient flow through the full backbone, or only through the LM head? If all three update the shared backbone, they could fight each other.

2. **What is `f_θ([start])` concretely?** The transformer takes a *sequence* as input. `[start]` is a single token. So `z_k` is the hidden state from a forward pass on just a `[start]` token? That means the policy has no context about what it's seen before — it's essentially memoryless across inner steps (all "memory" is in the weights θ_k). This is probably intentional given the Markovian framing, but worth being explicit about.

3. **What is `u_k` (the Q activation) exactly?** The doc says "last-token activation of x_k in context." Does this mean we do a *separate* forward pass of the transformer on x_k to get u_k? That's a second forward pass per inner step (one for z_k via `[start]`, one for u_k via x_k). If so, that's fine but should be stated. Also — is this the same forward pass we use for the LM loss `log p_θ(x_k)`, with u_k just being a byproduct?

4. **Dimensionality of h vs. φ(x).** The selection is `softmax(h^T E_S)`. This requires h ∈ ℝ^d and φ(x) ∈ ℝ^d to share the same dimensionality. But the transformer hidden dim and the off-the-shelf embedding model dim are almost certainly different. Is there a learned projection somewhere to align them? (e.g., W_μ projects z into the embedding space of φ?)

5. **How many inner steps K per outer step?** The doc says we "run inner-loop optimization steps" but doesn't specify K. This is a key hyperparameter — too few and Q can't learn, too many and S_t gets stale.

6. **Sampling with vs. without replacement.** When we sample x_k from S_t in the inner loop, is it with replacement? If without, the candidate set shrinks each step and the policy distribution changes. If with, we might train on the same example twice.

---

### Should-clarify (would require guessing in implementation)

7. **The reward `log p_θ(D_t)` — is that summed or averaged over D_t?** With ~100 context windows, the raw sum could be enormous. Presumably mean log-prob?

8. **Learning rates.** There's α for the LM step. What about the Q loss and the policy gradient — same learning rate, or separate ones?

9. **The V_{k+1} computation for the Bellman target requires running the policy at θ_{k+1}.** That means after the LM step we need to: (a) compute z_{k+1} = f_{θ_{k+1}}([start]), (b) get μ_{k+1}, σ_{k+1}, sample h_{k+1}, (c) sample M candidates from π_{θ_{k+1}}, (d) run each through the transformer to get their u's, (e) compute Q on each. That's M+1 extra forward passes *just for the Bellman target*. This is expensive — is it intentional? Or should we reuse the V_k from the policy-gradient step as an approximation (i.e., V_{k+1} ≈ V_{k+1} computed at the *next* inner step, making it one-step-delayed)?

10. **`∇_θ log π_θ(x_k|h_k)` — gradient through what?** π depends on θ only through z → μ, σ → h → softmax(h^T E_S). But h was *sampled* from N(μ, σ²). To get gradients through the sampling, you need the reparameterization trick (h = μ + σ ⊙ ε). This is probably assumed but worth noting.

11. **Line 146 still references "entropy regularization"** in the Analysis section, but we removed the entropy term from the objective.

---

Want me to go ahead and fix #11 (the stale entropy reference) now, and would you like to add clarifying notes for any of the others?

---

**User**

Please write u_k as a wunction of the weights explicitly (like z_k)

In 1/2, we should only we sampling once -- as in mu_k and sigma parametrize a diagonal gaussian, and that produces a distribution over the embeddings.



1. LM conceptually is theta' = theta + alpha Grda_theta log p_theta(x_k)

The idea that the weight updates are actually in the inner step, and just give rise to a reward. The trick is that they actually update our state, and we use theta' going forward

2. Second to last residual stream layer over <s>

3. Do it from the previous weights -- as in you can use the activation collected in (7)

4. Yeah, we will need to pick W_mu to have the dimension of the chosen embedding model

5. Doesn't 2.2 answer this?

6. With replacement

7. Average

8. same lr

9. As long as we just do a batch worth, I think it is fine.

10. My note at the beginning of this message should fix this 

11. Remove from analysis

---

**Cursor**



Let me work through all of these systematically.

Now I have a clear picture of all the changes. Let me apply them systematically.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data. Then we run inner-loop optimization steps on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector:}
    \begin{align}
        z_k &= f_{\theta_k}(\texttt{[start]}) \\
        \mu_k &= W_\mu z_k \\
        \log \sigma_k^2 &= W_\gamma z_k \\
        h_k &\sim \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))
    \end{align}
    
    \item \textbf{Select from candidates:}
    \begin{align}
        \pi_{\theta_k}(x|h_k) &= \text{softmax}(h_k^\top E_{S_t}) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot|h_k)
    \end{align}
    
    \item \textbf{Language modeling update:}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha \nabla_{\theta_k} \log p_{\theta_k}(x_k)
    \end{equation}
    
    \item \textbf{Reward:}
    \begin{equation}
        r_k = \log p_{\theta_{k+1}}(D_t) - \log p_{\theta_k}(D_t)
    \end{equation}
    In practice, store $\log p_{\theta_k}(D_t)$ before the update and recompute only $\log p_{\theta_{k+1}}(D_t)$ after, so we avoid keeping both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.
    
    \item \textbf{Q-function update:}
    The state in this MDP is the current weight vector $\theta_k$ (accessed implicitly---the transformer parameterised by $\theta_k$ produces the activation $u_k$) and the action is the selected example $x_k$.
    The Q-function is a standard state-action value function $Q(s,a)$, estimating the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    It is implemented as a linear projection of the last-token activation:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    We train $Q$ via the Bellman equation.  After the language-modeling update produces $\theta_{k+1}$, we estimate the next-state value by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\, \operatorname{sg}(V_{k+1}) \\
        L_Q &= \left(Q_{\theta_k}(x_k) - y_k\right)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient (detach): we do not back-propagate through $\theta_{k+1}$ or the sampled $Q$ values in the target, but we \emph{do} back-propagate through $W_Q$ on the left-hand side $Q_{\theta_k}(x_k)$.
    Note that $h_k$ does not appear in $Q$: it already determined the choice of $x_k$, so $Q$ need only assess the selected example itself.
    
    \item \textbf{Policy-gradient update:}
    We compute a state-value baseline by sampling a batch of candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are then:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now rewrite the algorithm section — this is the big one. I'll do steps 1–2 (selection), then 3–4 (LM + reward), then 5–6 (Q + policy gradient).

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

All three losses (language-modeling, $L_Q$, and policy gradient) share the same learning rate $\alpha$ and update the shared parameters $\theta$ (including the transformer backbone and the projection heads $W_\mu, W_\gamma, W_Q$).

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} (start) token under the current weights $\theta_k$.
    The policy parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi}\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    We draw a single sample via the reparameterisation trick ($\varepsilon \sim \mathcal{N}(0,I)$):
    \begin{equation}
        h_k = \mu_k + \sigma_k \odot \varepsilon \;\in\; \mathbb{R}^{d_\phi}.
    \end{equation}
    Because $\mu_k$ and $\sigma_k$ live in the same space as the embeddings $\phi(x)$, the vector $h_k$ directly parameterises a distribution over candidates (step~2).

    \item \textbf{Select from candidates.}
    \begin{align}
        \pi_{\theta_k}(x \mid h_k) &= \operatorname{softmax}\!\bigl(h_k^\top E_{S_t}\bigr) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot \mid h_k).
    \end{align}
    Sampling is with replacement from $S_t$.

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the ``action'' $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}
    In practice we cache $\frac{1}{|D_t|}\log p_{\theta_k}(D_t)$ before the update and compute only $\frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t)$ after, so we never need both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.

    \item \textbf{Q-function update.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We reuse the forward pass that computed $\log p_{\theta_k}(x_k)$ in step~3: let $u_k = f_{\theta_k}^{(L-1)}(x_k^{\text{last}})$ be the second-to-last residual-stream activation at the last token of $x_k$.  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    To form the Bellman target we need the next-state value.  We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}) \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    \item \textbf{Policy-gradient update.}
    We compute a state-value baseline by sampling $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection applied to already-available activations, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
    Gradients flow through the reparameterised $h_k = \mu_k + \sigma_k \odot \varepsilon$ into $W_\mu$, $W_\gamma$, and the backbone.
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now update the Architecture section and add the Data Streaming label, then fix the stale entropy reference.

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}\label{sec:streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

All three losses (language-modeling, $L_Q$, and policy gradient) share the same learning rate $\alpha$ and update the shared parameters $\theta$ (including the transformer backbone and the projection heads $W_\mu, W_\gamma, W_Q$).

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} (start) token under the current weights $\theta_k$.
    The policy parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi}\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    We draw a single sample via the reparameterisation trick ($\varepsilon \sim \mathcal{N}(0,I)$):
    \begin{equation}
        h_k = \mu_k + \sigma_k \odot \varepsilon \;\in\; \mathbb{R}^{d_\phi}.
    \end{equation}
    Because $\mu_k$ and $\sigma_k$ live in the same space as the embeddings $\phi(x)$, the vector $h_k$ directly parameterises a distribution over candidates (step~2).

    \item \textbf{Select from candidates.}
    \begin{align}
        \pi_{\theta_k}(x \mid h_k) &= \operatorname{softmax}\!\bigl(h_k^\top E_{S_t}\bigr) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot \mid h_k).
    \end{align}
    Sampling is with replacement from $S_t$.

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the ``action'' $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}
    In practice we cache $\frac{1}{|D_t|}\log p_{\theta_k}(D_t)$ before the update and compute only $\frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t)$ after, so we never need both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.

    \item \textbf{Q-function update.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We reuse the forward pass that computed $\log p_{\theta_k}(x_k)$ in step~3: let $u_k = f_{\theta_k}^{(L-1)}(x_k^{\text{last}})$ be the second-to-last residual-stream activation at the last token of $x_k$.  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    To form the Bellman target we need the next-state value.  We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}) \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    \item \textbf{Policy-gradient update.}
    We compute a state-value baseline by sampling $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection applied to already-available activations, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
    Gradients flow through the reparameterised $h_k = \mu_k + \sigma_k \odot \varepsilon$ into $W_\mu$, $W_\gamma$, and the backbone.
\end{enumerate}

\subsection{Architecture}

The transformer computes a representation $z = f_\theta(\texttt{[start]})$. The policy uses two projections:

\begin{itemize}[nosep]
    \item \textbf{Policy head:} $\mu = W_\mu z$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma z$
\end{itemize}

For value estimation, the Q-function is a standard state-action value function: the state is the current weight vector $\theta$ (encoded implicitly through the transformer's computation) and the action is the selected context window $x$.  Concretely, $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation.  A state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}\label{sec:streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

All three losses (language-modeling, $L_Q$, and policy gradient) share the same learning rate $\alpha$ and update the shared parameters $\theta$ (including the transformer backbone and the projection heads $W_\mu, W_\gamma, W_Q$).

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} (start) token under the current weights $\theta_k$.
    The policy parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi}\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    We draw a single sample via the reparameterisation trick ($\varepsilon \sim \mathcal{N}(0,I)$):
    \begin{equation}
        h_k = \mu_k + \sigma_k \odot \varepsilon \;\in\; \mathbb{R}^{d_\phi}.
    \end{equation}
    Because $\mu_k$ and $\sigma_k$ live in the same space as the embeddings $\phi(x)$, the vector $h_k$ directly parameterises a distribution over candidates (step~2).

    \item \textbf{Select from candidates.}
    \begin{align}
        \pi_{\theta_k}(x \mid h_k) &= \operatorname{softmax}\!\bigl(h_k^\top E_{S_t}\bigr) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot \mid h_k).
    \end{align}
    Sampling is with replacement from $S_t$.

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the ``action'' $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}
    In practice we cache $\frac{1}{|D_t|}\log p_{\theta_k}(D_t)$ before the update and compute only $\frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t)$ after, so we never need both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.

    \item \textbf{Q-function update.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We reuse the forward pass that computed $\log p_{\theta_k}(x_k)$ in step~3: let $u_k = f_{\theta_k}^{(L-1)}(x_k^{\text{last}})$ be the second-to-last residual-stream activation at the last token of $x_k$.  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    To form the Bellman target we need the next-state value.  We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}) \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    \item \textbf{Policy-gradient update.}
    We compute a state-value baseline by sampling $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection applied to already-available activations, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
    Gradients flow through the reparameterised $h_k = \mu_k + \sigma_k \odot \varepsilon$ into $W_\mu$, $W_\gamma$, and the backbone.
\end{enumerate}

\subsection{Architecture}

Let $L$ be the number of transformer layers.  All activations used by the policy and Q-function are read from the second-to-last residual-stream layer ($\ell = L-1$).

\begin{itemize}[nosep]
    \item \textbf{Policy state:} $z = f_\theta^{(L-1)}(\texttt{<s>}) \in \mathbb{R}^{d}$, the activation at the start token.
    \item \textbf{Policy head:} $\mu = W_\mu\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Q head:} $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, where $u_x = f_\theta^{(L-1)}(x^{\text{last}})$ is the activation at the last token of context window $x$.  This activation is obtained as a byproduct of the forward pass used for the language-modeling loss.
\end{itemize}

The state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve? Does entropy regularization produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now remove the stale entropy reference from Analysis:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}\label{sec:streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

All three losses (language-modeling, $L_Q$, and policy gradient) share the same learning rate $\alpha$ and update the shared parameters $\theta$ (including the transformer backbone and the projection heads $W_\mu, W_\gamma, W_Q$).

\begin{enumerate}[nosep]
    \item \textbf{Compute selection vector.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} (start) token under the current weights $\theta_k$.
    The policy parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi}\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    We draw a single sample via the reparameterisation trick ($\varepsilon \sim \mathcal{N}(0,I)$):
    \begin{equation}
        h_k = \mu_k + \sigma_k \odot \varepsilon \;\in\; \mathbb{R}^{d_\phi}.
    \end{equation}
    Because $\mu_k$ and $\sigma_k$ live in the same space as the embeddings $\phi(x)$, the vector $h_k$ directly parameterises a distribution over candidates (step~2).

    \item \textbf{Select from candidates.}
    \begin{align}
        \pi_{\theta_k}(x \mid h_k) &= \operatorname{softmax}\!\bigl(h_k^\top E_{S_t}\bigr) \quad \text{where } E_{S_t} = [\phi(x)]_{x \in S_t} \\
        x_k &\sim \pi_{\theta_k}(\cdot \mid h_k).
    \end{align}
    Sampling is with replacement from $S_t$.

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the ``action'' $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}
    In practice we cache $\frac{1}{|D_t|}\log p_{\theta_k}(D_t)$ before the update and compute only $\frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t)$ after, so we never need both $\theta_k$ and $\theta_{k+1}$ in memory simultaneously.

    \item \textbf{Q-function update.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We reuse the forward pass that computed $\log p_{\theta_k}(x_k)$ in step~3: let $u_k = f_{\theta_k}^{(L-1)}(x_k^{\text{last}})$ be the second-to-last residual-stream activation at the last token of $x_k$.  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}
    To form the Bellman target we need the next-state value.  We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}(\cdot \mid h_{k+1})$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}
    The Bellman target and regression loss are:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}) \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    \item \textbf{Policy-gradient update.}
    We compute a state-value baseline by sampling $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}(\cdot \mid h_k)$ from $S_t$:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection applied to already-available activations, evaluating $Q$ on each sampled candidate is cheap.
    The advantage and policy gradient are:
    \begin{align}
        A_k &= Q_{\theta_k}(x_k) - V_k \\
        \nabla_{\theta_k} J_\pi &= A_k\;\nabla_{\theta_k} \log \pi_{\theta_k}(x_k \mid h_k).
    \end{align}
    Gradients flow through the reparameterised $h_k = \mu_k + \sigma_k \odot \varepsilon$ into $W_\mu$, $W_\gamma$, and the backbone.
\end{enumerate}

\subsection{Architecture}

Let $L$ be the number of transformer layers.  All activations used by the policy and Q-function are read from the second-to-last residual-stream layer ($\ell = L-1$).

\begin{itemize}[nosep]
    \item \textbf{Policy state:} $z = f_\theta^{(L-1)}(\texttt{<s>}) \in \mathbb{R}^{d}$, the activation at the start token.
    \item \textbf{Policy head:} $\mu = W_\mu\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Q head:} $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, where $u_x = f_\theta^{(L-1)}(x^{\text{last}})$ is the activation at the last token of context window $x$.  This activation is obtained as a byproduct of the forward pass used for the language-modeling loss.
\end{itemize}

The state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve over training? Does the learned variance produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Let me do a final read-through to make sure everything is consistent:

Everything looks clean and consistent. Here's a summary of all the changes:

### Changes made

1. **`u_k` written explicitly** — `u_k = f_{\theta_k}^{(L-1)}(x_k^{\text{last}})`, the second-to-last residual-stream activation at the last token of x_k. Noted that this is reused from the forward pass that computes `log p_θ(x_k)` in step 3.

2. **Single sample, reparameterisation trick** — Step 1 now explicitly draws one sample `h_k = μ_k + σ_k ⊙ ε` and explains that because μ, σ live in ℝ^{d_φ}, h_k directly parameterises a distribution over candidates.

3. **LM step = state transition** — Step 3 is labeled "Language-modeling update (state transition)" with text: "the action x_k determines how the weights change, and θ_{k+1} becomes the state for the next inner step."

4. **`z_k` = second-to-last layer over `<s>`** — `z_k = f_{\theta_k}^{(L-1)}(<s>)`, with the layer notation `f_θ^{(ℓ)}` introduced in Setup.

5. **`u_k` reuses the LM forward pass** — "We reuse the forward pass that computed `log p_θ(x_k)` in step 3."

6. **W_μ, W_γ dimensions** — Setup now says `W_μ, W_γ ∈ ℝ^{d_φ × d}` (output dim matches φ).

7. **With replacement, average log-prob, shared lr** — All stated explicitly: "sampled with replacement," reward uses `1/|D_t| log p(D_t)`, and "all three losses share the same learning rate α."

8. **Entropy term removed** from the policy gradient and the Analysis section's "entropy regularization" wording replaced with "learned variance."

---

**User**

Is there anything else in this document that is confusing, or would stop you from writing an implementation?

---

**Cursor**



Let me re-read the full document one more time with implementation-focused eyes.

I think the document is now quite clear. There are only a few small things I'd want to nail down, but none are blockers — they're design choices I could make reasonable defaults for and expose as config:

1. **Step 1 requires a dedicated forward pass on just `<s>`.** To get `z_k = f_θ^{(L-1)}(<s>)` we run the transformer on a single start token. That's a separate forward pass from the one in step 3 (which runs on the full context window `x_k`). This is clear from the doc — just flagging that it's 2 forward passes minimum per inner step (plus the M candidates for V), so the implementation will need to be mindful of batching.

2. **Step 5 says "reuse the forward pass that computed `log p_θ(x_k)` in step 3."** But step 3 is the LM *gradient* update — it computes `∇_θ log p_θ(x_k)`, which requires a forward+backward pass. The `u_k` activation would need to be saved from that forward pass before the backward pass frees it. In PyTorch this means registering a forward hook on the (L-1)-th layer to stash `u_k`, or using `retain_grad`. Straightforward, just an implementation detail.

3. **The ordering of gradient applications within one inner step.** The doc lists steps 3→4→5→6 sequentially. In implementation, step 3 (LM update) must happen first since it defines θ_{k+1} and r_k. But L_Q and J_π both depend on θ_k (the *pre-update* weights). So the natural implementation is: (a) forward pass on `<s>` → get z_k, h_k, select x_k; (b) forward pass on x_k → get log p_θ(x_k) and stash u_k; (c) compute Q_θ(x_k) = W_Q u_k and V_k from M samples; (d) do the LM gradient step → θ_{k+1}; (e) compute r_k from cached vs new held-out log-prob; (f) compute V_{k+1} from M samples under θ_{k+1}; (g) form Bellman target and compute L_Q, A_k gradients *with respect to the old θ_k parameters* — but wait, we've already updated to θ_{k+1}. This means in practice we'd need to either accumulate the L_Q and policy gradients *before* the LM step and apply all three together, or do the LM step last. Could you confirm: **should all three gradients be computed at θ_k and applied as one combined update**, rather than sequentially?

That's the only real ambiguity I see. Everything else — the data source, embedding model choice, M, K, γ, α — are hyperparameters I'd just put in a config file. I'm ready to implement whenever you want.

---

**User**

In compute selection vector and select from candidates, mu_k and log sigma_k^2, we already have a perfectly valid distribution over the embedding space, which can be normalized over the specific set S_t to be a probability distribution. There is no reason to sample twice

1. That's fine
2. Let's recompute the p(x_k) after updating theta then
3. Let's just keep the old and new weights in memory

Can you write the final loss including the regression loss and the PG update loss

---

**Cursor**



You're right — the Gaussian in embedding space already defines a distribution over S_t via softmax(h^T E_S), and then we sample x_k from that. There's no second sampling step needed. Let me also think through the ordering carefully now.

The key insight for point 2: we compute `log p_θ(x_k)` *after* the LM update (i.e., at θ_{k+1}) to get u_k as a byproduct of the reward computation, not from a separate pre-update forward pass.

Let me rewrite the algorithm and add an explicit final loss.

Now let me also check the architecture section:

Now let me rewrite the algorithm section. The key changes:
- Steps 1+2 merged: the Gaussian directly induces a distribution over S_t, we sample x_k once
- u_k comes from the θ_{k+1} forward pass (the one that computes the post-update held-out log-prob)
- We keep both θ_k and θ_{k+1} in memory
- Explicit combined loss at the end

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}\label{sec:streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Select a training example.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} token under the current weights $\theta_k$.
    The Gaussian parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi},\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    This Gaussian $\mathcal{N}(\mu_k, \operatorname{diag}(\sigma_k^2))$ over the embedding space induces a distribution over $S_t$ by normalising the density at each candidate's embedding:
    \begin{equation}
        \pi_{\theta_k}(x) = \frac{\mathcal{N}\!\bigl(\phi(x);\;\mu_k,\;\operatorname{diag}(\sigma_k^2)\bigr)}
        {\sum_{x' \in S_t} \mathcal{N}\!\bigl(\phi(x');\;\mu_k,\;\operatorname{diag}(\sigma_k^2)\bigr)},
        \qquad x \in S_t.
    \end{equation}
    We sample a single example $x_k \sim \pi_{\theta_k}$ (with replacement from $S_t$).

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the action $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.  We retain both $\theta_k$ and $\theta_{k+1}$ in memory.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}

    \item \textbf{Q-value of selected example.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We obtain $u_k$ as a byproduct of the forward pass on $x_k$ under $\theta_{k+1}$ used to compute $\log p_{\theta_{k+1}}(D_t)$ in step~3 (specifically, $u_k = f_{\theta_{k+1}}^{(L-1)}(x_k^{\text{last}})$, the second-to-last residual-stream activation at the last token of $x_k$).  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}

    \item \textbf{Next-state value (Bellman target).}
    We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}

    \item \textbf{Baseline value.}
    We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}$ from $S_t$ and estimate:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection, evaluating $Q$ on each sampled candidate is cheap.

    \item \textbf{Combined loss and update.}
    The Bellman regression loss for the Q-function is:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}), \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    The policy-gradient loss uses the advantage $A_k = \operatorname{sg}\!\bigl(Q_{\theta_k}(x_k) - V_k\bigr)$:
    \begin{equation}
        L_\pi = -A_k \log \pi_{\theta_k}(x_k).
    \end{equation}
    Gradients of $L_\pi$ flow through $\pi_{\theta_k}(x_k)$ into $W_\mu$, $W_\gamma$, and the backbone (via $z_k$).

    The total loss for one inner step is:
    \begin{equation}
    \boxed{\;L_k = L_Q + L_\pi\;}
    \end{equation}
    We compute $\nabla_{\theta_k} L_k$ and apply a single gradient step with learning rate $\alpha$, updating the shared parameters $\theta$ (backbone, $W_\mu$, $W_\gamma$, $W_Q$).  The language-modeling update (step~2) is applied separately as the MDP state transition.
\end{enumerate}

\subsection{Architecture}

Let $L$ be the number of transformer layers.  All activations used by the policy and Q-function are read from the second-to-last residual-stream layer ($\ell = L-1$).

\begin{itemize}[nosep]
    \item \textbf{Policy state:} $z = f_\theta^{(L-1)}(\texttt{<s>}) \in \mathbb{R}^{d}$, the activation at the start token.
    \item \textbf{Policy head:} $\mu = W_\mu\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Variance head:} $\log \sigma^2 = W_\gamma\, z \in \mathbb{R}^{d_\phi}$
    \item \textbf{Q head:} $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, where $u_x = f_\theta^{(L-1)}(x^{\text{last}})$ is the activation at the last token of context window $x$.  This activation is obtained as a byproduct of the forward pass used for the language-modeling loss.
\end{itemize}

The state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve over training? Does the learned variance produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Now update the Architecture section to match — remove the "byproduct of LM loss" language and reflect the new policy formulation:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}

\title{\textbf{Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models}}
\author{Scott Viteri}
\date{}

\begin{document}
\maketitle

\section{Overview}

We propose an algorithm that enables language models to learn what data to learn from. Rather than training on randomly sampled data, the model takes actions that select training examples, receiving reward based on how much each selection improves prediction on held-out data. The core hypothesis is that learned curriculum selection will yield significantly better sample efficiency than random or heuristic curricula.

\section{Technical Approach}

\subsection{Setup}

\begin{itemize}[nosep]
    \item Transformer with parameters $\theta$ and $L$ layers; write $f_\theta^{(\ell)}(x)$ for the residual-stream activation at layer $\ell$ and position of token $x$
    \item Off-the-shelf embedding model $\phi \colon \text{text} \to \mathbb{R}^{d_\phi}$ for encoding streaming data
    \item Streaming data source, partitioned into training candidates and held-out evaluation
    \item Policy projections $W_\mu, W_\gamma \in \mathbb{R}^{d_\phi \times d}$ (output dimension matches $\phi$)
    \item Learned state-dependent log-variance $\log \sigma^2 = W_\gamma z$
    \item Discount factor $\gamma$; shared learning rate $\alpha$
    \item Scalar Q-function $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, a linear projection of the last-token activation of $x$
\end{itemize}

\subsection{Data Streaming}\label{sec:streaming}

Data streams in continuously. For approximately every 1000 data points that arrive:
\begin{itemize}[nosep]
    \item $\sim$900 become training candidates (the set $S_t$ at outer step $t$)
    \item $\sim$100 are reserved for held-out evaluation (the set $D_t$ at outer step $t$)
\end{itemize}

Each incoming data point $x$ is embedded via an off-the-shelf embedding model: $e_x = \phi(x)$. These embeddings are used for the selection mechanism described below.

Each element of $S_t$ or $D_t$ is conceptually one full transformer context window of text.

\subsection{Algorithm}

At each outer-loop stream step $t$, we refresh $S_t, D_t$ from newly arrived data.  Candidates are sampled with replacement, so $|S_t|$ stays fixed throughout the inner loop.  We then run $K$ inner-loop steps (one per data point in Section~\ref{sec:streaming}) on fixed $S_t, D_t$ to update $\theta$.

\begin{enumerate}[nosep]
    \item \textbf{Select a training example.}
    Let $z_k = f_{\theta_k}^{(L-1)}(\texttt{<s>})$ be the second-to-last residual-stream activation at the \texttt{<s>} token under the current weights $\theta_k$.
    The Gaussian parameters in embedding space are:
    \begin{align}
        \mu_k &= W_\mu\, z_k \;\in\; \mathbb{R}^{d_\phi},\\
        \log \sigma_k^2 &= W_\gamma\, z_k \;\in\; \mathbb{R}^{d_\phi}.
    \end{align}
    This Gaussian $\mathcal{N}(\mu_k, \operatorname{diag}(\sigma_k^2))$ over the embedding space induces a distribution over $S_t$ by normalising the density at each candidate's embedding:
    \begin{equation}
        \pi_{\theta_k}(x) = \frac{\mathcal{N}\!\bigl(\phi(x);\;\mu_k,\;\operatorname{diag}(\sigma_k^2)\bigr)}
        {\sum_{x' \in S_t} \mathcal{N}\!\bigl(\phi(x');\;\mu_k,\;\operatorname{diag}(\sigma_k^2)\bigr)},
        \qquad x \in S_t.
    \end{equation}
    We sample a single example $x_k \sim \pi_{\theta_k}$ (with replacement from $S_t$).

    \item \textbf{Language-modeling update (state transition).}
    \begin{equation}
        \theta_{k+1} = \theta_k + \alpha\, \nabla_{\theta_k} \log p_{\theta_k}(x_k).
    \end{equation}
    This is the state transition of the MDP: the action $x_k$ determines how the weights change, and $\theta_{k+1}$ becomes the state for the next inner step.  We retain both $\theta_k$ and $\theta_{k+1}$ in memory.

    \item \textbf{Reward.}
    \begin{equation}
        r_k = \frac{1}{|D_t|}\log p_{\theta_{k+1}}(D_t) - \frac{1}{|D_t|}\log p_{\theta_k}(D_t).
    \end{equation}

    \item \textbf{Q-value of selected example.}
    The state in this MDP is $\theta_k$ and the action is $x_k$.
    The Q-function estimates the \emph{discounted future expected return} from selecting $x_k$ in state $\theta_k$ and following the policy thereafter.
    We obtain $u_k$ as a byproduct of the forward pass on $x_k$ under $\theta_{k+1}$ used to compute $\log p_{\theta_{k+1}}(D_t)$ in step~3 (specifically, $u_k = f_{\theta_{k+1}}^{(L-1)}(x_k^{\text{last}})$, the second-to-last residual-stream activation at the last token of $x_k$).  Then:
    \begin{equation}
        Q_{\theta_k}(x_k) = W_Q\, u_k \;\in\; \mathbb{R}.
    \end{equation}

    \item \textbf{Next-state value (Bellman target).}
    We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_{k+1}}$ from $S_t$ and estimate:
    \begin{equation}
        V_{k+1} \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_{k+1}}(x^{(j)}).
    \end{equation}

    \item \textbf{Baseline value.}
    We sample a batch of $M$ candidates $\{x^{(j)}\}_{j=1}^{M} \sim \pi_{\theta_k}$ from $S_t$ and estimate:
    \begin{equation}
        V_k \approx \frac{1}{M}\sum_{j=1}^{M} Q_{\theta_k}(x^{(j)}).
    \end{equation}
    Since $W_Q$ is a single linear projection, evaluating $Q$ on each sampled candidate is cheap.

    \item \textbf{Combined loss and update.}
    The Bellman regression loss for the Q-function is:
    \begin{align}
        y_k &= r_k + \gamma\;\operatorname{sg}(V_{k+1}), \\
        L_Q &= \bigl(Q_{\theta_k}(x_k) - y_k\bigr)^2,
    \end{align}
    where $\operatorname{sg}(\cdot)$ denotes stop-gradient: we do not back-propagate through $\theta_{k+1}$ or the $Q$ values in the target, but we \emph{do} back-propagate into $W_Q$ (and the backbone that produces $u_k$) on the left-hand side.

    The policy-gradient loss uses the advantage $A_k = \operatorname{sg}\!\bigl(Q_{\theta_k}(x_k) - V_k\bigr)$:
    \begin{equation}
        L_\pi = -A_k \log \pi_{\theta_k}(x_k).
    \end{equation}
    Gradients of $L_\pi$ flow through $\pi_{\theta_k}(x_k)$ into $W_\mu$, $W_\gamma$, and the backbone (via $z_k$).

    The total loss for one inner step is:
    \begin{equation}
    \boxed{\;L_k = L_Q + L_\pi\;}
    \end{equation}
    We compute $\nabla_{\theta_k} L_k$ and apply a single gradient step with learning rate $\alpha$, updating the shared parameters $\theta$ (backbone, $W_\mu$, $W_\gamma$, $W_Q$).  The language-modeling update (step~2) is applied separately as the MDP state transition.
\end{enumerate}

\subsection{Architecture}

Let $L$ be the number of transformer layers.  All activations used by the policy and Q-function are read from the second-to-last residual-stream layer ($\ell = L-1$).

\begin{itemize}[nosep]
    \item \textbf{Policy state:} $z = f_\theta^{(L-1)}(\texttt{<s>}) \in \mathbb{R}^{d}$, the activation at the start token.
    \item \textbf{Policy head:} $\mu = W_\mu\, z \in \mathbb{R}^{d_\phi}$, $\log \sigma^2 = W_\gamma\, z \in \mathbb{R}^{d_\phi}$.  The Gaussian $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ in embedding space is normalised over $S_t$ to give $\pi_\theta(x)$.
    \item \textbf{Q head:} $Q_\theta(x) = W_Q\, u_x \in \mathbb{R}$, where $u_x = f_\theta^{(L-1)}(x^{\text{last}})$ is the activation at the last token of context window $x$.
\end{itemize}

The state-value baseline $V$ is estimated by sampling $M$ candidates from $\pi$ and averaging their $Q$ values, yielding the advantage $A = Q_\theta(x) - V$.

\section{Evaluation}

\subsection{Primary Metric: Sample Efficiency}

The central question is: how many training examples does the model need to reach a given performance level?

We will measure:
\begin{itemize}[nosep]
    \item Perplexity on a fixed evaluation set as a function of training examples seen
    \item Performance on downstream tasks (e.g., MMLU, HellaSwag) as a function of training examples
    \item Comparison to human learning efficiency as an aspirational benchmark
\end{itemize}

\subsection{Baselines}

\begin{itemize}[nosep]
    \item \textbf{Random curriculum:} Uniform sampling from candidates
    \item \textbf{Loss-based curriculum:} Prioritize high-loss examples
    \item \textbf{Uncertainty-based curriculum:} Prioritize examples with high model uncertainty
    \item \textbf{Competence-based curriculum:} Examples ordered by difficulty (requires difficulty labels)
\end{itemize}

\subsection{Analysis}

Beyond aggregate metrics, we will examine:
\begin{itemize}[nosep]
    \item \textbf{Curriculum structure:} Does the learned curriculum exhibit interpretable patterns? Developmental stages? Topic clustering?
    \item \textbf{Exploration dynamics:} How does $\sigma$ evolve over training? Does the learned variance produce reasonable exploration?
    \item \textbf{Q-function interpretability:} What does $Q_\theta(x)$ learn to predict? Can we understand what makes a training example ``valuable''?
\end{itemize}

\section{Relation to Prior Work}

\subsection{Curriculum Learning}

Graves et al.\ (2017) use multi-armed bandits to select tasks; Jiang et al.\ (2018) train a separate ``mentor'' network to weight examples. Our approach differs by embedding selection in the model's forward pass and using RL to optimize for held-out improvement directly.

\subsection{Meta-Learning}

MAML (Finn et al., 2017) learns initializations for fast adaptation. We share the computational structure---reasoning about the effect of gradient updates---but learn \emph{what} to train on rather than \emph{where} to start.

\subsection{Intrinsic Motivation}

Our objective relates to compression progress (Schmidhuber) and active inference, but avoids the memory requirements of the former and the dark room problem of the latter by using streaming held-out data as a proxy for predictive success.

\subsection{Connection to My Prior Work}

This proposal builds directly on my recent work on Markovian Transformers for Informative Language Modeling (\url{https://arxiv.org/abs/2404.18988}). That work introduces a framework for training language models to generate Chain-of-Thought reasoning that is \emph{causally load-bearing}---the CoT must contain all information needed to predict the answer, as the model cannot attend back to the original question.

Both projects share a common structure: use RL to learn intermediate representations that improve prediction on held-out data. In the Markovian Transformers work, we learn CoTs:
\[
\text{Question} \to \text{CoT} \to \text{Answer}
\]
In the current proposal, we learn curricula:
\[
\text{Model State} \to \text{Selected Data} \to \text{Improved Predictions}
\]

The Markovian Transformers work demonstrates that this general approach is tractable and yields large gains on reasoning benchmarks (e.g., GSM8K: 19.6\% $\to$ 57.1\%). The current proposal extends this framework from learning \emph{what to say} to learning \emph{what to study}.

\section{Broader Motivation}

Language models trained to predict text develop remarkable capabilities from a simple objective. Yet they require extensive post-training to behave agentically and arguably lack a kind of global coherence. One hypothesis: the training process is purely observational---the model never takes actions that affect what it observes.

This project is a stepping stone toward studying whether learned curriculum selection produces qualitatively different agents. The immediate goal is demonstrating sample efficiency gains. The longer-term question is whether controlling one's own learning process contributes to the coherence and agency that current models seem to lack.

\subsection{Long-Term Direction: Formalizing Homeostasis}

Biological agents are shaped by survival pressures. Hunger, pain, and fatigue are not arbitrary reward signals---they are tied to the organism's continued existence. Current approaches to intrinsic motivation (curiosity, empowerment, compression progress) capture aspects of adaptive behavior but lack this grounding in self-preservation.

A long-term goal of this research program is to formalize homeostasis and survival into a simple, biologically plausible metric that could serve as a foundation for intrinsic motivation in artificial systems. The current project---learning to select data that improves future prediction---is a minimal step in this direction: the agent takes actions that maintain its predictive capacity, a kind of epistemic homeostasis. Understanding what this simple case yields will inform more ambitious formulations.

\section{Timeline and Resources}

\begin{table}[h]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Phase} & \textbf{Duration} \\
\midrule
Infrastructure and baseline implementation & 6 weeks \\
Core algorithm implementation and debugging & 8 weeks \\
Initial experiments and hyperparameter tuning & 6 weeks \\
Scaling experiments on larger models & 4 weeks \\
Analysis and writeup & 4 weeks \\
\midrule
\textbf{Total} & \textbf{6 months} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Compute Resources}

\begin{itemize}[nosep]
    \item \textbf{Initial experiments:} Stanford Sherlock cluster (covered by existing allocation)
    \item \textbf{Final scaling experiments:} Cloud compute (Runpod or similar), approximately \$10K for H100/H200 time to validate results on larger models (1B+ parameters)
\end{itemize}

\section{Expected Outcomes}

\begin{enumerate}[nosep]
    \item Empirical demonstration of sample efficiency gains (or well-documented negative result)
    \item Open-source implementation of the algorithm
    \item Analysis of learned curriculum structure
    \item Foundation for future work on agency, homeostasis, and learned exploration in language models
\end{enumerate}

\section{About the Applicant}

Scott Viteri is a final-year PhD student at Stanford working on machine learning and reinforcement learning, with a focus on continual learning in transformers. His relevant prior work includes:

\begin{itemize}[nosep]
    \item \textbf{Markovian Transformers for Informative Language Modeling} (\url{https://arxiv.org/abs/2404.18988}): Introduces a framework for training language models with RL to produce causally load-bearing Chain-of-Thought reasoning. Demonstrates large gains on reasoning benchmarks (GSM8K, ARC-Challenge, MMLU). This work establishes the technical foundation for the current proposal.
    \item \textbf{Epistemic Phase Transitions in Mathematical Proofs} (Cognition, 2022): Studies how belief formation operates in mathematical reasoning using network models, showing how modular structure and bidirectional inference enable certainty to emerge despite local error rates.
    \item \textbf{ARC Eliciting Latent Knowledge Prize winner}: Recognized for work on methods to extract truthful information from language models.
    \item Experience with category-theoretic approaches to machine learning, collaborating with researchers at the Topos Institute.
\end{itemize}

\end{document}

```

Also update the Setup section — remove the redundant `h_k`-related sampling language:

The Setup looks fine. Let me verify the final algorithm section reads coherently:

Here's a summary of the changes:

### Steps 1+2 merged — no double sampling
The Gaussian N(μ_k, diag(σ_k²)) in embedding space directly induces a categorical distribution over S_t by normalising the density at each candidate's embedding:

$$\pi_{\theta_k}(x) = \frac{\mathcal{N}(\phi(x);\,\mu_k,\,\text{diag}(\sigma_k^2))}{\sum_{x'} \mathcal{N}(\phi(x');\,\mu_k,\,\text{diag}(\sigma_k^2))}$$

One sample x_k, no intermediate h_k.

### u_k from θ_{k+1} forward pass
We get u_k as a byproduct of the forward pass under θ_{k+1} that computes `log p_{θ_{k+1}}(D_t)` for the reward. Both θ_k and θ_{k+1} are kept in memory.

### Explicit combined loss

$$\boxed{L_k = L_Q + L_\pi}$$

where:
- **L_Q** = `(Q_θk(x_k) − y_k)²` with `y_k = r_k + γ sg(V_{k+1})` — backprop into W_Q and backbone only on the LHS
- **L_π** = `−A_k log π_θk(x_k)` with `A_k = sg(Q_θk(x_k) − V_k)` — backprop into W_μ, W_γ, and backbone via z_k

The LM update (step 2) is the MDP state transition, applied separately. The RL loss L_k is a single gradient step on the shared parameters.

---

**User**

Ok, could you go ahead and implement this in pytorch?

Use gpt2 and perhaps openai ada embeddings, using my API key which should be in env vars (also my huggingface api should be as well)

---

**Cursor**



Let me start by understanding the full spec, checking what's already in the repo, and then building the implementation.

Good — both API keys are available. Let me re-read the final algorithm from the tex to make sure my implementation matches exactly, then plan the code.

Now let me check what Python/CUDA environment is available:

No packages installed yet. Let me set up a virtual environment with the needed dependencies.

