# First Use of the Timing Skill

  This note summarizes the first use of the timing skill. The skill is meant to test simple runtime optimization ideas
  under strict constraints. It should know where to search, what parts of the repo define the training protocol, and
  what can be changed safely to reduce runtime without changing the experiment.

## 1. Idea to First Implementation

The first idea was to create a skill that could combine coding knowledge, internet research, and local repo exploration. It was meant to find one runtime idea, test it on Botero, connect useful results to sbatch creation, and run tests before and after each change.

Pros:
- The skill gave a clear workflow for moving from an idea to a first implementation.
- It used both repo knowledge and wider PyTorch/runtime knowledge.
- It connected local testing to Slurm sbatch creation.
- It treated tests as part of the process, not as an afterthought.

Cons:
- Checkpoint saving was not controlled well enough for timing-only work.
- Token usage became high because the skill had to inspect many files, results, and logs.

How to improve:
- Add a timing-only mode that disables or limits checkpoint saving.
- Make the skill summarize previous findings before reading many files again.
- Keep one short decision record per idea so repeated calls use fewer tokens.

Research value:
- This stage is important because it turns a vague optimization idea into a repeatable research method.

## 2. Botero 24-Hour Crontab Step

The Botero step used repeated crontab jobs over about 24 hours. This helped the skill explore many runtime ideas and use the RTX 4090 as a useful first baseline before spending Slurm time.

Pros:
- It allowed steady exploration without constant manual work.
- The RTX 4090 was a good local baseline for quick feedback.
- Failed or weak ideas could be filtered before Slurm.

Cons:
- Some runs created memory pressure or overload.
- Too many result files were created, which made inspection harder.
- The result structure became noisy.

How to improve:
- Add a strict output cleanup or retention rule.
- Save only compact summaries by default, with full logs only for winners or failures.
- Add memory checks before launching repeated jobs.

Research value:
- This stage is useful because it cheaply separates ideas worth deeper testing from ideas that are only interesting on paper.

## 3. Slurm Inspection

The Slurm step made sbatch inspection understandable and easy to use. It helped show how local timing ideas could be moved into the cluster environment.

Pros:
- The generated sbatches were readable.
- It was easier to inspect what each job would run.
- The step connected local evidence to realistic cluster testing.

Cons:
- The Slurm results did not always produce a clear final recommendation.
- The skill did not always say whether the evidence was strong enough for a code change.
- It did not always explain the exact change that should be made next.

How to improve:
- Add a required final recommendation after Slurm inspection.
- The skill should say one of three things: change the code, run full-epoch validation, or reject the idea.
- Each recommendation should include the protocol, baseline, candidate result, speedup, and reason.

Research value:
- This stage matters because local wins are not enough. Slurm inspection checks whether the idea still makes sense on the hardware used for real training.

## 4. Full-Epoch Inspection

The full-epoch inspection was required by the user to verify compile usage. Short timing runs were useful, but they were not enough to prove that compile behavior was correct across a real epoch.

Pros:
- It checked behavior in a more realistic training setting.
- It helped verify whether compile was actually being used.
- It reduced the risk of recommending a change based only on short benchmarks.

Cons:
- Full-epoch runs are expensive.
- They create more logs and outputs.
- They can slow down the research loop if used too early.

How to improve:
- The skill should decide whether stage 3 results are already enough for a code change.
- If stage 3 is enough, the skill should state the exact code or config change to make.
- If stage 3 is not enough, the skill should explain why and name the required full-epoch check.
- For compile ideas, the skill should require evidence that compile was active, not only that runtime changed.

Research value:
- This stage is important because some runtime changes only look good in short tests. Full-epoch inspection checks whether the idea still works in a realistic training path.

## Overall Skill Improvements

The skill should keep the current research flow, but make the decision points sharper. After each stage it should write a short result: what was tested, what happened, whether the evidence is enough, and what should happen next.

The most important improvement is to separate exploration from recommendation. Exploration can create many ideas and files. Recommendation should be small, clear, and strict: either make a specific change, run one specific validation, or do not change the code.
